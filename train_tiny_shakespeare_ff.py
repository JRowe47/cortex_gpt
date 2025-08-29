# train_tiny_shakespeare_ff.py
# ----------------------------------------------------
# Forward-Forward (Hinton) training loop for a 7-region Transformer on Tiny Shakespeare.
# This script is designed to be a drop-in companion to your repository's existing model code.
#
# Key ideas:
#  - Uses layer-local "goodness" objectives instead of global backprop.
#  - For each Transformer block, we do a local update that increases goodness for "positive" inputs
#    (context + correct next token) and decreases goodness for "negative" inputs (context + corrupted token).
#  - We keep the architecture features you built for the project (block-sparse 7-region topology,
#    dendrites, sparsity/KWTA, MFS head, region feedback, etc.).
#
# Notes:
#  - The script assumes NanoGPT-like module names: model.transformer.h is a list of residual blocks.
#    If your repo uses slightly different names, tweak `get_blocks()`.
#  - We keep *per-block* optimizers to ensure layer-local updates.
#  - Embeddings are *optionally* updated via a special "pre-block 0" step if available; see the hook comments.
#  - This is a *clean-room* implementation that does not depend on any ff.py in your repo;
#    it should run with only PyTorch + your model.py/cortex_model.py present.
#
# Usage (example):
#   python train_tiny_shakespeare_ff.py --data_dir data/tiny_shakespeare \
#       --steps 2000 --batch_size 64 --block_size 128 \
#       --n_layer 8 --n_head 6 --n_embd 384 --target_params_m 15.0
#
# Generation (FF scanning) is included as a utility but disabled by default because
# it is slower (requires trying multiple candidate tokens). See `ff_generate` below.
# ----------------------------------------------------

import os
import math
import time
import random
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Model import: try your two common entry points ----
GPT = None
GPTConfig = None
try:
    from model import GPT, GPTConfig  # your project default
except Exception:
    try:
        from cortex_model import GPT, GPTConfig  # alt path in your repo
    except Exception as e:
        raise RuntimeError(
            "Could not import GPT/GPTConfig from model.py or cortex_model.py"
        ) from e


# ---------------------------
# Dataset (byte-level LM data)
# ---------------------------


def maybe_download_shakespeare(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    in_path = os.path.join(data_dir, "input.txt")
    if os.path.exists(in_path):
        return in_path
    # Lightweight safe fallback: write a tiny snippet if offline
    sample = (
        "From fairest creatures we desire increase,\n"
        "That thereby beauty's rose might never die,\n"
        "But as the riper should by time decease,\n"
        "His tender heir might bear his memory:\n"
    )
    with open(in_path, "w", encoding="utf-8") as f:
        f.write(sample)
    return in_path


def load_bytes_dataset(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    path = maybe_download_shakespeare(data_dir)
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    data_bytes = torch.tensor(list(data.encode("utf-8")), dtype=torch.long)
    n = int(0.9 * len(data_bytes))
    train_data = data_bytes[:n]
    val_data = data_bytes[n:]
    return train_data, val_data


@torch.no_grad()
def eval_lm(
    model: nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    steps: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (negative log-likelihood, accuracy) for a byte dataset."""
    model.eval()
    tot_nll = 0.0
    tot_tokens = 0
    tot_correct = 0
    for _ in range(steps):
        max_start = len(data) - block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"data length {len(data)} insufficient for block_size {block_size}"
            )
        ix = torch.randint(0, max_start, (batch_size,), device=device)
        x = torch.stack([data[i : i + block_size] for i in ix], dim=0)
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix], dim=0)
        logits, _ = model(x, targets=None)
        logp = F.log_softmax(logits, dim=-1)
        V = logp.size(-1)
        nll = F.nll_loss(
            logp.view(-1, V), y.view(-1), ignore_index=-1, reduction="mean"
        )
        preds = logp.argmax(dim=-1)
        tot_correct += (preds == y).sum().item()
        tot_tokens += y.numel()
        tot_nll += float(nll)
    avg_nll = tot_nll / max(1, steps)
    acc = tot_correct / max(1, tot_tokens)
    return avg_nll, acc


def bpc_from_nll(nll: float) -> float:
    return nll / math.log(2.0)


def ppl_from_nll(nll: float) -> float:
    return math.exp(nll)


def get_batch_bytes(
    split_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    posneg_negatives: int = 1,
):
    """
    Returns (pos_seq, neg_seq), each LongTensor of shape [B, T]
      - Build sequences where the last token is the "candidate next token".
      - Positive = correct next token for the context
      - Negative = corrupted candidate(s) (uniformly sampled from [0..255] != y_true)
    """
    B = batch_size
    T = block_size
    assert T >= 2, "block_size must be >= 2"
    # We use T-1 for context and require enough data to sample a full window.
    max_start = len(split_data) - T + 1  # inclusive range of valid start positions
    if max_start <= 0:
        raise ValueError(
            f"split_data length {len(split_data)} is insufficient for block_size {T}."
            " Reduce block_size or provide more data."
        )
    ix = torch.randint(low=0, high=max_start, size=(B,), device=device)
    ctx = torch.stack([split_data[i : i + (T - 1)] for i in ix], dim=0)  # [B, T-1]
    y_true = torch.stack([split_data[i + (T - 1)] for i in ix], dim=0)  # [B]

    pos_seq = torch.cat([ctx, y_true.unsqueeze(1)], dim=1)  # [B, T]

    if posneg_negatives <= 0:
        return pos_seq, None

    # Build K negatives for each sample by corrupting the final token
    negs = []
    for _ in range(posneg_negatives):
        rand = torch.randint(0, 256, (B,), device=device, dtype=torch.long)
        conflict = rand == y_true
        # ensure negative differs from positive
        if conflict.any():
            rand[conflict] = (rand[conflict] + 1) % 256
        negs.append(torch.cat([ctx, rand.unsqueeze(1)], dim=1))
    neg_seq = torch.stack(negs, dim=1)  # [B, K, T]
    return pos_seq, neg_seq


# ---------------------------
# Forward-Forward utilities
# ---------------------------


def get_blocks(model: nn.Module) -> List[nn.Module]:
    """
    Try to extract the residual blocks list in a NanoGPT-like model.
    Adjust here if your repo uses different names.
    """
    # common: model.transformer.h -> ModuleList
    node = model
    for attr in ("transformer", "h"):
        if hasattr(node, attr):
            node = getattr(node, attr)
        else:
            node = None
            break
    if isinstance(node, nn.ModuleList):
        return list(node)

    # fallback: look for the first ModuleList under model named 'h' or 'blocks' or 'layers'
    for name in ["h", "blocks", "layers"]:
        if hasattr(model, name) and isinstance(getattr(model, name), nn.ModuleList):
            return list(getattr(model, name))

    # last resort: collect child blocks that have a forward(x)->x signature
    # (order not guaranteed; better to adapt your model to expose .transformer.h)
    blocks = []
    for mod in model.modules():
        if hasattr(mod, "forward") and mod is not model:
            # naive heuristic: block outputs same shape as input; we can't check shapes here
            blocks.append(mod)
    if not blocks:
        raise RuntimeError(
            "Could not find model blocks. Please expose model.transformer.h or adjust get_blocks()."
        )
    return blocks


@torch.no_grad()
def snapshot_block_inputs(
    model: nn.Module, x: torch.Tensor, blocks: List[nn.Module]
) -> List[torch.Tensor]:
    """
    Run a forward pass, capturing the *inputs* to each block.
    We use forward_pre_hooks to record the tensors fed into each residual block.
    """
    cached_inputs: List[Optional[torch.Tensor]] = [None for _ in blocks]

    handles = []

    def make_pre_hook(idx):
        def hook(mod, mod_input):
            # mod_input is a tuple; take first element (x)
            cached_inputs[idx] = mod_input[0].detach()

        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_pre_hook(make_pre_hook(i)))
    # run a full forward to trigger hooks
    _ = model(x)
    # clean up hooks
    for h in handles:
        h.remove()

    # assert we got all inputs
    for i, t in enumerate(cached_inputs):
        if t is None:
            raise RuntimeError(
                f"Did not capture input for block {i}. "
                f"Adjust hook logic for your model."
            )
    return cached_inputs  # list of [B, T, C] tensors


def layer_goodness(
    x: torch.Tensor, token_index: int = -1, eps: float = 1e-6
) -> torch.Tensor:
    """
    Hinton-style 'goodness' = mean of squared activations (after nonlinearity).
    We take the goodness at a particular token position (default: last token).
    x: [B, T, C]
    returns: [B] goodness per sample
    """
    # If the model uses LN + residuals, post-block activations should already be after nonlinearity.
    # We'll compute goodness on the block *output* (or recomputation output).
    x_last = x[:, token_index, :]  # [B, C]
    g = (x_last**2).mean(dim=1)  # [B]
    # numerical stability
    return g + eps


def ff_binary_loss(
    goodness: torch.Tensor, y_posneg: torch.Tensor, theta: float = 2.0
) -> torch.Tensor:
    """
    Logistic loss on goodness values with a margin/threshold theta.
    y_posneg: +1 for positive, -1 for negative, shape [N]
    goodness: shape [N]
    """
    # loss = log(1 + exp(-y * (g - theta)))
    return F.softplus(-y_posneg * (goodness - theta)).mean()


# ---------------------------
# Build model with ~15M params and 7-region topology
# ---------------------------


def build_7region_15M_model(
    block_size: int,
    vocab_size: int = 256,
    n_layer: int = 8,
    n_head: int = 6,
    n_embd: int = 384,
    **extra_cfg,
):
    """
    Defaults chosen to land near ~15M params for a GPT-like arch:
      per-block params approx ~12 * n_embd^2; with n_embd=384, n_layer=8 -> ~14.16M + embeddings & heads.
    We enable the project features by default (sparsity, dendrites, block-sparse 7-region attention, etc.).
    """
    cfg = dict(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=True,
        # --- HTM / sparsity / dendrites ---
        kwta_frac=0.20,  # turn on sparsity (20% winners)
        use_dendrites=True,
        dendrite_segments=8,
        # --- ROPE / FEROPE anchors ---
        ff_mode="sumsq",
        ferope_anchor_relative=True,
        anchor_min_gap=32,
        anchor_jitter=8,
        ferope_m=32,
        rope_base=10000.0,
        # --- 7-region local attention ---
        use_block_sparse=True,
        neighbor_rings=1,  # 1 ring -> 7 regions (center + 6 neighbors)
        # --- heads / auxiliaries ---
        tie_weights=True,
        use_mfs_head=True,
        mfs_K=4,
        mfs_P=4,
        mfs_lowrank_r=8,
        add_l4_losses=True,
        l4_loss_weight=0.1,
        share_region_router=True,
        use_region_feedback=True,
        feedback_hops=1,
        region_slate_include_error=True,
        region_slate_include_prior=True,
    )
    cfg.update(extra_cfg or {})
    gpt_conf = GPTConfig(**cfg)
    model = GPT(gpt_conf)
    num_params = sum(p.numel() for p in model.parameters())
    return model, num_params, gpt_conf


# ---------------------------
# Per-layer optimizer helpers
# ---------------------------


def per_layer_optimizers(
    model: nn.Module,
    blocks: List[nn.Module],
    lr: float = 3e-4,
    weight_decay: float = 0.01,
) -> List[torch.optim.Optimizer]:
    """
    Create an AdamW optimizer per residual block.
    """
    opts = []
    for i, blk in enumerate(blocks):
        params = [p for p in blk.parameters() if p.requires_grad]
        if not params:
            # just in case
            opts.append(None)
            continue
        opt = torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        opts.append(opt)
    return opts


def renorm_incoming_weights(module: nn.Module, eps: float = 1e-8) -> None:
    """Renormalize each neuron's incoming weight vector to unit norm.

    This follows Hinton's FF recommendation to keep activations bounded and
    helps monitor when a layer's weights start to explode. We apply it after
    each local optimizer step.
    """
    for p in module.parameters():
        if p.ndim >= 2:
            w = p.data.view(p.size(0), -1)
            norms = w.norm(dim=1, keepdim=True).clamp(min=eps)
            w.div_(norms)
            p.data.copy_(w.view_as(p))


# ---------------------------
# Training (FF)
# ---------------------------


def train_ff(args):
    use_cpu = getattr(args, "cpu", False)
    if not use_cpu and not torch.cuda.is_available():
        print("WARNING: CUDA is not available, falling back to CPU")
        use_cpu = True
    device = torch.device("cuda" if not use_cpu else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_data, val_data = load_bytes_dataset(args.data_dir)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    model, num_params, gpt_conf = build_7region_15M_model(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model.to(device)
    model.train()

    approx_M = num_params / 1e6
    print(f"Model params: {approx_M:.2f}M  (target ~{args.target_params_m:.1f}M)")
    if abs(approx_M - args.target_params_m) > 3.0:
        print(
            "Warning: parameter count is not within ±3M of target. "
            "Adjust n_layer/n_embd if you need to be closer."
        )

    # Extract residual blocks
    blocks = get_blocks(model)
    print(f"Found {len(blocks)} residual blocks for local FF updates.")

    # Per-layer optimizers
    opts = per_layer_optimizers(
        model, blocks, lr=args.lr, weight_decay=args.weight_decay
    )

    # Optimizer for the token embedding + LM head (ties to embedding if configured).
    head_params = []
    if hasattr(model, "lm_head_mfs"):
        head_params.extend(p for p in model.lm_head_mfs.parameters() if p.requires_grad)
    if hasattr(model, "lm_head"):
        head_params.extend(p for p in model.lm_head.parameters() if p.requires_grad)
    # include token embedding (shared when tie_weights=True)
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        head_params.extend(p for p in model.transformer.wte.parameters() if p.requires_grad)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        head_params.extend(p for p in model.transformer.ln_f.parameters() if p.requires_grad)
    head_opt = (
        torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
        if head_params
        else None
    )

    # Optional: gradient clipping
    grad_clip = args.grad_clip

    # Training loop
    global_step = 0
    last_log = time.time()
    ema_pos_g = None
    ema_neg_g = None
    ema_noise_g = None

    for step in range(1, args.steps + 1):
        # optional linear LR decay
        cur_lr = args.lr
        if args.lr_final is not None:
            t = (step - 1) / max(1, args.steps - 1)
            cur_lr = args.lr + t * (args.lr_final - args.lr)
            for opt in opts:
                if opt is not None:
                    for pg in opt.param_groups:
                        pg["lr"] = cur_lr
            if head_opt is not None:
                for pg in head_opt.param_groups:
                    pg["lr"] = cur_lr
        # --- Build a minibatch of positive and negative sequences ---
        pos_seq, neg_seq = get_batch_bytes(
            train_data,
            args.batch_size,
            args.block_size,
            device,
            posneg_negatives=args.negatives,
        )
        # Extra pure noise batch for monitoring collapse
        noise_seq = torch.randint(0, 256, pos_seq.shape, device=device)

        # --- Capture *inputs* to each block for pos/neg/noise via one forward pass each ---
        with torch.no_grad():
            pos_inputs_per_block = snapshot_block_inputs(model, pos_seq, blocks)
            if neg_seq is not None:
                # We use only one negative set for local losses; you can extend to K>1 below
                neg_inputs_per_block = snapshot_block_inputs(
                    model, neg_seq[:, 0, :], blocks
                )
            else:
                neg_inputs_per_block = None
            noise_inputs_per_block = snapshot_block_inputs(model, noise_seq, blocks)

        # --- Per-layer local FF update ---
        total_loss_this_step = 0.0
        head_ce = 0.0
        head_ppl = float("nan")
        head_grad_norm = 0.0
        head_acc = float("nan")
        g_pos = g_neg = g_noise_gauss = g_noise_pure = None  # last block metrics
        layer_pos_means = []
        layer_neg_means = []
        layer_noise_means = []
        layer_grad_norms = []
        layer_weight_norms = []
        layer_ff_losses = []
        layer_posneg_accs = []

        for li, blk in enumerate(blocks):
            opt = opts[li]
            if opt is None:
                layer_pos_means.append(float("nan"))
                layer_neg_means.append(float("nan"))
                layer_noise_means.append(float("nan"))
                layer_grad_norms.append(0.0)
                layer_weight_norms.append(float("nan"))
                layer_ff_losses.append(float("nan"))
                layer_posneg_accs.append(float("nan"))
                continue
            opt.zero_grad(set_to_none=True)

            # Re-run THIS block on detached cached inputs so grads do not flow into earlier layers
            x_pos_in = pos_inputs_per_block[li].detach().requires_grad_(True)
            y_pos = blk(x_pos_in)
            g_pos = layer_goodness(y_pos, token_index=-1)  # [B]

            # Gaussian noise based "bad" activations using pos activations
            if args.noise_std > 0:
                act_std = y_pos.std().detach()
                noise = torch.randn_like(y_pos) * act_std * args.noise_std
                g_noise_gauss = layer_goodness(y_pos + noise, token_index=-1)  # [B]
            else:
                g_noise_gauss = None

            # Pure noise sequence for collapse monitoring
            x_noise_in = noise_inputs_per_block[li].detach()
            y_noise = blk(x_noise_in)
            g_noise_pure = layer_goodness(y_noise, token_index=-1)  # [B]

            if neg_inputs_per_block is not None:
                x_neg_in = neg_inputs_per_block[li].detach().requires_grad_(True)
                y_neg = blk(x_neg_in)
                g_neg = layer_goodness(y_neg, token_index=-1)  # [B]

                # Build labels: +1 for pos, -1 for neg/noise, concat
                parts = [g_pos]
                labels = [torch.ones_like(g_pos)]
                parts.append(g_neg)
                labels.append(-torch.ones_like(g_neg))
                if g_noise_gauss is not None:
                    parts.append(g_noise_gauss)
                    labels.append(-torch.ones_like(g_noise_gauss))
                g_all = torch.cat(parts, dim=0)
                y_lbl = torch.cat(labels, dim=0)
                loss_l = ff_binary_loss(g_all, y_lbl, theta=args.theta)
                acc = float((g_pos > g_neg).float().mean())
            else:
                # Positive-only regularization toward a target goodness (rarely used)
                if g_noise_gauss is not None:
                    g_all = torch.cat([g_pos, g_noise_gauss], dim=0)
                    y_lbl = torch.cat(
                        [torch.ones_like(g_pos), -torch.ones_like(g_noise_gauss)],
                        dim=0,
                    )
                    loss_l = ff_binary_loss(g_all, y_lbl, theta=args.theta)
                    acc = float("nan")
                else:
                    y_lbl = torch.ones_like(g_pos)
                    loss_l = ff_binary_loss(g_pos, y_lbl, theta=args.theta)
                    acc = float("nan")

            loss_l.backward()
            # Optional grad clip per block and capture grad norm for debugging
            if grad_clip is not None and grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(blk.parameters(), grad_clip))
            else:
                grad_norm = math.sqrt(
                    sum(
                        (p.grad.detach().pow(2).sum().item())
                        for p in blk.parameters()
                        if p.grad is not None
                    )
                )
            opt.step()

            if args.renorm:
                renorm_incoming_weights(blk)
            w_norm = math.sqrt(
                sum(p.detach().pow(2).sum().item() for p in blk.parameters())
            )

            layer_pos_means.append(float(g_pos.mean()))
            layer_neg_means.append(float(g_neg.mean()) if g_neg is not None else float("nan"))
            layer_noise_means.append(
                float(g_noise_pure.mean()) if g_noise_pure is not None else float("nan")
            )
            layer_grad_norms.append(grad_norm)
            layer_weight_norms.append(w_norm)
            layer_ff_losses.append(float(loss_l.detach()))
            layer_posneg_accs.append(acc)

            total_loss_this_step += float(loss_l.detach())

        # --- Local cross-entropy update for LM head / embeddings ---
        if head_opt is not None:
            head_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                final_in = snapshot_block_inputs(model, pos_seq, blocks)[-1]
                final_h = blocks[-1](final_in)
            # apply final layer norm before the head for parity with eval path
            if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                final_h = model.transformer.ln_f(final_h)
            if hasattr(model, "lm_head_mfs"):
                head_logp = model.lm_head_mfs(final_h, return_logprobs=True)
            else:
                head_logp = model.lm_head(final_h).log_softmax(dim=-1)
            V = head_logp.size(-1)
            logp = head_logp[:, :-1, :]
            target = pos_seq[:, 1:]
            ce_loss = F.nll_loss(logp.reshape(-1, V), target.reshape(-1))
            preds = logp.argmax(dim=-1)
            head_acc = float((preds == target).float().mean().item())
            ce_loss.backward()
            if grad_clip is not None and grad_clip > 0:
                head_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(head_params, grad_clip)
                )
            else:
                head_grad_norm = math.sqrt(
                    sum(
                        (p.grad.detach().pow(2).sum().item())
                        for p in head_params
                        if p.grad is not None
                    )
                )
            head_opt.step()
            head_ce = float(ce_loss.detach())
            head_ppl = math.exp(head_ce)
            total_loss_this_step += head_ce

        # --- Simple metrics/logging ---
        with torch.no_grad():
            mean_pos = float(g_pos.mean()) if g_pos is not None else float("nan")
            mean_neg = float(g_neg.mean()) if g_neg is not None else float("nan")
            mean_noise = (
                float(g_noise_gauss.mean()) if g_noise_gauss is not None else float("nan")
            )
            mean_noise_seq = (
                float(g_noise_pure.mean()) if g_noise_pure is not None else float("nan")
            )
            ema_pos_g = (
                mean_pos if ema_pos_g is None else 0.98 * ema_pos_g + 0.02 * mean_pos
            )
            ema_neg_g = (
                mean_neg if ema_neg_g is None else 0.98 * ema_neg_g + 0.02 * mean_neg
            )
            ema_noise_g = (
                mean_noise_seq
                if ema_noise_g is None
                else 0.98 * ema_noise_g + 0.02 * mean_noise_seq
            )

        global_step += 1
        if step % args.log_interval == 0 or step == 1:
            dt = time.time() - last_log
            last_log = time.time()
            print(
                f"step {step:6d}/{args.steps}  "
                f"ff_loss {total_loss_this_step:.4f}  "
                f"head_ce {head_ce:.4f}  head_ppl {head_ppl:.2f}  head_acc {head_acc:.3f}  "
                f"g_pos {mean_pos:.3f}  g_neg {mean_neg:.3f}  g_noise {mean_noise:.3f}  g_noise_seq {mean_noise_seq:.3f}  "
                f"ema_pos {ema_pos_g:.3f}  ema_neg {ema_neg_g:.3f}  ema_noise_seq {ema_noise_g:.3f}  "
                f"lr {cur_lr:.2e}  head_grad {head_grad_norm:.3f}  "
                f"({dt:.1f}s)"
            )
            print(
                "    layer_pos "
                + str([round(v, 3) for v in layer_pos_means])
                + " layer_neg "
                + str([round(v, 3) for v in layer_neg_means])
                + " layer_noise "
                + str([round(v, 3) for v in layer_noise_means])
                + " ff_loss "
                + str([round(v, 3) for v in layer_ff_losses])
                + " acc_pos>neg "
                + str([round(v, 3) for v in layer_posneg_accs])
                + " grad_norms "
                + str([round(v, 3) for v in layer_grad_norms])
                + " w_norms "
                + str([round(v, 3) for v in layer_weight_norms])
            )

        # Evaluation: goodness probe + LM metrics for parity with backprop run
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # LM NLL / bpc / ppl
                val_lm, val_acc = eval_lm(
                    model,
                    val_data,
                    args.block_size,
                    args.batch_size,
                    steps=args.eval_batches,
                    device=device,
                )
                bpc = bpc_from_nll(val_lm)
                ppl = ppl_from_nll(val_lm)
                print(
                    f"[eval] step {step:6d}  val_lm_nll {val_lm:.4f}  bpc {bpc:.3f}  ppl {ppl:.2f}  acc {val_acc:.3f}"
                )

                # Optional goodness gap on val split
                val_block_size = min(args.block_size, len(val_data))
                vpos, vneg = get_batch_bytes(
                    val_data,
                    args.batch_size,
                    val_block_size,
                    device,
                    posneg_negatives=1,
                )
                vpos_inputs = snapshot_block_inputs(model, vpos, blocks)
                vneg_inputs = snapshot_block_inputs(model, vneg[:, 0, :], blocks)
                per_block = []
                for li, blk in enumerate(blocks):
                    vp_all = layer_goodness(blk(vpos_inputs[li]), token_index=-1)
                    vn_all = layer_goodness(blk(vneg_inputs[li]), token_index=-1)
                    vp = vp_all.mean().item()
                    vn = vn_all.mean().item()
                    acc = (vp_all > vn_all).float().mean().item()
                    per_block.append((vp, vn, acc))
                last_pos, last_neg, last_acc = per_block[-1]
                print(
                    "[val probe] goodness per-block pos/neg: "
                    + " ".join(
                        f"{i}:{p:.2f}/{n:.2f}" for i, (p, n, _) in enumerate(per_block)
                    )
                )
                print(
                    "[val probe] pos>neg acc: "
                    + " ".join(
                        f"{i}:{a:.2f}" for i, (_, _, a) in enumerate(per_block)
                    )
                )
                print(
                    f"[val probe] last-block gap: pos {last_pos:.3f}  neg {last_neg:.3f}  gap {last_pos - last_neg:.3f}  acc {last_acc:.3f}"
                )
            model.train()

        # Save checkpoint
        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            ckpt = {
                "model": model.state_dict(),
                "config": (
                    gpt_conf.__dict__
                    if hasattr(gpt_conf, "__dict__")
                    else dict(gpt_conf)
                ),
                "args": vars(args),
                "step": step,
            }
            os.makedirs(args.out_dir, exist_ok=True)
            path = os.path.join(args.out_dir, f"ff_ckpt_step{step}.pt")
            torch.save(ckpt, path)
            print(f"Saved checkpoint to {path}")

    # Final save
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "ff_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "config": (
                gpt_conf.__dict__ if hasattr(gpt_conf, "__dict__") else dict(gpt_conf)
            ),
        },
        final_path,
    )
    print(f"Saved final model to {final_path}")

    # Quick sample for sanity, using standard autoregressive generation
    model.eval()
    ctx = torch.randint(0, 256, (1, min(64, args.block_size)), device=device)
    out = model.generate(ctx, max_new_tokens=200, temperature=1.0, top_k=50)
    sample = bytes(out[0].tolist()).decode("utf-8", errors="ignore")
    print("\n=== Sample ===")
    print(sample)


# ---------------------------
# FF-based generation (optional, slow)
# ---------------------------


@torch.no_grad()
def ff_generate(
    model: nn.Module,
    blocks: List[nn.Module],
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    device: torch.device,
    vocab_size: int = 256,
    theta: float = 2.0,
    num_candidates: Optional[int] = None,
) -> torch.Tensor:
    """
    Slow FF generation: at each step, score candidate next tokens and pick the
    one with the highest aggregated goodness across blocks. If `num_candidates`
    is None or >= vocab_size, all tokens in the vocabulary are evaluated.
    """
    model.eval()
    for _ in range(max_new_tokens):
        ctx = idx[:, -(block_size - 1) :] if idx.size(1) >= block_size else idx
        B = ctx.size(0)
        # Sample candidate tokens to score (for speed). You can also brute-force 0..255.
        if num_candidates is None or num_candidates >= vocab_size:
            cand = torch.arange(vocab_size, device=device).unsqueeze(0).repeat(B, 1)
        else:
            cand = torch.randint(
                0, vocab_size, (B, num_candidates), device=device, dtype=torch.long
            )
        best_tokens = []
        K = cand.size(1)
        for b in range(B):
            # ensure the true next token could be in the set if you want to test ground-truth scoring
            # For pure generation, we just sample K tokens.
            ctx_b = ctx[b : b + 1, :]  # [1, t]
            best_g = -1e9
            best_tok = 0
            for k in range(K):
                seq = torch.cat([ctx_b, cand[b : b + 1, k : k + 1]], dim=1)  # [1, t+1]
                inputs_per_block = snapshot_block_inputs(model, seq, blocks)
                # Aggregate goodness across all blocks for the final token
                g_total = 0.0
                for li, blk in enumerate(blocks):
                    out = blk(inputs_per_block[li])
                    g = layer_goodness(out, token_index=-1).item()
                    g_total += g
                if g_total > best_g:
                    best_g = g_total
                    best_tok = int(cand[b, k].item())
            best_tokens.append(best_tok)
        idx = torch.cat(
            [
                idx,
                torch.tensor(best_tokens, device=device, dtype=torch.long).unsqueeze(1),
            ],
            dim=1,
        )
    return idx


# ---------------------------
# CLI
# ---------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Forward-Forward training for Tiny Shakespeare (7-region model)"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="data/tiny_shakespeare",
        help="directory with input.txt",
    )
    p.add_argument(
        "--out_dir", type=str, default="out_ff", help="where to save checkpoints"
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--eval_interval",
        type=int,
        default=200,
        help="val goodness probe interval (0=off)",
    )
    p.add_argument(
        "--eval_batches", type=int, default=20, help="batches for LM eval at each probe"
    )
    p.add_argument(
        "--ckpt_interval",
        type=int,
        default=0,
        help="checkpoint interval in steps (0=off)",
    )
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="context len; we use T-1 ctx + 1 candidate token",
    )
    p.add_argument(
        "--negatives",
        type=int,
        default=1,
        help="number of negative candidates per sample (K)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--lr_final",
        type=float,
        default=None,
        help="final learning rate for linear decay (defaults to no decay)",
    )
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--theta", type=float, default=2.0, help="goodness threshold for logistic loss"
    )
    p.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="std multiplier for gaussian noise negatives",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    # Model sizing
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--target_params_m", type=float, default=15.0)
    p.add_argument(
        "--no_renorm",
        action="store_false",
        dest="renorm",
        help="disable weight renormalization after each block update",
    )
    p.set_defaults(renorm=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ff(args)
