# train_tiny_shakespeare.py
import os, math, time, argparse
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig  # uses your Fe‑RoPE + MFS + AE model :contentReference[oaicite:1]{index=1}
from htm_meta import apply_column_multipliers, ColumnMetaController  # meta hook (LR/WD scaling) :contentReference[oaicite:2]{index=2}
from cortex.hexgrid import make_grid, build_adjacency
from cortex.cortex_model import CortexModel
from cortex.io_patches import TextSensor

# -----------------------
# Data utilities (char)
# -----------------------
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def make_charset(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def split_train_val(enc, val_frac=0.1):
    # hold out the last val_frac of tokens for validation
    n = len(enc)
    n_train = int(n * (1.0 - val_frac))
    return enc[:n_train], enc[n_train:]

def get_batch(data, block_size, batch_size, device):
    # random contiguous chunks
    idx = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size]     for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x.to(device), y.to(device)

@torch.no_grad()
def eval_lm(model, data, block_size, batch_size, steps, vocab_size, device):
    """
    Evaluate *LM-only* negative log-likelihood (NLL) on held-out split.
    We purposely exclude AE aux here to measure generalization cleanly.
    """
    model.eval()
    tot = 0.0
    cnt = 0
    for _ in range(steps):
        x, y = get_batch(data, block_size, batch_size, device)
        # Forward once without targets to get log-probs from MFS head
        logits, _ = model(x, targets=None)   # with our model, these are LOG-PROBS when MFS is enabled
        V = logits.size(-1)
        nll = F.nll_loss(logits.view(-1, V), y.view(-1), ignore_index=-1, reduction='mean')
        tot += float(nll)
        cnt += 1
    return tot / max(1, cnt)

def bpc_from_nll(nll):
    return nll / math.log(2.0)

def ppl_from_nll(nll):
    return math.exp(nll)


def build_cortex_and_inputs(vocab_size: int,
                            d_model: int = 384,
                            regions_w: int = 17,
                            regions_h: int = 18,
                            k_active: int = 6,
                            num_facets: int = 7,
                            top_m_facets: int = 2,
                            router_top_k: int = 4,
                            device: str = "cuda"):
    hexes = make_grid(regions_w, regions_h)
    adj = build_adjacency(hexes, long_range_per_node=2)
    neighbor_indices = [adj[i] for i in range(len(hexes))]
    R = len(hexes)
    io_idxs = {'sensor': 0, 'motor': R - 1}

    model = CortexModel(R=R,
                        d_model=d_model,
                        neighbor_indices=neighbor_indices,
                        io_idxs=io_idxs,
                        vocab_size=vocab_size,
                        num_facets=num_facets,
                        top_m_facets=top_m_facets,
                        k_active=k_active,
                        n_slots=8,
                        router_top_k=router_top_k).to(device)

    sensor = TextSensor(vocab_size=vocab_size, d_model=d_model, ctx_len=128, tie_embedding=True).to(device)
    return model, sensor, neighbor_indices, io_idxs

# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/shakespeare_char/input.txt",
                   help="path to the Shakespeare text (char-level). Your shakespeare_char file works too.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--emb", type=int, default=384)

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=100)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--wd", type=float, default=0.1)

    p.add_argument("--rings", type=int, default=1)
    p.add_argument("--ferope_m", type=int, default=32)

    p.add_argument("--mfs", type=int, default=1)
    p.add_argument("--aux", type=int, default=1)
    p.add_argument("--aux_weight", type=float, default=0.1)
    p.add_argument("--router_share", type=int, default=1)
    p.add_argument("--feedback", type=int, default=0)

    # Meta knobs (HTM-style optimizer modulation)
    p.add_argument("--meta", type=int, default=1)
    p.add_argument("--meta_delay", type=int, default=None, help="steps to delay meta (default: warmup)")
    p.add_argument("--meta_floor", type=float, default=0.05)
    p.add_argument("--meta_lr", type=float, default=1.1)
    p.add_argument("--meta_wd", type=float, default=0.0)  # extra WD (scaled by LR); keep 0.0 by default
    p.add_argument("--meta_log", type=int, default=0)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--outdir", default="ckpt_shakespeare")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ---- data ----
    text = load_text(args.data)
    stoi, itos = make_charset(text)
    enc = encode(text, stoi)
    train_enc, val_enc = split_train_val(enc, val_frac=0.1)
    V = len(stoi)

    # ---- model ----
    cfg = GPTConfig(
        block_size=args.block, vocab_size=V,
        n_layer=args.layers, n_head=args.heads, n_embd=args.emb,
        dropout=0.0, bias=True,

        # Fe‑RoPE + PDS sparsity (geometric routing) 
        ferope_anchor_relative=True, ferope_m=args.ferope_m,
        use_block_sparse=True, neighbor_rings=args.rings,
        anchor_min_gap=max(32, args.block // 8), anchor_jitter=max(4, args.block // 64),

        # Heads (MFS) :contentReference[oaicite:4]{index=4}
        tie_weights=True, use_mfs_head=bool(args.mfs), mfs_K=3, mfs_P=4, mfs_lowrank_r=0,

        # Aux AE (sliding-window L4) :contentReference[oaicite:5]{index=5}
        add_l4_losses=bool(args.aux), l4_loss_weight=args.aux_weight,

        # Extra flags (router share/feedback slate infra if present in your local model)
        share_region_router=bool(args.router_share),
        use_region_feedback=bool(args.feedback), feedback_hops=1,
        region_slate_include_error=True, region_slate_include_prior=True,
    )
    model = GPT(cfg).to(device)
    opt = model.configure_optimizers(weight_decay=args.wd, learning_rate=args.lr,
                                     betas=(0.9, 0.95),
                                     device_type="cuda" if device.type == "cuda" else "cpu")

    # ---- scheduler: linear warmup → cosine decay ----
    def lr_lambda(step):
        if step < args.warmup:
            return max(1e-8, (step + 1) / max(1, args.warmup))
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return max(args.min_lr / args.lr, 0.5 * (1.0 + math.cos(math.pi * t)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    # ---- meta controller (per-block columns) :contentReference[oaicite:6]{index=6}
    meta = ColumnMetaController(
        lr_boost=float(args.meta_lr),
        wd_boost=float(args.meta_wd),
        ema=0.9,
        success_floor=float(args.meta_floor),
    )
    for b_idx, blk in enumerate(model.transformer.h):
        for _, p in blk.named_parameters():
            setattr(p, "_col_id", b_idx)

    os.makedirs(args.outdir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(args.outdir, "best.pt")

    # ---- train ----
    model.train()
    log_every = 50
    base_aux_w = float(getattr(model, "_l4_loss_weight", args.aux_weight))
    meta_delay = args.warmup if args.meta_delay is None else args.meta_delay

    t0 = time.time()
    for step in range(1, args.steps + 1):
        # Ramp the AE aux weight during warmup
        warm_frac = min(1.0, step / max(1, args.warmup))
        model._l4_loss_weight = base_aux_w * warm_frac

        x, y = get_batch(train_enc, args.block, args.batch, device)
        opt.zero_grad(set_to_none=True)

        # Suppose your batch provides: tokens [B, T] and targets [B, T]
        # Build once:
        # model, sensor, neighbor_indices, io_idxs = build_cortex_and_inputs(V, args.emb, 17, 18, 6, 7, 2, 4, device)
        #
        # For each step:
        # x_emb, tied = sensor(tokens)          # [B, T, D]; use last step or a pooling as region input
        # x_per_region = torch.zeros(len(neighbor_indices), tokens.size(0), model.d_model, device=device)
        # x_per_region[io_idxs['sensor']] = x_emb[:, -1, :]  # Example: feed last-token embedding into sensor region
        # logp, loss, aux = model(x_per_region, targets=targets[:, -1])

        # forward: combined loss (LM + scaled AE)
        logits, total_loss = model(x, targets=y)

        # LM-only NLL (from MFS log-probs) for clear tracking
        V = logits.size(-1)
        lm_nll = F.nll_loss(logits.view(-1, V), y.view(-1), ignore_index=-1, reduction='mean')

        total_loss.backward()

        # HTM meta (delay until after warmup)
        if args.meta and step >= meta_delay:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)  # okay even if logits are log-probs
                p_true = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean().item()
                p_baseline = 1.0 / V
                ratio = max(1e-8, p_true / p_baseline)
                s_norm = max(0.0, min(1.0, (ratio - 1.0) / 4.0))  # 0 at baseline, 1 at 5× baseline
                col_scores = {b: s_norm for b in range(cfg.n_layer)}
                meta.update_scores(col_scores)
                col_lr, col_wd = meta.compute_multipliers()
                current_lr = float(opt.param_groups[0]["lr"])
                apply_column_multipliers(
                    model.named_parameters(), col_lr, col_wd,
                    weight_decay_base=args.wd, lr_scale=current_lr
                )
                if args.meta_log and (step % log_every == 0):
                    print(f"[meta] step {step} p_true={p_true:.4e} ratio={ratio:.2f} s_norm={s_norm:.3f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if step % log_every == 0:
            dt = time.time() - t0; t0 = time.time()
            print(f"step {step:5d}/{args.steps}  lm_nll {float(lm_nll):.4f}  total {float(total_loss):.4f}  "
                  f"{train_enc.numel()/1e6:.1f}M chars  ({dt*1000/log_every:.1f} ms/iter)")

        # periodic evaluation
        if step % args.eval_every == 0 or step == args.steps:
            val_lm = eval_lm(model, val_enc, args.block, args.batch, args.eval_batches, V, device)
            bpc = bpc_from_nll(val_lm)
            ppl = ppl_from_nll(val_lm)
            print(f"[eval] step {step:5d}  val_lm_nll {val_lm:.4f}  bpc {bpc:.3f}  ppl {ppl:.2f}")

            # checkpoint best by validation LM NLL
            if val_lm < best_val:
                best_val = val_lm
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "val_lm_nll": val_lm,
                    "step": step,
                }, best_path)
                print(f"[ckpt] saved best to: {best_path}")

    # ---- quick sample from the trained model ----
    model.eval()
    itos = {i: ch for ch, i in stoi.items()}
    ctx = torch.randint(0, V, (1, min(64, args.block)), device=device)
    out = model.generate(ctx, max_new_tokens=200, temperature=1.0, top_k=50)
    print("\n=== Sample ===")
    print("".join(itos[int(i)] for i in out[0].tolist()))
    print("\nDone.")
if __name__ == "__main__":
    main()
