import argparse
from typing import Optional

import torch
from model import GPT, GPTConfig


def load_model(ckpt_path: str, device: torch.device) -> GPT:
    # Use weights_only=True for safer loading; the checkpoint contains only tensors
    # and simple Python types, so this is compatible with the restricted loader
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ckpt.get("config") or ckpt.get("model_args")
    if config is None:
        raise ValueError("Checkpoint is missing model config")
    gpt_conf = GPTConfig(**config)
    model = GPT(gpt_conf)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def decode_bytes(tokens) -> str:
    """Decode a list of byte tokens into UTF-8 text, ignoring malformed bytes."""
    return bytes(tokens).decode("utf-8", errors="ignore")


def get_blocks(model: torch.nn.Module):
    node = model
    for attr in ("transformer", "h"):
        if hasattr(node, attr):
            node = getattr(node, attr)
        else:
            node = None
            break
    if isinstance(node, torch.nn.ModuleList):
        return list(node)
    raise RuntimeError("Model does not expose transformer.h blocks")


@torch.no_grad()
def snapshot_block_inputs(model, x, blocks):
    cached = [None for _ in blocks]
    handles = []

    def make_pre_hook(i):
        def hook(mod, inp):
            cached[i] = inp[0].detach()

        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_pre_hook(make_pre_hook(i)))
    _ = model(x)
    for h in handles:
        h.remove()
    return cached


def layer_goodness(x, token_index=-1):
    x_last = x[:, token_index, :]
    return (x_last**2).mean(dim=1)


@torch.no_grad()
def ff_generate(
    model,
    blocks,
    idx,
    max_new_tokens,
    block_size,
    device,
    num_candidates: Optional[int] = None,
):
    for _ in range(max_new_tokens):
        ctx = idx[:, -(block_size - 1) :] if idx.size(1) >= block_size else idx
        B = ctx.size(0)
        if num_candidates is None or num_candidates >= model.config.vocab_size:
            cand = torch.arange(model.config.vocab_size, device=device).unsqueeze(0).repeat(B, 1)
        else:
            cand = torch.randint(0, model.config.vocab_size, (B, num_candidates), device=device)
        next_tokens = []
        K = cand.size(1)
        for b in range(B):
            ctx_b = ctx[b : b + 1, :]
            best_g = -1e9
            best_tok = 0
            for k in range(K):
                seq = torch.cat([ctx_b, cand[b : b + 1, k : k + 1]], dim=1)
                inputs = snapshot_block_inputs(model, seq, blocks)
                g_tot = 0.0
                for li, blk in enumerate(blocks):
                    g = layer_goodness(blk(inputs[li]), token_index=-1).item()
                    g_tot += g
                if g_tot > best_g:
                    best_g = g_tot
                    best_tok = int(cand[b, k])
            next_tokens.append(best_tok)
        idx = torch.cat([idx, torch.tensor(next_tokens, device=device).unsqueeze(1)], dim=1)
    return idx


def main():
    p = argparse.ArgumentParser(description="Generate text from FF-trained Tiny Shakespeare model")
    p.add_argument("--ckpt", type=str, default="out_ff/ff_final.pt")
    p.add_argument("--prompt", type=str, default="The thing about gooses is ")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--ff_scan", action="store_true", help="use slow FF scanning instead of model.generate")
    p.add_argument(
        "--scan_candidates",
        type=int,
        default=256,
        help="candidates per step for FF scan (default: evaluate all 256 tokens)",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = p.parse_args()

    use_cpu = getattr(args, "cpu", False)
    if not use_cpu and not torch.cuda.is_available():
        print("WARNING: CUDA is not available, falling back to CPU")
        use_cpu = True
    device = torch.device("cuda" if not use_cpu else "cpu")
    torch.manual_seed(args.seed)

    model = load_model(args.ckpt, device)

    prompt_bytes = args.prompt.encode("utf-8")
    idx = torch.tensor(list(prompt_bytes), dtype=torch.long, device=device)[None, :]

    with torch.no_grad():
        if args.ff_scan:
            blocks = get_blocks(model)
            out = ff_generate(
                model,
                blocks,
                idx,
                args.max_new_tokens,
                model.config.block_size,
                device,
                num_candidates=args.scan_candidates,
            )
        else:
            out = model.generate(
                idx,
                args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

    # `out` already contains the prompt followed by generated tokens.
    # Decode the full sequence so the printed text begins with the prompt.
    text = decode_bytes(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
