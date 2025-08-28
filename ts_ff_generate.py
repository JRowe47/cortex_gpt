import argparse
import torch

# Prefer the cortex model utilities when available for 7-region checkpoints
try:
    from cortex.cortex_model import CortexModel
    from cortex.io_patches import TextSensor
    from cortex.hexgrid import make_grid, build_adjacency
except Exception:  # pragma: no cover - cortex utilities optional
    CortexModel = None
    TextSensor = None

from model import GPT, GPTConfig


def load_model(ckpt_path: str, device: torch.device):
    """Load either a standard GPT or a CortexModel checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ckpt.get("config") or ckpt.get("model_args") or {}

    # Detect CortexModel checkpoints via presence of region keys
    if CortexModel is not None and any(k in config for k in ("R", "neighbor_indices")):
        R = config.get("R")
        d_model = config.get("d_model")
        vocab_size = config.get("vocab_size")

        if config.get("neighbor_indices") is not None:
            neighbor_indices = config["neighbor_indices"]
        else:
            # Rebuild a default hex grid if adjacency not stored
            w = config.get("regions_w", int(R ** 0.5))
            h = config.get("regions_h", max(1, R // max(1, w)))
            hexes = make_grid(w, h)
            adj = build_adjacency(hexes, long_range_per_node=2)
            neighbor_indices = [adj[i] for i in range(len(hexes))]

        io_idxs = config.get("io_idxs", {"sensor": 0, "motor": R - 1})
        num_facets = config.get("num_facets", 7)
        top_m_facets = config.get("top_m_facets", 2)
        k_active = config.get("k_active", 6)
        router_top_k = config.get("router_top_k", 4)

        model = CortexModel(R=R,
                            d_model=d_model,
                            neighbor_indices=neighbor_indices,
                            io_idxs=io_idxs,
                            vocab_size=vocab_size,
                            num_facets=num_facets,
                            top_m_facets=top_m_facets,
                            k_active=k_active,
                            router_top_k=router_top_k).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        sensor = TextSensor(vocab_size=vocab_size, d_model=d_model, ctx_len=config.get("block_size", 128),
                            tie_embedding=True).to(device)
        return model, sensor

    # Fallback GPT model
    gpt_conf = GPTConfig(**config)
    model = GPT(gpt_conf)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, None


def decode_bytes(tokens) -> str:
    """Decode a list of byte tokens into UTF-8 text, ignoring malformed bytes."""
    return bytes(tokens).decode("utf-8", errors="ignore")


def main():
    p = argparse.ArgumentParser(description="Generate text from FF-trained Tiny Shakespeare model")
    p.add_argument("--ckpt", type=str, default="out_ff/ff_final.pt")
    p.add_argument("--prompt", type=str, default="The thing about gooses is ")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = p.parse_args()

    use_cpu = getattr(args, "cpu", False)
    if not use_cpu and not torch.cuda.is_available():
        print("WARNING: CUDA is not available, falling back to CPU")
        use_cpu = True
    device = torch.device("cuda" if not use_cpu else "cpu")
    torch.manual_seed(args.seed)

    model, sensor = load_model(args.ckpt, device)

    prompt_bytes = args.prompt.encode("utf-8")
    idx = torch.tensor(list(prompt_bytes), dtype=torch.long, device=device)[None, :]

    if sensor is not None:
        # CortexModel expects sensor embeddings per region
        B, T = idx.shape
        x_emb, _ = sensor(idx)
        x_per_region = torch.zeros(model.R, B, model.d_model, device=device)
        x_per_region[model.io_idxs['sensor']] = x_emb[:, 0]
        with torch.no_grad():
            logp, _, _ = model(x_per_region, targets=None)
            # For simplicity, sample greedily from motor region logits
            next_tok = logp.argmax(dim=-1)
        tokens = idx[0].tolist() + [int(next_tok.item())]
        text = decode_bytes(tokens)
        print(text)
    else:
        with torch.no_grad():
            out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        text = decode_bytes(out[0].tolist())
        print(text)


if __name__ == "__main__":
    main()
