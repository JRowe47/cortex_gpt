# tests/smoke_test.py
import argparse, os, math, time
import torch
from torch import nn

# Import your updated model (this must be your repo's model.py)
from model import GPT, GPTConfig  # ⟵ relies on your current file
# If your project uses a package, adjust the import accordingly.

def pretty(n):
    return f"{n/1e6:.2f}M"

def run_once(cfg, device, batch_size=2, seq_len=32, step_lr=3e-4, do_aux=True):
    print("\n=== Instantiating model ===")
    print(cfg)
    m = GPT(cfg).to(device)
    m.eval()
    print(f"Params: {pretty(sum(p.numel() for p in m.parameters()))}")

    # ----- Inference path (no targets) -----
    print("\n=== Forward (inference) ===")
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        logits, loss = m(x)  # loss should be None in inference
    print("logits shape:", tuple(logits.shape), " loss:", loss)

    # Treat outputs as generic logits; turn them into probabilities for the sanity check
    probs = torch.softmax(logits, dim=-1)
    row_sums = probs.sum(dim=-1)
    mean_sum = row_sums.mean().item()
    print(f"Mean(prob sum) over batch×time (expect ~1.0): {mean_sum:.6f}")
    assert abs(mean_sum - 1.0) < 5e-6, "probabilities did not sum to 1 after softmax"


    # ----- Training path (with targets) -----
    print("\n=== Forward (train) + 1 optimizer step ===")
    m.train()
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    logits, loss = m(x, targets=y)
    print("train logits shape:", tuple(logits.shape), " loss:", float(loss))
    assert torch.isfinite(loss), "Loss has NaNs/Inf"

    opt = torch.optim.AdamW(m.parameters(), lr=step_lr, betas=(0.9, 0.95))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    total_grad_norm = nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
    print(f"Total grad norm (clipped to 1.0): {float(total_grad_norm):.4f}")
    opt.step()

    # ----- Generation path -----
    print("\n=== Generation (8 tokens) ===")
    m.eval()
    ctx = torch.randint(0, cfg.vocab_size, (batch_size, min(8, seq_len)), device=device)
    out = m.generate(ctx, max_new_tokens=8, temperature=1.0, top_k=50)
    print("input length:", ctx.shape[1], " -> output length:", out.shape[1])
    assert out.shape[1] == ctx.shape[1] + 8, "Generation did not produce 8 new tokens"

    # ----- Crop block size sanity (optional) -----
    print("\n=== Crop block size (sanity) ===")
    new_bs = max(16, min(cfg.block_size, 64))
    m.crop_block_size(new_bs)
    small_ctx = out[:, -min(new_bs, out.shape[1]):]
    out2 = m.generate(small_ctx, max_new_tokens=4)
    print("post-crop generate ok ->", tuple(out2.shape))

    # ----- Report -----
    print("\nSmoke test completed OK ✅")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda","cpu"], help="device to run the test on")
    p.add_argument("--vocab", type=int, default=256, help="vocab size for the smoke test (bytes=256 is cheap)")
    p.add_argument("--block", type=int, default=128, help="block size to test")
    p.add_argument("--seq", type=int, default=32, help="sequence length for the test batch")
    p.add_argument("--batch", type=int, default=2, help="batch size for the test batch")
    p.add_argument("--mfs", type=int, default=1, help="1=use MFS head, 0=use classic linear head")
    p.add_argument("--aux", type=int, default=1, help="1=enable L4 auxiliary losses, 0=disable")
    p.add_argument("--rings", type=int, default=1, help="neighbor rings for PDS block sparsity")
    p.add_argument("--ferope_m", type=int, default=32, help="Fe‑RoPE rotary pairs (2m dims used)")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(123)

    # Small, fast config for the smoke test. Adjust n_layer/n_head/n_embd if you like.
    cfg = GPTConfig(
        block_size=args.block,
        vocab_size=args.vocab,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=True,

        # Fe‑RoPE + PDS sparsity
        ferope_anchor_relative=True,
        anchor_min_gap=32,
        anchor_jitter=8,
        ferope_m=args.ferope_m,
        use_block_sparse=True,
        neighbor_rings=args.rings,

        # Heads
        tie_weights=True,
        use_mfs_head=bool(args.mfs),
        mfs_K=3,
        mfs_P=4,
        mfs_lowrank_r=0,

        # Aux heads
        add_l4_losses=bool(args.aux),
        l4_loss_weight=0.1,
    )

    ok = run_once(cfg, device, batch_size=args.batch, seq_len=args.seq)
    if not ok: raise SystemExit(1)

    # Also try the fallback linear head path (to verify both code paths)
    if args.mfs:
        print("\n\n=== Second pass: fallback linear head (use_mfs_head=False) ===")
        cfg2 = cfg
        cfg2.use_mfs_head = False
        cfg2.add_l4_losses = False
        _ = run_once(cfg2, device, batch_size=args.batch, seq_len=args.seq)

        cfg3 = GPTConfig(
            block_size=args.block, vocab_size=args.vocab,
            n_layer=2, n_head=2, n_embd=128, dropout=0.0, bias=True,
            ferope_anchor_relative=True, anchor_min_gap=32, anchor_jitter=8, ferope_m=args.ferope_m,
            use_block_sparse=True, neighbor_rings=args.rings,
            tie_weights=True, use_mfs_head=bool(args.mfs),
            add_l4_losses=bool(args.aux), l4_loss_weight=0.1,
            # NEW
            share_region_router=True,
            use_region_feedback=True, feedback_hops=1,
        )
        _ = run_once(cfg3, device, batch_size=args.batch, seq_len=args.seq)
        
def test_cortex_minimal():
    from cortex.hexgrid import make_grid, build_adjacency
    from cortex.cortex_model import CortexModel

    vocab_size = 100
    d_model = 64
    hexes = make_grid(5, 5)  # 25 regions (small)
    adj = build_adjacency(hexes, long_range_per_node=1)
    neighbor_indices = [adj[i] for i in range(len(hexes))]
    R = len(neighbor_indices)
    io_idxs = {'sensor': 0, 'motor': R - 1}

    model = CortexModel(R=R, d_model=d_model, neighbor_indices=neighbor_indices,
                        io_idxs=io_idxs, vocab_size=vocab_size, k_active=1, num_facets=7, top_m_facets=2)

    B = 3
    x_per_region = torch.zeros(R, B, d_model)
    x_per_region[io_idxs['sensor']] = torch.randn(B, d_model)
    targets = torch.randint(0, vocab_size, (B,))

    logp, loss, aux = model(x_per_region, targets=targets)
    assert logp.shape == (B, vocab_size)
    assert loss is not None
    k_selected = int(aux['gate']['k_selected_avg'])
    assert k_selected >= 1, f"Expected at least 1 active region, got {k_selected}"
    print("OK: minimal cortex test passes.")

if __name__ == "__main__":
    main()
