# debug_geo.py
"""
Quick probe of the geometric routing setup:
  - Samples PDS anchors on a sequence of length T (using Torch RNG)
  - Builds the neighbor-ring mask
  - Reports #anchors (A), per-cell sizes, and mask density (allowed fraction)
Adds an 'expected density' ≈ (2*rings+1)/A for equal-sized cells.
"""
import argparse, statistics as stats, random
import torch
from ferope_ar import build_pd_anchors_1d                     # PDS anchors (1-D) :contentReference[oaicite:3]{index=3}
from sparse_routing import allowed_anchor_pairs               # neighbor-ring mask builder :contentReference[oaicite:4]{index=4}

def probe(T:int, min_gap:int, jitter:int, rings:int, seed:int=123):
    torch.manual_seed(seed); random.seed(seed)
    anchors_pos, assign = build_pd_anchors_1d(T, min_gap=min_gap, jitter=jitter, device="cpu")
    A = int(anchors_pos.numel())
    neighbor_ids = {a: set([a]) for a in range(A)}
    for a in range(A):
        for r in range(1, rings + 1):
            if a - r >= 0:  neighbor_ids[a].add(a - r)
            if a + r < A:   neighbor_ids[a].add(a + r)
    M = allowed_anchor_pairs(assign, assign, neighbor_ids, T, T)  # [T,T] bool
    density = M.float().mean().item()
    counts = torch.bincount(assign, minlength=A).tolist()
    expected = min(1.0, (2 * rings + 1) / max(1, A))
    return dict(T=T, A=A, rings=rings, min_gap=min_gap, jitter=jitter,
                density=density, expected_density=expected,
                cell_stats=dict(min=min(counts), max=max(counts),
                mean=sum(counts)/len(counts), median=stats.median(counts)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--min_gap", type=int, default=32)
    ap.add_argument("--jitter", type=int, default=8)
    ap.add_argument("--rings", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    info = probe(args.T, args.min_gap, args.jitter, args.rings, args.seed)
    print("=== Geometric routing probe ===")
    for k in ["T","A","rings","min_gap","jitter","density","expected_density","cell_stats"]:
        print(f"{k}: {info[k]}")
if __name__ == "__main__":
    main()
