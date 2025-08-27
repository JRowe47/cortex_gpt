# ferope_ar.py
# Fe‑RoPE (anchor‑relative) + PDS anchors + optional blockwise stitch
from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 1) Poisson-disk anchors
# ---------------------------

@torch.no_grad()
def build_pd_anchors_1d(T: int, min_gap: int = 256, jitter: int = 32, device=None, generator: torch.Generator | None = None):
    """
    1‑D 'blue-noise' anchors along the token axis. Greedy is sufficient in 1‑D.
    Deterministic when you seed torch: uses torch.randint() for jitter.

    Args:
      T: total sequence length
      min_gap: minimum stride between successive anchors (in tokens)
      jitter: integer jitter applied to each stride, sampled in [-jitter, +jitter]
      device: torch device for outputs
      generator: optional torch.Generator to control RNG stream

    Returns:
      anchors_pos: [A] float tensor of anchor coordinates (0..T-1)
      assign:      [T] long tensor mapping each token index to nearest anchor id
    """
    import torch
    device = device or 'cpu'
    anchors = [0]
    last = 0
    while last + min_gap < T:
        if jitter > 0:
            # sample in [-jitter, +jitter]
            jit = int(torch.randint(low=-jitter, high=jitter + 1, size=(1,), generator=generator).item())
        else:
            jit = 0
        step = max(1, int(min_gap + jit))
        nxt = max(last + 1, last + step)
        nxt = min(T - 1, nxt)
        anchors.append(nxt)
        last = nxt

    anchors_pos = torch.tensor(sorted(set(anchors)), device=device, dtype=torch.float32)  # [A]
    t = torch.arange(T, device=device, dtype=torch.float32)
    if anchors_pos.numel() == 1:
        assign = torch.zeros(T, dtype=torch.long, device=device)
        return anchors_pos, assign
    mids = (anchors_pos[1:] + anchors_pos[:-1]) * 0.5
    assign = torch.bucketize(t, mids).to(torch.long)  # [T] in {0..A-1}
    return anchors_pos, assign


# ---------------------------
# 2) Fe‑RoPE anchor‑relative rotations
# ---------------------------

def _pair_rotate(x, cos, sin):
    # x: [B,H,T,2m], cos/sin: [B,1,T,m]
    B,H,T,_ = x.shape
    m = cos.shape[-1]
    x = x.view(B,H,T,m,2)
    x0, x1 = x[...,0], x[...,1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack([y0, y1], dim=-1).reshape(B,H,T,2*m)

def ferope_ar_rotate(q, k, delta, W):
    """
    Apply Fe‑RoPE to (q,k) using **anchor‑relative** offsets delta \in R^{B,T,d}
    W: [m, d] frequency bank (share per head or not; pass same W for all heads)
    Assumes first 2m dims of head_dim are the rotary dims.
    """
    B,H,T,Dh = q.shape
    device = q.device
    m, d = W.shape
    rot = 2*m
    # theta = delta @ W^T
    theta = torch.einsum('btd,md->btm', delta, W)  # [B,T,m]
    cos = torch.cos(theta).unsqueeze(1)  # [B,1,T,m]
    sin = torch.sin(theta).unsqueeze(1)

    def rotate(x):
        x_rot, x_pass = x[...,:rot], x[...,rot:]
        y = _pair_rotate(x_rot, cos, sin)
        return torch.cat([y, x_pass], dim=-1)

    return rotate(q), rotate(k)

def stitch_keys_for_block(k_block, dtheta, m):
    """
    Extra constant rotation for a **target** anchor block when a query block from A
    attends a key block from B.  dtheta = W @ (a_B - a_A)  [m]
    k_block: [B,H,Tk,Dh]
    """
    rot = 2*m
    c = torch.cos(dtheta).view(1,1,1,m)
    s = torch.sin(dtheta).view(1,1,1,m)
    kb, kp = k_block[...,:rot], k_block[...,rot:]
    kb = kb.view(*kb.shape[:3], m, 2)
    k0, k1 = kb[...,0], kb[...,1]
    y0 = k0 * c - k1 * s
    y1 = k0 * s + k1 * c
    y = torch.stack([y0, y1], dim=-1).reshape(*k_block.shape[:3], rot)
    return torch.cat([y, kp], dim=-1)

# ---------------------------
# 3) Utilities
# ---------------------------

def make_positions(B:int, T:int, device, normalize=True):
    """
    1-D byte (or token) coordinate per position, optionally normalized to [0,1].
    Returns p: [B,T,1] float.
    """
    t = torch.arange(T, device=device).float().view(1,T,1).expand(B,-1,-1)
    if normalize and T>1:
        t = t / (T-1)
    return t

def anchor_local_offsets(p, anchors_pos, assign):
    """
    p: [B,T,1]; anchors_pos: [A]; assign: [T]
    returns delta: [B,T,1]
    """
    B,T,_ = p.shape
    a = anchors_pos.index_select(0, assign).view(1,T,1).expand(B,-1,-1)
    return p - a  # [B,T,1]
