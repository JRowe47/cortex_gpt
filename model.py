"""
Full definition of a GPT Language Model, all of it in this single file.

This variant integrates:
- Fe‑RoPE (anchor‑relative) with Poisson‑disk anchors + block‑sparse neighbor attention
- Multi‑Facet Softmax (MFS) head
- Optional L4 sliding‑window AE auxiliary losses
"""

import math
import inspect
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# Geometry + sparsity helpers (already in your repo)
from ferope_ar import (
    build_pd_anchors_1d, make_positions, anchor_local_offsets, ferope_ar_rotate
)
from sparse_routing import allowed_anchor_pairs

# Heads (already in your repo)
from mfs import MFSHead
from ff import FFHead, KWTA, DendriteGate
from sliding_window_ae import SlidingWindowAE


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _ironrope_make_W(d, m, kind="log", base=10000.0, sigma=1.0, device=None, dtype=None):
    """Frequency bank W ∈ R^{m×d}. 'log' gives classic RoPE-like bands; 'gaussian' gives RFF-style rows."""
    if m == 0:
        return torch.empty(0, d, device=device, dtype=dtype)
    if kind == "log":
        # allocate evenly across axes, frequencies in [1/base, 1]
        m_per_axis = max(1, m // d)
        rows = []
        for ax in range(d):
            freqs = torch.logspace(
                start=math.log(1.0 / base), end=0.0, steps=m_per_axis, base=math.e,
                device=device, dtype=dtype or torch.float32
            )
            block = torch.zeros(m_per_axis, d, device=device, dtype=dtype or torch.float32)
            block[:, ax] = freqs
            rows.append(block)
        W = torch.cat(rows, dim=0)
        if W.shape[0] < m:  # pad if not divisible
            extra = m - W.shape[0]
            freqs = torch.logspace(
                start=math.log(1.0 / base), end=0.0, steps=extra, base=math.e,
                device=device, dtype=dtype or torch.float32
            )
            block = torch.zeros(extra, d, device=device, dtype=dtype or torch.float32)
            block[:, -1] = freqs
            W = torch.cat([W, block], dim=0)
        return W  # [m,d]
    elif kind == "gaussian":
        return torch.randn(m, d, device=device, dtype=dtype or torch.float32) * sigma
    else:
        raise ValueError(f"unknown freq kind: {kind}")


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ---------------------------------------------------------------------
# RegionRouter (PDS anchors + neighbor rings) shared across layers
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# RegionRouter (PDS anchors + neighbor rings) shared across layers
# ---------------------------------------------------------------------
class RegionRouter:
    def __init__(self, T:int, device, anchor_min_gap:int, anchor_jitter:int, neighbor_rings:int):
        self.T = T
        self.device = device
        # Poisson-disk anchors and token->anchor assignment
        self.anchors_pos, self.assign = build_pd_anchors_1d(
            T, min_gap=anchor_min_gap, jitter=anchor_jitter, device=device
        )  # [A], [T]
        self.A = int(self.anchors_pos.numel())
        # neighbor ids on anchor graph
        self.neighbor_ids = {a: set([a]) for a in range(self.A)}
        for a in range(self.A):
            for r in range(1, neighbor_rings + 1):
                if a - r >= 0:  self.neighbor_ids[a].add(a - r)
                if a + r < self.A: self.neighbor_ids[a].add(a + r)
        # token-level block-sparse mask [T,T]
        self.mask = allowed_anchor_pairs(self.assign, self.assign, self.neighbor_ids, T, T)

    def delta(self, B:int, normalize:bool=True):
        # anchor-relative offsets for Fe‑RoPE
        p = make_positions(B, self.T, device=self.device, normalize=normalize)  # [B,T,1]
        return anchor_local_offsets(p, self.anchors_pos, self.assign)           # [B,T,1]

    @torch.no_grad()
    def _counts(self):
        # [A] integer counts
        return torch.bincount(self.assign, minlength=self.A).clamp_min(1)

    def pool_per_anchor(self, h: torch.Tensor, reduce:str="mean"):
        """
        h: [B,T,D] -> returns [B,A,D] pooled per anchor (no broadcast).
        """
        B,T,D = h.shape
        A = self.A
        idx = self.assign.view(1, T, 1).expand(B, T, D)  # [B,T,D]
        out = h.new_zeros(B, A, D)
        out.scatter_add_(1, idx, h)  # sum by anchor
        if reduce == "mean":
            counts = self._counts().view(1, A, 1).to(h.device)
            out = out / counts
        return out

    def broadcast_from_anchor(self, anchor_feats: torch.Tensor):
        """
        anchor_feats: [B,A,D] -> [B,T,D] by indexing the token's anchor id.
        """
        # index_select along anchor dimension (1) using [T] assignment
        return anchor_feats.index_select(1, self.assign)

    def pool_broadcast(self, h: torch.Tensor, reduce:str="mean"):
        """
        Convenience: pooled per anchor then broadcast to tokens. [B,T,D]
        """
        return self.broadcast_from_anchor(self.pool_per_anchor(h, reduce=reduce))

    def neighbor_aggregate(self, anchor_feats: torch.Tensor, hops:int=1, reduce:str="mean"):
        """
        anchor_feats: [B,A,D] (pooled per-anchor features) -> neighbor-aggregated [B,A,D]
        Aggregates along the anchor graph defined by neighbor_ids.
        """
        A = self.A
        # adjacency/averaging weights W[a_dst, a_src]
        W = torch.zeros(A, A, device=anchor_feats.device, dtype=anchor_feats.dtype)
        for a_dst in range(A):
            neigh = sorted(self.neighbor_ids[a_dst])
            if len(neigh) == 0:
                continue
            if reduce == "mean":
                W[a_dst, torch.tensor(neigh, device=W.device, dtype=torch.long)] = 1.0 / len(neigh)
            elif reduce == "sum":
                W[a_dst, torch.tensor(neigh, device=W.device, dtype=torch.long)] = 1.0
            else:
                raise ValueError(f"unknown reduce mode: {reduce}")

        # Multi-hop neighbor diffusion.
        # out[b, a_dst, d] = sum_{a_src} anchor_feats[b, a_src, d] * W[a_dst, a_src]
        out = anchor_feats
        for _ in range(max(1, hops)):
            out = torch.einsum('bcd,ac->bad', out, W)  # NOTE: 'b' (batch) is NOT used in W
        return out


# ---------------------------------------------------------------------
# Attention block with Fe‑RoPE (anchor‑relative) + PDS block sparsity
# ---------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0

        # QKV projection (batched)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # causal mask (lower triangle) sized to block_size
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
            persistent=False
        )

        # Fe‑RoPE frequency bank (anchor‑relative 1‑D positions)
        self.ferope_m = max(0, min(self.head_dim // 2, int(getattr(config, "ferope_m", 0))))
        if self.ferope_m > 0:
            W = _ironrope_make_W(d=1, m=self.ferope_m, kind="log", base=getattr(config, "rope_base", 10000.0))
            self.register_buffer("rope_W", W, persistent=True)   # [m,1]
        else:
            self.register_buffer("rope_W", torch.empty(0, 1), persistent=True)

        self._router_ctx: Optional[RegionRouter] = None
        
    # NEW: set shared region router (called by GPT.forward before block runs)        
    def set_router(self, router: Optional['RegionRouter']):
        self._router_ctx = router

    def forward(self, x, ctx=None):
        B, T, C = x.size()

        # project to QKV and reshape
        qkv = self.c_attn(x)  # [B, T, 3C]
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- Anchor positions & Fe‑RoPE rotations (anchor‑relative) ---
        anchors_pos = None
        assign = None

        shared = self._router_ctx is not None
        if shared:
            router = self._router_ctx
            anchors_pos, assign = router.anchors_pos, router.assign

        elif getattr(self.config, "ferope_anchor_relative", True) or getattr(self.config, "use_block_sparse", False):
            p = make_positions(B, T, device=x.device, normalize=True)
            anchors_pos, assign = build_pd_anchors_1d(
                T,
                getattr(self.config, "anchor_min_gap", 256),
                getattr(self.config, "anchor_jitter", 32),
                device=x.device
            )

        if getattr(self.config, "ferope_anchor_relative", True) and self.ferope_m > 0:
            if shared:
                delta = router.delta(B, normalize=True)
            else:
                p = make_positions(B, T, device=x.device, normalize=True)
                delta = anchor_local_offsets(p, anchors_pos, assign)
            q, k = ferope_ar_rotate(q, k, delta, self.rope_W[:self.ferope_m])  # [B,H,T,Dh]

        # --- Scaled dot‑product attention ---
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # CHG: block sparsity from shared router mask (if any)
        if getattr(self.config, "use_block_sparse", False) and (anchors_pos is not None):
            if shared:
                M = router.mask  # [T,T]
            else:
                A = int(anchors_pos.numel())
                rings = int(getattr(self.config, "neighbor_rings", 1))
                neighbor_ids = {a: set([a]) for a in range(A)}
                for a in range(A):
                    for r in range(1, rings + 1):
                        if a - r >= 0: neighbor_ids[a].add(a - r)
                        if a + r < A: neighbor_ids[a].add(a + r)
                M = allowed_anchor_pairs(assign, assign, neighbor_ids, T, T)  # [T,T]
            att = att.masked_fill(~M.view(1, 1, T, T), float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # weighted sum of values
        y = att @ v  # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, ctx=None):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class RegionFeedback(nn.Module):
    """Dense feedback along region graph (neighbor aggregate → residual)."""
    def __init__(self, d_model, dropout):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        # small gate init near 0 so it's opt-in via training
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, router: RegionRouter, hops:int=1):
        # x: [B,T,D] -> neighbor-aggregated anchor states, broadcast to tokens
        B,T,D = x.shape
        anchor = router.pool_per_anchor(x, reduce="mean")      # [B,A,D]
        agg    = router.neighbor_aggregate(anchor, hops=hops)  # [B,A,D]
        fb     = router.broadcast_from_anchor(agg)             # [B,T,D]
        return x + self.drop(self.proj(fb)) * self.gate.tanh()

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        kwta_frac = float(getattr(config, "kwta_frac", 0.0))
        self.kwta = KWTA(config.n_embd, int(config.n_embd * kwta_frac)) if kwta_frac > 0 else None
        self.dgate = DendriteGate(config.n_embd, config.n_embd, m=int(getattr(config, "dendrite_segments", 4))) if getattr(config, "use_dendrites", False) else None
        self.ff_head = FFHead(config.n_embd, mode=getattr(config, "ff_mode", "sumsq"))
        # NEW: optional region feedback head
        self.use_region_feedback = bool(getattr(config, "use_region_feedback", False))
        self.fb_hops = int(getattr(config, "feedback_hops", 1))
        self.fb = RegionFeedback(config.n_embd, config.dropout) if self.use_region_feedback else None
        self._router_ctx: Optional[RegionRouter] = None

    # NEW
    def set_router(self, router: Optional[RegionRouter]):
        self._router_ctx = router
        self.attn.set_router(router)

    def forward(self, x, ctx=None):
        x = x + self.attn(self.ln_1(x))
        # NEW: feedback residual based on shared router, if enabled
        if self.fb is not None and self._router_ctx is not None:
            x = self.fb(x, self._router_ctx, hops=self.fb_hops)
        h = self.mlp(self.ln_2(x))
        if self.dgate is not None:
            h = self.dgate(h, ctx if ctx is not None else x)
        if self.kwta is not None:
            h = self.kwta(h)
        x = x + h
        return x


# ---------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2.
    kwta_frac: float = 0.0
    use_dendrites: bool = False
    dendrite_segments: int = 4
    ff_mode: str = "sumsq"

    # Fe‑RoPE + PDS
    ferope_anchor_relative: bool = True
    anchor_min_gap: int = 256
    anchor_jitter: int = 32
    ferope_m: int = 64             # rotary pairs (2m dims)
    rope_base: float = 10000.0
    use_block_sparse: bool = True
    neighbor_rings: int = 1

    # Heads
    tie_weights: bool = True
    use_mfs_head: bool = True
    mfs_K: int = 3
    mfs_P: int = 4
    mfs_lowrank_r: int = 0

    # Auxiliary L4 losses
    add_l4_losses: bool = True
    l4_loss_weight: float = 0.1  # each; recon and next-step

    # Sharing router & feedback
    share_region_router: bool = True
    use_region_feedback: bool = False
    feedback_hops: int = 1

    # Region slate composition (project back to D)
    region_slate_include_error: bool = True
    region_slate_include_prior: bool = True

# ---------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Transformer core
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Fallback linear head (kept for compatibility when use_mfs_head=False)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Feature flags (safe defaults if not present)
        self._use_mfs_head   = bool(getattr(self.config, "use_mfs_head", True))
        self._add_l4_losses  = bool(getattr(self.config, "add_l4_losses", True))
        self._l4_loss_weight = float(getattr(self.config, "l4_loss_weight", 0.1))

        self._mfs_out_is_logprobs = None  # autodetected on first forward

        # Tie to input embedding if requested
        tie_weight = self.transformer.wte.weight if getattr(self.config, "tie_weights", True) else None

        # MFS LM head (Multi‑Facet Softmax; Fig. 2 of the paper)
        # Uses multi‑input facets + partitioned first softmax for ambiguous contexts.
        if self._use_mfs_head:
            self.lm_head_mfs = MFSHead(
                self.config.n_embd, self.config.vocab_size,
                K=int(getattr(self.config, "mfs_K", 3)),
                P=int(getattr(self.config, "mfs_P", 4)),
                slate_dim=self.config.n_embd,
                low_rank_r=int(getattr(self.config, "mfs_lowrank_r", 0)),
                tie_weight=tie_weight
            )
        # L4 auxiliary head
        if self._add_l4_losses:
            self.l4_head = SlidingWindowAE(
                self.config.n_embd, self.config.vocab_size,
                mfs_k=int(getattr(self.config, "mfs_K", 3)),   # NOTE: lower-case arg names
                mfs_p=int(getattr(self.config, "mfs_P", 4)),
                slate_dim=self.config.n_embd,
                low_rank_r=int(getattr(self.config, "mfs_lowrank_r", 0)),
                tie_weight=tie_weight
            )

        # init all weights
        self.apply(self._init_weights)
        # special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # slate helpers
        self._use_mfs_head   = bool(getattr(self.config, "use_mfs_head", True))
        self._add_l4_losses  = bool(getattr(self.config, "add_l4_losses", True))
        self._l4_loss_weight = float(getattr(self.config, "l4_loss_weight", 0.1))
        self._mfs_out_is_logprobs = None

        # NEW: project (pooled + prior + error_embed) -> D for the slate
        comp = 1  # pooled-by-anchor
        if getattr(self.config, "region_slate_include_prior", True): comp += 1
        if getattr(self.config, "region_slate_include_error", True): comp += 1
        self.region_slate_proj = nn.Linear(comp * self.config.n_embd, self.config.n_embd, bias=True)
        self.error_scalar_to_vec = nn.Linear(1, self.config.n_embd, bias=True)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward_acts(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        router = None
        want_router = bool(getattr(self.config, "share_region_router", False) or getattr(self.config, "use_block_sparse", False))
        if want_router:
            try:
                router = RegionRouter(T=t, device=device,
                                     anchor_min_gap=getattr(self.config, "anchor_min_gap", 256),
                                     anchor_jitter=getattr(self.config, "anchor_jitter", 32),
                                     neighbor_rings=int(getattr(self.config, "neighbor_rings", 1)))
            except NameError:
                router = None
        acts = {}
        for i, block in enumerate(self.transformer.h):
            if router is not None and hasattr(block, "set_router"):
                block.set_router(router)
            x = block(x, ctx=x)
            acts[i] = x
        return acts


    # ----------------------------
    # Slate (local and regional) for MFS facets (cheap)
    # ----------------------------
    def _make_local_slate(self, h: torch.Tensor, k:int=8) -> torch.Tensor:
        """Causal left average (current behavior)."""
        B, T, D = h.shape
        k = min(k, T)
        avg = F.avg_pool1d(h.transpose(1, 2), kernel_size=k, stride=1, padding=0).transpose(1, 2)
        if T - avg.size(1) > 0:
            pad_left = T - avg.size(1)
            left = h[:, :1, :].expand(B, pad_left, D)
            avg = torch.cat([left, avg], dim=1)
        return avg

    def _make_region_slate(
        self,
        h: torch.Tensor,
        router: RegionRouter,
        prior: Optional[torch.Tensor] = None,   # [B,T,D]
        error_scalar: Optional[torch.Tensor] = None  # [B,T,1]
    ) -> torch.Tensor:
        B,T,D = h.shape
        parts = [router.pool_broadcast(h, reduce="mean")]  # [B,T,D], pooled by anchor
        if getattr(self.config, "region_slate_include_prior", True):
            parts.append(prior if prior is not None else self._make_local_slate(h))
        if getattr(self.config, "region_slate_include_error", True):
            if error_scalar is None:
                err_vec = torch.zeros(B, T, 1, device=h.device, dtype=h.dtype)
            else:
                err_vec = error_scalar
            parts.append(self.error_scalar_to_vec(err_vec))  # [B,T,D]
        slate = torch.cat(parts, dim=-1)  # [B,T,comp*D]
        return self.region_slate_proj(slate)  # [B,T,D]

    # ----------------------------
    # Boilerplate
    # ----------------------------
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t,)

        # --- core transformer ---
        tok_emb = self.transformer.wte(idx)      # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)      # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Optional shared router (if available in this build)
        router = None
        want_router = bool(getattr(self.config, "share_region_router", False) or
                           getattr(self.config, "use_block_sparse", False))
        if want_router:
            try:
                router = RegionRouter(
                    T=t, device=device,
                    anchor_min_gap=getattr(self.config, "anchor_min_gap", 256),
                    anchor_jitter=getattr(self.config, "anchor_jitter", 32),
                    neighbor_rings=int(getattr(self.config, "neighbor_rings", 1)),
                )
            except NameError:
                router = None  # RegionRouter not defined in this build

        for block in self.transformer.h:
            if router is not None and hasattr(block, "set_router"):
                block.set_router(router)
            x = block(x, ctx=x)

        x = self.transformer.ln_f(x)             # final hidden [B,T,D]

        logits = None
        loss = None

        # --- Prepare slates ---
        # Cheap local slate (acts as L6 prior stand-in)
        if hasattr(self, "_make_local_slate"):
            local_slate = self._make_local_slate(x)
        else:
            local_slate = self._make_slate(x)  # backwards-compat

        region_slate = None
        error_scalar = None  # [B,T,1]
        aux = None

        # --- Optional L4 auxiliary losses + error scalar for region slate ---
        if getattr(self, "_add_l4_losses", False) and targets is not None and hasattr(self, "l4_head"):
            # Build next-step labels; ignore last position with -1
            next_targets = torch.roll(targets, shifts=-1, dims=1)
            next_targets[:, -1] = -1

            # (A) Proper CE/NLL auxiliaries (the AE returns log-probs internally)
            l_rec, l_nxt = self.l4_head(
                x, slate=local_slate,
                targets=targets, next_targets=next_targets,
                return_losses=True
            )
            if torch.is_tensor(l_rec): l_rec = l_rec.mean()
            if torch.is_tensor(l_nxt): l_nxt = l_nxt.mean()
            aux = l_rec + l_nxt

            # (B) Tokenwise next-step NLL (no-grad) to form error_scalar for the region slate
            with torch.no_grad():
                logp_rec, logp_nxt = self.l4_head(
                    x, slate=local_slate,
                    targets=targets, next_targets=next_targets,
                    return_losses=False
                )  # both are LOG-PROBS
                V = logp_nxt.size(-1)
                nll = F.nll_loss(
                    logp_nxt.view(-1, V), next_targets.view(-1),
                    reduction='none', ignore_index=-1
                )  # [B*T]
                error_scalar = nll.view(b, t, 1)  # [B,T,1]

        # --- Region slate (pooled anchors + prior + error) if available ---
        if getattr(self, "_use_mfs_head", False):
            if router is not None and hasattr(self, "_make_region_slate"):
                region_slate = self._make_region_slate(
                    x, router, prior=local_slate, error_scalar=error_scalar
                )
            else:
                region_slate = local_slate  # fallback
        else:
            region_slate = None

        # --- MFS or linear head path ---
        if getattr(self, "_use_mfs_head", False) and hasattr(self, "lm_head_mfs"):
            # IMPORTANT: ask the MFS head for LOG-PROBS to avoid ambiguous handling.
            mfs_out = self.lm_head_mfs(x, slate=region_slate, return_logprobs=True)   # [B,T,V] (LOG-PROBS)

            # Autodetect once (should now always be log-probs)
            if self._mfs_out_is_logprobs is None:
                with torch.no_grad():
                    s = mfs_out[0, 0].exp().sum().item()        # ≈1.0 if log-probs
                self._mfs_out_is_logprobs = (abs(s - 1.0) < 1e-3)
                print(f"[MFS] detected {'log-probs' if self._mfs_out_is_logprobs else 'logits'} from head "
                      f"(mean exp-sum ≈ {s:.4f}).")

            # For generation we can treat log-probs as logits (they are logits of a normalized dist)
            logits = mfs_out
            if targets is not None:
                V = mfs_out.size(-1)
                flat_tgt = targets.view(-1)
                # With log-probs, use NLL loss directly
                loss = F.nll_loss(mfs_out.view(-1, V), flat_tgt, ignore_index=-1)

        else:
            # Fallback linear head
            logits = self.lm_head(x)                             # [B,T,V] (logits)
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1
                )

        # --- Add AE aux (if computed) ---
        if aux is not None:
            if loss is None:
                loss = torch.zeros((), device=x.device, dtype=torch.float32)
            loss = loss + self._l4_loss_weight * aux

        assert logits is not None, "internal error: logits not set in forward()"
        return logits, loss

    # ----------------------------
    # Misc helpers
    # ----------------------------
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024   # always 1024 for GPT model checkpoints
        config_args['bias'] = True         # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]  # discard buffer

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not (k.endswith('.attn.masked_bias') or k.endswith('.attn.bias'))]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # "Conv1D" weights in HF need transpose
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits/log-probs for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
