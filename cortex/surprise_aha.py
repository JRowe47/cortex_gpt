# cortex/surprise_aha.py
# Surprise metrics from a MultiFacetSoftmax and an "Aha" diffuser w/ burst budgeting.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SurpriseMeter(nn.Module):
    """
    Computes:
      r: responsibilities over facets, given observed target (posterior)
      s: per-facet "surprise" advantage s_k = log p_k(y) - log p_mix(y)
      S: global surprise KL(r || g)
    Requires a MultiFacetSoftmax-like head with attributes: K (num facets), gate, facets (list of Linear).
    """
    def __init__(self, mfs, eps: float = 1e-9):
        super().__init__()
        self.mfs = mfs
        self.eps = eps

    @torch.no_grad()
    def forward(self, h: torch.Tensor, targets: torch.Tensor):
        """
        h: [B, D] or [B, T, D]
        targets: [B] or [B, T]
        Returns: r [B,T,K], s [B,T,K], S [B,T], g [B,T,K], seli [B,T,m]
        """
        squeeze = (h.dim() == 2)
        if squeeze:
            h = h.unsqueeze(1)
        B, T, D = h.shape
        K = self.mfs.K

        g = F.softmax(self.mfs.gate(h), dim=-1)  # [B, T, K]
        m = self.mfs.top_m if self.mfs.top_m is not None else min(2, K)
        _, seli = g.topk(m, dim=-1)

        # Gather per-facet log prob for the target only (no full log-softmax over V for loss here)
        N = B * T
        h2 = h.reshape(N, D)
        t_flat = targets.reshape(N)

        logp_k = []
        for k in range(K):
            logits_k = self.mfs.facets[k](h2)                              # [N, V]
            lp_k = F.log_softmax(logits_k, dim=-1)
            lp_k_y = lp_k.gather(-1, t_flat.unsqueeze(-1)).squeeze(-1)      # [N]
            logp_k.append(lp_k_y)
        logp_k = torch.stack(logp_k, dim=-1).reshape(B, T, K)               # [B, T, K]

        logg = torch.log(g.clamp_min(self.eps))
        log_mix = torch.logsumexp(logg + logp_k, dim=-1)                     # [B, T]
        log_r = logg + logp_k - log_mix.unsqueeze(-1)                        # [B, T, K]
        r = torch.exp(log_r)
        s = logp_k - log_mix.unsqueeze(-1)                                   # [B, T, K]
        S = (r * (log_r - logg)).sum(dim=-1)                                 # [B, T]

        return r, s, S, g, seli


class BurstBudget:
    """
    Maps global surprise S into a short-lived "burst" scalar in [0, 1],
    which you can translate into +Δk active regions, larger alpha for StateCell, etc.
    """
    def __init__(self, S_thresh: float = 0.2, hard_cap: float = 0.6, H: int = 2):
        self.S_thresh = S_thresh
        self.hard_cap = hard_cap
        self.H = H

    def burst_signal(self, S: torch.Tensor) -> float:
        s = float(S.mean().item())
        if s <= self.S_thresh:
            return 0.0
        x = min(s, self.hard_cap)
        return (x - self.S_thresh) / max(1e-9, (self.hard_cap - self.S_thresh))


class AhaDiffuser(nn.Module):
    """
    Surprise-aware variant of the FacetResidualEmitter:
      - Boosts unselected facets whose s_k exceeds s_thresh.
      - Optionally co-propagates the top selected facet when an aha occurs.
    """
    def __init__(self,
                 emitter,                      # FacetResidualEmitter
                 surprise_meter: SurpriseMeter,
                 s_thresh: float = 0.7,
                 boost_gain: float = 2.0,
                 pair_with_selected: bool = True,
                 max_pairs: int = 1):
        super().__init__()
        self.emitter = emitter
        self.surprise = surprise_meter
        self.s_thresh = s_thresh
        self.boost_gain = boost_gain
        self.pair = pair_with_selected
        self.max_pairs = max_pairs

    def forward(self, h: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # Inference fallback: no targets -> plain emitter
        if targets is None:
            return self.emitter(h)

        # Base emitter products
        b0, diag0 = self.emitter(h)
        states, leftover, G, seli = self.emitter.states_and_leftover(h)
        r, s, S, g, _ = self.surprise(h, targets)

        if h.dim() == 2:
            h = h.unsqueeze(1)
        K = g.size(-1)
        m = self.emitter.top_m or min(2, K)
        _, seli2 = G.topk(m, dim=-1)
        sel_mask = torch.zeros_like(G, dtype=torch.bool).scatter_(-1, seli2, True)

        aha_mask = (s > self.s_thresh) & (~sel_mask)  # only boost unselected facets
        boosted = leftover * torch.where(aha_mask, torch.full_like(leftover, self.boost_gain),
                                         torch.ones_like(leftover))
        if self.pair:
            top_sel = seli2[..., :self.max_pairs]
            sel_add = torch.zeros_like(boosted)
            sel_add.scatter_(-1, top_sel, 0.5)
            boosted = torch.where(aha_mask.any(dim=-1, keepdim=True), boosted + sel_add, boosted)

        boosted = boosted / boosted.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        b = torch.einsum('btk,btkd->btd', boosted, states)
        b = self.emitter.compress(self.emitter.norm(b))
        b = b[:, -1, :]

        diag = {'aha_frac': float(aha_mask.float().mean().item()),
                'S_mean': float(S.mean().item()),
                **diag0}
        return b, diag
