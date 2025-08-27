# cortex/facet_broadcast.py
# Unselected-Facet Diffusion (UFD) emitter and a lightweight broadcast router.

import torch
import torch.nn as nn
import torch.nn.functional as F

class FacetResidualEmitter(nn.Module):
    """
    Emits a compact broadcast vector from the *unselected* facets of a 7-facet head.
    Uses a factorized projection to compute facet states cheaply, then mixes the
    'leftover' (unselected) facet weights into a single vector for routing.

    Tie `share_gate` to the same gate used by your MultiFacetSoftmax for consistency.
    """
    def __init__(self,
                 d_model: int,
                 num_facets: int = 7,
                 proj_rank: int = 128,
                 top_m_selected: int = 2,
                 top_u_unselected: int = 2,
                 tau_broadcast: float = 1.2,
                 pool: str = "last",
                 share_gate: nn.Linear | None = None,
                 state_dim: int | None = None):
        super().__init__()
        self.K = num_facets
        self.top_m = top_m_selected
        self.top_u = top_u_unselected
        self.tau_broadcast = tau_broadcast
        self.pool = pool
        self.state_dim = state_dim or d_model

        self.gate = share_gate or nn.Linear(d_model, self.K)
        self.U = nn.Linear(d_model, proj_rank, bias=False)  # shared basis
        self.V = nn.Parameter(torch.randn(self.K, proj_rank, self.state_dim) / (proj_rank ** 0.5))
        self.norm = nn.LayerNorm(self.state_dim)
        self.compress = nn.Linear(self.state_dim, self.state_dim)

    def _gates(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(1)
        return F.softmax(self.gate(h), dim=-1)  # [B, T, K]

    def states_and_leftover(self, h: torch.Tensor):
        """
        Returns:
          states:   [B, T, K, D] per-facet states
          leftover: [B, T, K]    renormalized unselected facet weights (with optional top-U)
          G:        [B, T, K]    prior gates
          seli:     [B, T, m]    indices of selected facets (top-m)
        """
        squeeze = (h.dim() == 2)
        if squeeze:
            h = h.unsqueeze(1)
        B, T, D = h.shape
        G = self._gates(h)  # [B, T, K]
        m = min(self.top_m, self.K)
        _, seli = G.topk(m, dim=-1)
        sel_mask = torch.zeros_like(G, dtype=torch.bool).scatter_(-1, seli, True)
        unselected = torch.where(sel_mask, torch.zeros_like(G), G)
        if self.top_u and self.top_u < self.K - m:
            uv, ui = unselected.topk(self.top_u, dim=-1)
            u_mask = torch.zeros_like(G, dtype=torch.bool).scatter_(-1, ui, True)
            unselected = torch.where(u_mask, unselected, torch.zeros_like(unselected))
        # Renormalize 'leftover' and apply temperature
        leftover = unselected / (unselected.sum(-1, keepdim=True).clamp_min(1e-9))
        leftover = F.softmax(torch.log(leftover.clamp_min(1e-9)) / self.tau_broadcast, dim=-1)
        # Factorized per-facet states
        basis = self.U(h)  # [B, T, R]
        states = torch.einsum('btr,krd->btkd', basis, self.V)  # [B, T, K, D]
        states = self.norm(states)
        return states, leftover, G, seli

    def forward(self, h: torch.Tensor):
        """
        h: [B, D] or [B, T, D] region features.
        Returns:
          broadcast: [B, D] (pooled over time dimension)
          diag: dict with debug info (currently just gates)
        """
        states, leftover, G, _ = self.states_and_leftover(h)
        b = torch.einsum('btk,btkd->btd', leftover, states)  # [B, T, D]
        b = self.compress(b)
        if self.pool == "last":
            b = b[:, -1, :]
        elif self.pool == "mean":
            b = b.mean(dim=1)
        else:
            raise ValueError("pool must be 'last' or 'mean'")
        return b, {'gates': G.detach()}
    

class BroadcastRouter(nn.Module):
    """
    Routes broadcast vectors over a sparse neighbor graph.
    Mixes local and neighbor aggregate with a tiny linear.
    """
    def __init__(self, d_model: int, top_k_neighbors: int = 4):
        super().__init__()
        self.top_k = top_k_neighbors
        self.mix = nn.Linear(2 * d_model, d_model)

    def forward(self,
                bcast_by_region: torch.Tensor,    # [R, B, D]
                feats_by_region: torch.Tensor,    # [R, B, D]
                neighbor_indices: list[list[int]]):
        R, B, D = feats_by_region.shape
        msgs = []
        for r in range(R):
            nbrs = neighbor_indices[r]
            if self.top_k and len(nbrs) > self.top_k:
                sims = []
                hr = feats_by_region[r]  # [B, D]
                for j in nbrs:
                    sim = (hr * feats_by_region[j]).mean().unsqueeze(0)  # scalar
                    sims.append(sim)
                top = torch.topk(torch.cat(sims), k=self.top_k).indices.tolist()
                nbrs = [nbrs[t] for t in top]
            if nbrs:
                agg = torch.stack([bcast_by_region[j] for j in nbrs], dim=0).mean(dim=0)
            else:
                agg = torch.zeros_like(bcast_by_region[r])
            msgs.append(self.mix(torch.cat([bcast_by_region[r], agg], dim=-1)))
        return torch.stack(msgs, dim=0)  # [R, B, D]
