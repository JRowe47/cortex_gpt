# sparse_routing.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def kwta(scores: torch.Tensor, k: int):
    """
    scores: [*, N]  -> binary mask [*, N] with exactly k winners per last dim
    """
    if k <= 0: return torch.zeros_like(scores, dtype=torch.bool)
    topk = torch.topk(scores, k=min(k, scores.shape[-1]), dim=-1)
    thr = torch.min(topk.values, dim=-1, keepdim=True).values
    return scores >= thr

@torch.no_grad()
def pds_nms_select(xyz: torch.Tensor, scores: torch.Tensor, radius: float, top_k: int):
    """
    Greedy blue-noise NMS in 3D (or 2D if z==0). xyz: [M,3], scores: [M]
    Returns indices of kept points.
    """
    order = torch.argsort(scores, descending=True)
    keep = []
    for i in order.tolist():
        if len(keep)==0:
            keep.append(i)
        else:
            d2 = (xyz[i] - xyz[keep]).pow(2).sum(-1)
            if torch.all(d2 > radius*radius):
                keep.append(i)
        if len(keep) >= top_k: break
    return torch.tensor(keep, device=xyz.device, dtype=torch.long)

def allowed_anchor_pairs(assign_q: torch.Tensor, assign_k: torch.Tensor, neighbor_ids: dict, Tq:int, Tk:int):
    """
    Build a [Tq,Tk] boolean mask indicating which q->k pairs are allowed under block sparsity.
    neighbor_ids: maps anchor_id -> set/list of allowed target anchor_ids (including itself).
    """
    device = assign_q.device
    M = torch.zeros(Tq, Tk, dtype=torch.bool, device=device)
    # For each (A,B) allowed, admit all time indices in A x B
    for A, neigh in neighbor_ids.items():
        qs = (assign_q == A).nonzero(as_tuple=True)[0]
        if qs.numel()==0: continue
        for B in neigh:
            ks = (assign_k == B).nonzero(as_tuple=True)[0]
            if ks.numel()==0: continue
            M[qs.unsqueeze(1), ks.unsqueeze(0)] = True
    return M

class SparseMessagePassing(nn.Module):
    """
    Single-step sparse inter-region message passing:
      - Projects neighbor features to messages
      - Averages over top-k neighbors (by similarity)
      - Merges local + neighbor message
    """
    def __init__(self, d_model: int, top_k_neighbors: int = 4):
        super().__init__()
        self.msg = nn.Linear(d_model, d_model)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.top_k = top_k_neighbors

    def forward(self, h_by_region: torch.Tensor, neighbor_indices: list[list[int]]):
        """
        h_by_region: [R, B, D]
        neighbor_indices: list of neighbors for each region r
        Returns:
          out: [R, B, D], aux: {'avg_fanin': float}
        """
        R, B, D = h_by_region.shape
        out = []
        fanins = []
        for r in range(R):
            nbrs = neighbor_indices[r]
            # Optional top-k by instantaneous cosine-like similarity
            if self.top_k and len(nbrs) > self.top_k:
                hr = h_by_region[r]  # [B, D]
                sims = []
                for j in nbrs:
                    sim = (hr * h_by_region[j]).mean(dim=-1).mean()  # scalar
                    sims.append(sim)
                top = torch.topk(torch.stack(sims), k=self.top_k).indices.tolist()
                nbrs = [nbrs[t] for t in top]
            if nbrs:
                msgs = [self.msg(h_by_region[j]) for j in nbrs]  # list of [B, D]
                m = torch.stack(msgs, dim=0).mean(dim=0)         # [B, D]
            else:
                m = torch.zeros_like(h_by_region[r])
            out.append(self.merge(torch.cat([h_by_region[r], m], dim=-1)))
            fanins.append(len(nbrs))
        out = torch.stack(out, dim=0)
        aux = {'avg_fanin': float(sum(fanins) / max(1, len(fanins)))}
        return out, aux