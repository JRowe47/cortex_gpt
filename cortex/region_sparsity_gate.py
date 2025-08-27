# cortex/region_sparsity_gate.py
# Ultra-sparse region activation with hex-NMS, refractory & feedback inhibition, and homeostasis.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Iterable, Optional

class RegionSparsityGate(nn.Module):
    """
    Selects ~k_active regions per step (≈2% of total) using:
      - Greedy hex-graph non-maximum suppression (prevents adjacent co-activation)
      - Refractory inhibition (recent activity suppressed)
      - Feedback inhibition (recent incoming routed message suppressed)
      - Per-region homeostatic thresholds (long-run firing -> target_rate)

    Designed to run *inside* the normal forward pass (no extra model pass).
    """
    def __init__(self,
                 R: int,
                 d_model: int,
                 neighbor_indices: List[List[int]],
                 k_active: int = 6,
                 io_keep: Optional[Iterable[int]] = None,
                 use_learned_score: bool = True,
                 refractory_decay: float = 0.9,
                 feedback_decay: float = 0.9,
                 refractory_weight: float = 1.0,
                 feedback_weight: float = 0.5,
                 homeo_lr: float = 1e-3,
                 target_rate: Optional[float] = None):
        super().__init__()
        self.R = R
        self.d = d_model
        self.neighbor_indices = neighbor_indices
        self.k_active = k_active
        self.io_keep = set(io_keep) if io_keep is not None else set()
        self.use_learned = use_learned_score
        self.refractory_decay = refractory_decay
        self.feedback_decay = feedback_decay
        self.refractory_weight = refractory_weight
        self.feedback_weight = feedback_weight
        self.homeo_lr = homeo_lr
        self.target_rate = target_rate or (k_active / float(R))

        if self.use_learned:
            self.score = nn.Linear(d_model, 1, bias=False)
        else:
            self.register_parameter('score', None)

        self.register_buffer('theta', torch.zeros(R))
        self._refractory = None  # [B, R]
        self._feedback = None    # [B, R]

    def _ensure_state(self, B: int, device):
        if self._refractory is None or self._refractory.size(0) != B or self._refractory.device != device:
            self._refractory = torch.zeros(B, self.R, device=device)
            self._feedback = torch.zeros(B, self.R, device=device)

    def _scores(self, H: torch.Tensor) -> torch.Tensor:
        # H: [R, B, D] -> [B, R]
        if self.use_learned:
            R, B, D = H.shape
            return self.score(H.reshape(R * B, D)).reshape(R, B).transpose(0, 1).contiguous()
        # Norm-based fallback
        return H.norm(dim=-1).transpose(0, 1).contiguous()

    @staticmethod
    def _greedy_hex_nms(scores: torch.Tensor, neighbor_indices: List[List[int]], k: int, force_on=None):
        # scores: [B, R] -> mask [B, R]
        B, R = scores.shape
        device = scores.device
        force_on = set(force_on or [])
        mask = torch.zeros(B, R, device=device)
        for b in range(B):
            selected = set(force_on)
            suppressed = set()
            order = torch.argsort(scores[b], descending=True).tolist()
            for idx in order:
                if len(selected) >= k + len(force_on):
                    break
                if idx in selected or idx in suppressed:
                    continue
                selected.add(idx)
                suppressed.add(idx)
                for n in neighbor_indices[idx]:
                    suppressed.add(n)
            if selected:
                mask[b, list(selected)] = 1.0
        return mask

    def forward(self,
                H: torch.Tensor,
                neighbor_msg: torch.Tensor | None = None,
                burst_extra_k: int = 0,
                io_force_on: bool = True):
        """
        H: [R, B, D] features per region
        neighbor_msg: [R, B, D] routed messages from last step (for feedback inhibition)
        burst_extra_k: increase k_active temporarily (e.g., surprise burst)
        Returns:
          H_masked: [R, B, D] features masked by STE gate
          hard_mask: [B, R]   binary selection per batch
          adj_scores: [B, R]  adjusted scores before NMS
          diag: dict
        """
        R, B, D = H.shape
        device = H.device
        self._ensure_state(B, device)

        s = self._scores(H)  # [B, R]
        fb_mag = neighbor_msg.norm(dim=-1).transpose(0, 1) if neighbor_msg is not None else torch.zeros_like(s)

        adj = s - self.theta.view(1, R)
        if self.refractory_weight:
            adj = adj - self.refractory_weight * self._refractory
        if self.feedback_weight:
            self._feedback = self.feedback_decay * self._feedback + (1 - self.feedback_decay) * fb_mag
            adj = adj - self.feedback_weight * self._feedback

        k = int(self.k_active + max(0, burst_extra_k))
        force_on = self.io_keep if io_force_on else set()
        hard = self._greedy_hex_nms(adj, self.neighbor_indices, k, force_on=force_on)

        # Update traces + homeostasis
        self._refractory = self.refractory_decay * self._refractory + (1 - self.refractory_decay) * hard
        mean_act = hard.mean(dim=0)  # [R]
        with torch.no_grad():
            self.theta += self.homeo_lr * (mean_act - self.target_rate)

        # Straight-Through Estimator mask
        soft = torch.sigmoid(adj)
        ste = hard + soft - soft.detach()
        Hs = H * ste.transpose(0, 1).unsqueeze(-1)

        diag = {
            'k_selected_avg': float(hard.sum(dim=1).mean().item()),
            'mean_theta': float(self.theta.mean().item()),
            'mean_refractory': float(self._refractory.mean().item()),
            'mean_feedback': float(self._feedback.mean().item())
        }
        return Hs, hard, adj, diag
