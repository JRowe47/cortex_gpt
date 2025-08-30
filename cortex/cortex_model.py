# cortex/cortex_model.py
# Integrated neocortical sensorimotor recurrent system with sparsity, UFD, and surprise.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from cortex.multitau import MultiTauState, RegionMemoryKV
from cortex.facet_broadcast import FacetResidualEmitter, BroadcastRouter
from cortex.surprise_aha import SurpriseMeter, AhaDiffuser, BurstBudget
from cortex.region_sparsity_gate import RegionSparsityGate
from mfs import MultiFacetSoftmax
from sparse_routing import SparseMessagePassing

class Region(nn.Module):
    """Region with multi-timescale state and pose-aware heads."""

    def __init__(self, d_model: int, d_pose: int = 12, memory_cap: int = 128):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.state = MultiTauState(d_model)
        self.pose_head = nn.Linear(d_model, d_pose)
        self.predictor = nn.Linear(d_model, d_model)
        self.kv = RegionMemoryKV(memory_cap, key_dim=d_model, val_dim=d_model)

    def forward(self, x: torch.Tensor, neighbor_msg: torch.Tensor | None = None, alpha_boost: float = 1.0):
        """x: [B,D]; neighbor_msg: [B,D] or None"""
        if neighbor_msg is None:
            neighbor_msg = torch.zeros_like(x)
        h = self.ff(x)
        s = self.state(h + neighbor_msg)
        pose = self.pose_head(s)
        pred = self.predictor(s)
        aux = {'pose': pose.detach(), 'pred': pred.detach()}
        return h + s, aux


class CortexModel(nn.Module):
    """
    A 2% active, hex-tiling based recurrent model with:
      - 7-facet LM head (Mixture-of-Softmax)
      - Unselected-Facet Diffusion (UFD)
      - Surprise-based "Aha" diffuser & burst budgeting
      - Region sparsity gate (hex-NMS k-WTA + refractory/feedback/homeostasis)
      - Sparse message passing across hex graph
    """
    def __init__(self,
                 R: int,
                 d_model: int,
                 neighbor_indices: List[List[int]],
                 io_idxs: Dict[str, int],
                 vocab_size: int,
                 num_facets: int = 7,
                 top_m_facets: int = 2,
                 k_active: int = 6,
                 router_top_k: int = 4):
        super().__init__()
        self.R = R
        self.d_model = d_model
        self.neighbor_indices = neighbor_indices
        self.io_idxs = io_idxs

        self.regions = nn.ModuleList([Region(d_model) for _ in range(R)])

        # Sparse router among regions
        self.router = SparseMessagePassing(d_model, top_k_neighbors=router_top_k)

        # 7-facet head
        self.mfs = MultiFacetSoftmax(d_model, vocab_size, num_facets=num_facets, top_m=top_m_facets)
        self.motor_proj = nn.Linear(d_model, d_model)

        # UFD + surprise
        self.emitter = FacetResidualEmitter(d_model, num_facets=num_facets, top_m_selected=top_m_facets, share_gate=self.mfs.gate)
        self.surprise = SurpriseMeter(self.mfs)
        self.aha = AhaDiffuser(self.emitter, self.surprise, s_thresh=0.7, boost_gain=2.0)
        self.burster = BurstBudget(S_thresh=0.2, hard_cap=0.6, H=2)

        # Broadcast router
        self.bcast_router = BroadcastRouter(d_model, top_k_neighbors=router_top_k)

        # Region sparsity (~2%)
        self.gate = RegionSparsityGate(R, d_model, neighbor_indices, k_active=k_active,
                                       io_keep=[io_idxs.get('sensor', 0), io_idxs.get('motor', R - 1)])

        # neighbor message buffer (initialized lazily per-batch)
        self._neighbor_msg_prev: torch.Tensor | None = None

    def _ensure_neighbor_buf(self, B: int, device):
        if self._neighbor_msg_prev is None or self._neighbor_msg_prev.size(1) != B or self._neighbor_msg_prev.device != device:
            self._neighbor_msg_prev = torch.zeros(self.R, B, self.d_model, device=device)

    def forward(self, x_per_region: torch.Tensor, targets: torch.Tensor | None = None):
        """
        x_per_region: [R, B, D] input features per region (sensor regions get embeddings; others zeros)
        targets: next-token targets [B] (or [B, T] if you drive multiple steps)
        Returns: (logp, loss, aux)
        """
        R, B, D = x_per_region.shape
        device = x_per_region.device
        self._ensure_neighbor_buf(B, device)

        # 1) Per-region compute + state update with last step's neighbor messages
        H_list, s_aux_all = [], []
        for r, reg in enumerate(self.regions):
            out, saux = reg(x_per_region[r], neighbor_msg=self._neighbor_msg_prev[r])
            H_list.append(out)
            s_aux_all.append(saux)
        H = torch.stack(H_list, dim=0)  # [R, B, D]

        # 2) Ultra-sparse region gating (~2%) before routing
        Hs, reg_mask, adj_scores, gate_diag = self.gate(H,
                                                        neighbor_msg=self._neighbor_msg_prev,
                                                        burst_extra_k=0,
                                                        io_force_on=True)

        # 3) Sparse inter-region message passing among active regions
        H2, route_aux = self.router(Hs, self.neighbor_indices)

        # 4) Motor readout (average of designated region(s)); compute LM outputs
        motor_idx = self.io_idxs['motor']
        motor_state = self.motor_proj(H2[motor_idx])
        if motor_state.dim() == 3:
            motor_state = motor_state.mean(dim=0)
        # The MFS head always returns three outputs with ``loss`` set to ``None``
        # when ``targets`` are not supplied.  This keeps the interface
        # consistent between training and inference.
        logp, loss, mfs_aux = self.mfs(motor_state, targets=targets)

        # 5) Surprise/aha -> build broadcast messages
        if targets is not None:
            _, _, S, _, _ = self.surprise(motor_state, targets)
            burst_now = self.burster.burst_signal(S)
        else:
            burst_now = 0.0

        # 6) Surprise-aware UFD broadcast from each region; queue for next tick
        b_list = []
        for r in range(self.R):
            if targets is not None:
                b, _ = self.aha(H2[r], targets=targets)
            else:
                b, _ = self.emitter(H2[r])
            b_list.append(b)
        Bcast = torch.stack(b_list, dim=0)  # [R, B, D]
        neighbor_msg_next = self.bcast_router(Bcast, H2, self.neighbor_indices)
        self._neighbor_msg_prev = neighbor_msg_next.detach()  # used at next forward step

        aux = {
            'state': s_aux_all,
            'routing': route_aux,
            'mfs': mfs_aux,
            'gate': gate_diag,
            'burst_now': burst_now
        }
        return logp, loss, aux
