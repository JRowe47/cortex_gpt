# cortex/statecell.py
# Slot-based state with fast/slow traces and neighbor-aware updates.

import torch
import torch.nn as nn
import torch.nn.functional as F

class StateCell(nn.Module):
    """
    Slot-based persistent state with:
      - Slow state (persistent) and fast state (short-term)
      - Gated updates with alpha_min to ensure each pass updates
      - Neighbor-aware write (incorporates routed context)
    """
    def __init__(self, d_model: int, n_slots: int = 8, alpha_min: float = 0.02, fast_decay: float = 0.95):
        super().__init__()
        self.n_slots = n_slots
        self.d = d_model
        self.alpha_min = alpha_min
        self.fast_decay = fast_decay

        self.keys = nn.Parameter(torch.randn(n_slots, d_model) / (d_model ** 0.5))
        self.read_proj = nn.Linear(d_model, d_model)
        self.write = nn.Linear(2 * d_model, d_model)
        self.ctrl = nn.Linear(2 * d_model, 2)  # produces [add_gate, erase_gate]

        self.register_buffer('state', torch.zeros(n_slots, d_model))
        self.register_buffer('fast', torch.zeros(n_slots, d_model))

    def reset_state(self, device=None):
        dev = device or self.state.device
        with torch.no_grad():
            self.state.zero_()
            self.fast.zero_()

    def forward(self, h: torch.Tensor, neighbor_msg: torch.Tensor | None = None, alpha_boost: float = 1.0):
        """
        h: [B, D]
        neighbor_msg: [B, D] or None
        alpha_boost: multiplicative gain (e.g., from surprise bursts)
        Returns: read vector [B, D], aux dict
        """
        if neighbor_msg is None:
            neighbor_msg = torch.zeros_like(h)

        q = self.read_proj(h)                      # [B, D]
        attn = F.softmax(q @ self.keys.T, dim=-1) # [B, n_slots]
        read = attn @ (self.state + self.fast)    # [B, D]

        add, erase = self.ctrl(torch.cat([h, read], dim=-1)).chunk(2, dim=-1)
        add = torch.sigmoid(add)     # [B, 1] broadcastable
        erase = torch.sigmoid(erase) # not used directly; reserved for future

        # Ensure non-zero update each step; scale by alpha_boost (burst)
        alpha = ((self.alpha_min + (1 - self.alpha_min) * add.mean()).clamp_(0, 1) * alpha_boost).clamp_(0, 1)

        delta = attn.transpose(0, 1) @ self.write(torch.cat([h + neighbor_msg, read], dim=-1))  # [n_slots, D]

        # Update slow and fast states
        self.state = (1 - alpha) * self.state + alpha * delta
        self.fast = self.fast_decay * self.fast + (1 - self.fast_decay) * delta

        aux = {'attn': attn.detach(), 'alpha': alpha.detach()}
        return read, aux
