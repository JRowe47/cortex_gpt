import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFHead(nn.Module):
    """Small reusable goodness head for Forward-Forward learning."""

    def __init__(self, d: int, mode: str = "sumsq", temp: float = 1.0, k: int = 0):
        super().__init__()
        self.mode = mode
        self.temp = temp
        self.k = k
        if mode in ("proto_ce", "linear_logit"):
            if k <= 0:
                raise ValueError("mode requires k>0")
            self.proj = nn.Linear(d, k, bias=(mode == "linear_logit"))
        elif mode != "sumsq":
            raise ValueError(f"unknown mode {mode}")

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Return goodness scores for activations h [B,D]."""
        if self.mode == "sumsq":
            h = F.layer_norm(h, (h.shape[-1],))
            return h.pow(2).mean(-1)
        else:
            z = self.proj(h) / self.temp
            return z


def ff_loss(good: torch.Tensor, bad: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    return F.softplus(-(good - bad - margin)).mean()


class KWTA(nn.Module):
    """k-Winners-Take-All with simple homeostatic boosting."""

    def __init__(self, d: int, k: int, alpha: float = 0.01, gamma: float = 1.0,
                 target_sparsity: Optional[float] = None):
        super().__init__()
        self.k = max(1, k)
        self.alpha = alpha
        self.gamma = gamma
        self.target = target_sparsity if target_sparsity is not None else k / d
        self.register_buffer("duty", torch.zeros(1, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = x.topk(self.k, dim=-1)
        mask = torch.zeros_like(x).scatter(-1, topk_idx, 1.0)
        # update duty cycle
        self.duty.mul_(1 - self.alpha).add_(self.alpha * mask.mean(dim=0, keepdim=True))
        boost = torch.exp(-self.gamma * (self.duty - self.target))
        x = x * boost
        return x * mask


class DendriteGate(nn.Module):
    """Active dendrite-style context gating."""

    def __init__(self, d_ctx: int, d_hid: int, m: int = 4):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(m, d_ctx) / math.sqrt(d_ctx))
        self.gain = nn.Linear(m, d_hid, bias=False)
        self.theta = nn.Parameter(torch.zeros(m))

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        ctx_n = F.normalize(ctx, dim=-1)
        key_n = F.normalize(self.keys, dim=-1)
        s = F.linear(ctx_n, key_n) - self.theta  # [B,m]
        s = F.relu(s)
        g = torch.sigmoid(self.gain(s))
        return h * g


class UnionSDR:
    """Maintain a sparse union over steps."""

    def __init__(self, d: int, decay: float = 0.9):
        self.decay = decay
        self.buf = None
        self.d = d

    def update(self, mask: torch.Tensor) -> torch.Tensor:
        if self.buf is None:
            self.buf = mask.clone()
        else:
            self.buf.mul_(self.decay).clamp_(0, 1)
            self.buf = torch.maximum(self.buf, mask)
        return self.buf


def anomaly_scale(anomaly: float, base_lr: float, base_beta2: float,
                  k1: float = 0.1, k2: float = 0.1) -> Tuple[float, float]:
    lr = base_lr * (1.0 + k1 * anomaly)
    beta2 = max(0.8, base_beta2 - k2 * anomaly)
    return lr, beta2
