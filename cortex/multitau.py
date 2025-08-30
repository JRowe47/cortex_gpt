import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULite(nn.Module):
    """A minimal GRU-style gate with a single sigmoid input gate."""
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, s_prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Interpolate between previous state and candidate x."""
        g = torch.sigmoid(self.gate(torch.cat([s_prev, x], dim=-1)))
        return g * x + (1 - g) * s_prev


class MultiTauState(nn.Module):
    """Multi-timescale exponential traces with learnable mixing."""
    def __init__(self, d_model: int, taus: tuple[int, ...] = (4, 16, 64, 256)):
        super().__init__()
        self.d_model = d_model
        self.taus = taus
        self.K = len(taus)
        init_alpha = torch.logit(torch.tensor([1.0 / t for t in taus], dtype=torch.float32))
        self.logit_alpha = nn.Parameter(init_alpha)  # per-timescale update rate
        self.beta = nn.Parameter(torch.ones(self.K))  # mixing weights
        self.gru = GRULite(d_model)
        self.register_buffer('traces', None)  # [K,B,D]
        self.register_buffer('state', None)   # [B,D]

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.traces = torch.zeros(self.K, batch_size, self.d_model, device=device, dtype=dtype)
        self.state = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        if self.traces is None or self.traces.size(1) != B or self.traces.device != x.device:
            self.reset_state(B, x.device, x.dtype)

        alpha = torch.sigmoid(self.logit_alpha).view(self.K, 1, 1)
        x_exp = x.unsqueeze(0)
        self.traces = (1 - alpha) * self.traces + alpha * x_exp

        beta = torch.softmax(self.beta, dim=0).view(self.K, 1, 1)
        mixed = (beta * self.traces).sum(0)  # [B,D]

        self.state = self.gru(self.state, mixed)
        return self.state


class RegionMemoryKV(nn.Module):
    """Simple key-value memory with fixed capacity per region."""
    def __init__(self, capacity: int, key_dim: int, val_dim: int):
        super().__init__()
        self.capacity = capacity
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.register_buffer('keys', torch.zeros(capacity, key_dim))
        self.register_buffer('vals', torch.zeros(capacity, val_dim))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

    def write(self, key: torch.Tensor, val: torch.Tensor) -> None:
        idx = int(self.ptr.item()) % self.capacity
        self.keys[idx] = key.detach()
        self.vals[idx] = val.detach()
        self.ptr[0] = (self.ptr + 1) % self.capacity

    def read(self, key: torch.Tensor) -> torch.Tensor:
        if self.ptr.item() == 0:
            return torch.zeros(self.val_dim, device=key.device, dtype=key.dtype)
        sims = F.cosine_similarity(key.unsqueeze(0), self.keys, dim=-1)
        idx = torch.argmax(sims)
        return self.vals[idx]
