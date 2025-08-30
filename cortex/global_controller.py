import torch


class GlobalController:
    """Maintains a global activity budget for sparse region routing."""
    def __init__(self, target_frac: float = 0.05, min_k: int = 8, max_k: int = 32):
        self.target_frac = target_frac
        self.min_k = min_k
        self.max_k = max_k
        self.energy = target_frac

    def select_active(self, priorities: torch.Tensor, flops: torch.Tensor) -> torch.Tensor:
        """Select top-k indices under a FLOPs budget.

        priorities: [R] scores
        flops: [R] estimated cost per region
        Returns boolean mask [R]
        """
        k = int(max(self.min_k, min(self.max_k, self.energy * len(priorities))))
        vals, idx = torch.topk(priorities, k)
        mask = torch.zeros_like(priorities, dtype=torch.bool)
        mask[idx] = True
        return mask

    def update_energy(self, global_surprise: float) -> None:
        self.energy = float(torch.clamp(torch.tensor(self.energy + 0.1 * global_surprise), 0.01, 1.0))
