from __future__ import annotations
import torch

@torch.no_grad()
def apply_column_multipliers(
    named_params,
    col_lr_mult: dict[int|None, float],
    col_wd_mult: dict[int|None, float],
    weight_decay_base: float,
    lr_scale: float | None = None,
):
    """
    Cheap per-parameter hook to emulate HTM-AdamW metaplasticity without changing your optimizer:

      • Scales grads in-place (lr mult).
      • Applies decoupled weight decay (wd mult) *scaled by current LR*, just like AdamW.

    Usage:
        apply_column_multipliers(model.named_parameters(), lr_mults, wd_mults,
                                 weight_decay_base=args.wd, lr_scale=current_lr)
        optimizer.step()

    Notes:
      - Each parameter may carry a ._col_id (int) set by your column router; defaults to None.
      - If lr_scale is None, WD is skipped (LR is needed to keep units consistent with AdamW).
    """
    lr_scale = 1.0 if (lr_scale is None) else float(lr_scale)
    do_wd = (weight_decay_base > 0.0) and (lr_scale > 0.0)

    for name, p in named_params:
        if p.grad is None or not p.requires_grad:
            continue
        if not torch.isfinite(p.grad).all():
            continue

        col = getattr(p, "_col_id", None)
        lr_m = float(col_lr_mult.get(col, 1.0))
        wd_m = float(col_wd_mult.get(col, 0.0))

        # gradient scale
        p.grad.mul_(lr_m)

        # decoupled weight decay, AdamW-style (scale by current LR)
        if do_wd and wd_m != 0.0:
            # p <- p - (lr * wd_base * wd_mult) * p
            p.data.add_(p.data, alpha=-(weight_decay_base * wd_m) * lr_scale)


class ColumnMetaController:
    """
    Tracks per-column success and emits lr/wd multipliers each step.
    Columns are arbitrary integers (e.g., block idx, head idx).
    """
    def __init__(self, lr_boost=1.1, wd_boost=0.0, decay_mult=1.0, ema=0.9, success_floor=0.05):
        self.lr_boost = float(lr_boost)
        self.wd_boost = float(wd_boost)
        self.decay_mult = float(decay_mult)
        self.ema = float(ema)
        self.success_floor = float(success_floor)
        self._ema_scores: dict[int, float] = {}

    def update_scores(self, col_scores: dict[int, float]):
        # EMA over observed success scores in [0,1]
        for c, s in col_scores.items():
            s = max(0.0, min(1.0, float(s)))
            prev = self._ema_scores.get(c, s)
            self._ema_scores[c] = self.ema * prev + (1.0 - self.ema) * s

    def compute_multipliers(self) -> tuple[dict[int, float], dict[int, float]]:
        col_lr, col_wd = {}, {}
        for c, s in self._ema_scores.items():
            if s >= self.success_floor:
                col_lr[c] = self.lr_boost
                col_wd[c] = 0.0
            else:
                col_lr[c] = 1.0
                col_wd[c] = self.wd_boost if self.wd_boost > 0.0 else 0.0
        return col_lr, col_wd
