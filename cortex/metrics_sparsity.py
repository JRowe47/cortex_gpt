# cortex/metrics_sparsity.py
# Simple FLOPs/active-parameter accounting helper.

from collections import defaultdict

class FlopsMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.flops = 0
        self.active_params = defaultdict(int)

    def linear(self, in_features: int, out_features: int, used: bool = True, tag: str = 'linear'):
        if used:
            self.flops += 2 * in_features * out_features
            self.active_params[tag] += in_features * out_features

    def report(self):
        total = int(sum(self.active_params.values()))
        return {
            'flops': int(self.flops),
            'active_params_total': total,
            'active_params_by_tag': dict(self.active_params)
        }
