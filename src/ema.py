"""Exponential moving average of weights.

The decay ramps in (`1 - exp(-step/tau)`) so the shadow copy is not dominated by the
random initialisation for the first few hundred steps - a plain fixed 0.9998 makes the
EMA model useless on a short schedule, which is exactly the regime this project runs in.
"""
import math
from copy import deepcopy

import torch


class ModelEMA:
    def __init__(self, model, decay=0.9998, tau=2000, updates=0):
        self.ema = deepcopy(self._unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = updates
        self.fn = lambda x: decay * (1 - math.exp(-x / tau))

    @staticmethod
    def _unwrap(m):
        return m.module if hasattr(m, "module") else m

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.fn(self.updates)
        msd = self._unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach().to(v.dtype)
            else:
                v.copy_(msd[k])
