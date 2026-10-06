"""Decoupled anchor-free head, weights shared across levels.

Sharing the head across levels (with a per-level learnable scale on the box branch)
is what lets one set of filters serve stride 8/16/32: the regression target is in
*stride units*, so the same filter sees the same apparent object size at every level.
"""
import math

import torch
import torch.nn as nn

from .neck import conv_bn


class Head(nn.Module):
    def __init__(self, ch, num_classes, num_levels, stacks=2, prior_prob=0.01):
        super().__init__()
        self.nc = num_classes
        self.cls_stem = nn.Sequential(*[conv_bn(ch, ch) for _ in range(stacks)])
        self.reg_stem = nn.Sequential(*[conv_bn(ch, ch) for _ in range(stacks)])
        self.cls_pred = nn.Conv2d(ch, num_classes, 3, 1, 1)
        self.reg_pred = nn.Conv2d(ch, 4, 3, 1, 1)
        # one scalar per level so the shared box filter can adapt its dynamic range
        self.scales = nn.Parameter(torch.ones(num_levels))
        # bias init so initial sigmoid ~= prior_prob; without this the focal/BCE loss
        # starts with ~8k predicted positives per image and the first epochs are wasted
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - prior_prob) / prior_prob))
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)

    def forward(self, feats):
        cls_out, reg_out = [], []
        for i, f in enumerate(feats):
            c = self.cls_pred(self.cls_stem(f))
            r = self.reg_pred(self.reg_stem(f)) * self.scales[i]
            cls_out.append(c)
            reg_out.append(r)
        return cls_out, reg_out
