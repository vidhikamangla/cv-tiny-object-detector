"""FPN top-down + PAN bottom-up. Both directions matter for small objects:
top-down injects semantics into the stride-8 map, bottom-up puts the fine detail
back into the deeper maps so the classifier is not guessing from context alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class FPNPAN(nn.Module):
    def __init__(self, in_channels, out_ch=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.td_smooth = nn.ModuleList([conv_bn(out_ch, out_ch) for _ in in_channels[:-1]])
        self.down = nn.ModuleList([conv_bn(out_ch, out_ch, 3, 2) for _ in in_channels[:-1]])
        self.bu_smooth = nn.ModuleList([conv_bn(out_ch, out_ch) for _ in in_channels[:-1]])
        self.out_ch = out_ch

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        # top-down
        td = [None] * len(lat)
        td[-1] = lat[-1]
        for i in range(len(lat) - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=lat[i].shape[-2:], mode="nearest")
            td[i] = self.td_smooth[i](lat[i] + up)
        # bottom-up
        out = [td[0]]
        for i in range(len(td) - 1):
            d = self.down[i](out[-1])
            out.append(self.bu_smooth[i](td[i + 1] + d))
        return out
