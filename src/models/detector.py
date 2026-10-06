"""Backbone (timm, ImageNet-pretrained) + FPN/PAN + shared anchor-free head."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import Head
from .neck import FPNPAN


def make_points(shapes, strides, device, dtype=torch.float32):
    """Anchor points at pixel centres. -> points [N,2], stride_per_point [N]"""
    pts, st = [], []
    for (h, w), s in zip(shapes, strides):
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype), indexing="ij")
        p = torch.stack(((xs + 0.5) * s, (ys + 0.5) * s), -1).reshape(-1, 2)
        pts.append(p)
        st.append(torch.full((h * w,), float(s), device=device, dtype=dtype))
    return torch.cat(pts), torch.cat(st)


class TinyDet(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=10, out_ch=128,
                 p2=False, pretrained=True, head_stacks=2):
        super().__init__()
        out_indices = (1, 2, 3, 4) if p2 else (2, 3, 4)
        self.backbone = timm.create_model(
            backbone, features_only=True, out_indices=out_indices, pretrained=pretrained)
        chans = self.backbone.feature_info.channels()
        self.strides = tuple(self.backbone.feature_info.reduction())
        self.neck = FPNPAN(chans, out_ch)
        self.head = Head(out_ch, num_classes, len(chans), stacks=head_stacks)
        self.num_classes = num_classes
        self.export = False

    def forward(self, x):
        """x: float tensor [B,3,H,W] normalised. Returns raw per-level maps when
        self.export, else flattened cls logits / ltrb distances + anchor geometry."""
        feats = self.neck(self.backbone(x))
        cls_out, reg_out = self.head(feats)
        if self.export:
            return cls_out, reg_out
        shapes = [f.shape[-2:] for f in feats]
        B = x.shape[0]
        cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes) for c in cls_out], 1)
        reg = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4) for r in reg_out], 1)
        points, stride_pp = make_points(shapes, self.strides, x.device, torch.float32)
        # ltrb distances, in pixels, always positive
        dist = F.softplus(reg) * stride_pp[None, :, None]
        return {"cls": cls, "dist": dist, "points": points, "stride": stride_pp}


def dist2box(points, dist):
    """points [N,2], dist [...,N,4] (l,t,r,b) -> xyxy boxes."""
    x, y = points[..., 0], points[..., 1]
    return torch.stack([x - dist[..., 0], y - dist[..., 1],
                        x + dist[..., 2], y + dist[..., 3]], -1)


def box2dist(points, boxes):
    """boxes [...,4] xyxy at matching points -> (l,t,r,b)."""
    x, y = points[..., 0], points[..., 1]
    return torch.stack([x - boxes[..., 0], y - boxes[..., 1],
                        boxes[..., 2] - x, boxes[..., 3] - y], -1)
