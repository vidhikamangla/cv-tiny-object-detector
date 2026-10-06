"""IoU family, implemented here on purpose: these are the box losses the ablation
switches between, so they are the last thing that should be a black box.

CIoU adds, on top of GIoU, a centre-distance term and an aspect-ratio term. For tiny
boxes the centre term dominates, which is exactly what we want: when two 12x12 boxes
barely overlap, GIoU's gradient is nearly flat while CIoU still pulls the centres together.
"""
import math

import torch


def box_area(b):
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def pairwise_iou(a, b, eps=1e-7):
    """a [N,4], b [M,4] xyxy -> IoU [N,M]"""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / (union + eps)


def bbox_iou(p, t, kind="ciou", eps=1e-7):
    """Elementwise IoU variant between matched boxes. -> value in (-1, 1], shape [...]"""
    lt = torch.max(p[..., :2], t[..., :2])
    rb = torch.min(p[..., 2:], t[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    ap, at = box_area(p), box_area(t)
    union = ap + at - inter
    iou = inter / (union + eps)
    if kind == "iou":
        return iou

    # smallest enclosing box
    elt = torch.min(p[..., :2], t[..., :2])
    erb = torch.max(p[..., 2:], t[..., 2:])
    ewh = (erb - elt).clamp(min=0)

    if kind == "giou":
        enc = ewh[..., 0] * ewh[..., 1]
        return iou - (enc - union) / (enc + eps)

    if kind in ("diou", "ciou"):
        c2 = ewh[..., 0] ** 2 + ewh[..., 1] ** 2 + eps
        pcx = (p[..., 0] + p[..., 2]) / 2; pcy = (p[..., 1] + p[..., 3]) / 2
        tcx = (t[..., 0] + t[..., 2]) / 2; tcy = (t[..., 1] + t[..., 3]) / 2
        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        diou = iou - rho2 / c2
        if kind == "diou":
            return diou
        pw = (p[..., 2] - p[..., 0]).clamp(min=eps); ph = (p[..., 3] - p[..., 1]).clamp(min=eps)
        tw = (t[..., 2] - t[..., 0]).clamp(min=eps); th = (t[..., 3] - t[..., 1]).clamp(min=eps)
        v = (4 / math.pi ** 2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return diou - alpha * v

    raise ValueError(kind)


def iou_loss(p, t, kind="ciou"):
    return 1.0 - bbox_iou(p, t, kind)
