"""Ties assignment + classification + box loss, including VisDrone ignore regions.

Ignore regions are the subtle part. VisDrone marks crowds and unlabelled areas with
category 0. A detection there is neither right nor wrong, so those points are dropped
from the classification loss instead of being trained as background - otherwise the
model is actively taught to suppress real objects and AP_tiny suffers.
"""
import torch
import torch.nn as nn

from ..models.detector import dist2box
from .assigner import center_prior_assign, tal_assign
from .cls import get_cls_loss
from .iou import iou_loss


class Criterion(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        lc = cfg.loss
        self.num_classes = num_classes
        self.assigner = lc.assigner                      # center_prior | tal
        self.cls_name = lc.cls                            # bce | focal | varifocal
        self.box_kind = lc.box                            # iou | giou | diou | ciou
        self.w_cls = float(lc.w_cls)
        self.w_box = float(lc.w_box)
        self.topk = int(lc.topk)
        self.radius = float(lc.get("radius", 2.5))
        self.alpha = float(lc.get("tal_alpha", 0.5))
        self.beta = float(lc.get("tal_beta", 6.0))
        self.cls_loss = get_cls_loss(self.cls_name)

    def forward(self, out, targets):
        cls, dist, points, stride_pp = out["cls"], out["dist"], out["points"], out["stride"]
        B, N, nc = cls.shape
        boxes_pred = dist2box(points[None, :, :].expand(B, -1, -1), dist)

        total_cls = cls.new_zeros(())
        total_box = cls.new_zeros(())
        total_pos = cls.new_zeros(())
        total_score = cls.new_zeros(())

        for b in range(B):
            gtb = targets["boxes"][b].to(cls.device, cls.dtype)
            gtl = targets["labels"][b].to(cls.device)
            ign = targets["ignore"][b].to(cls.device, cls.dtype)

            if self.assigner == "tal":
                fg, gi, tboxes, tscores = tal_assign(
                    points, stride_pp, gtb, gtl, cls[b], boxes_pred[b].detach(),
                    nc, topk=self.topk, alpha=self.alpha, beta=self.beta)
            else:
                fg, gi, tboxes, tscores = center_prior_assign(
                    points, stride_pp, gtb, gtl, nc,
                    topk=self.topk, radius=self.radius)

            keep = torch.ones(N, dtype=torch.bool, device=cls.device)
            if len(ign):
                px, py = points[:, 0][None, :], points[:, 1][None, :]
                inside_ign = ((px > ign[:, 0][:, None]) & (px < ign[:, 2][:, None]) &
                              (py > ign[:, 1][:, None]) & (py < ign[:, 3][:, None])).any(0)
                keep = ~inside_ign | fg          # never drop a point that is a positive

            l_cls = self.cls_loss(cls[b], tscores)
            total_cls = total_cls + l_cls[keep].sum()

            if fg.any():
                l_box = iou_loss(boxes_pred[b][fg], tboxes[fg], self.box_kind)
                # weight each positive by its soft target mass: TAL's quality signal
                w = tscores[fg].sum(-1).clamp(min=1e-6) if self.assigner == "tal" else None
                total_box = total_box + ((l_box * w).sum() if w is not None else l_box.sum())
                total_pos = total_pos + fg.sum()
                total_score = total_score + (w.sum() if w is not None else fg.sum())

        norm_cls = total_score.clamp(min=1.0) if self.assigner == "tal" else total_pos.clamp(min=1.0)
        norm_box = total_score.clamp(min=1e-6) if self.assigner == "tal" else total_pos.clamp(min=1.0)
        loss_cls = self.w_cls * total_cls / norm_cls
        loss_box = self.w_box * total_box / norm_box
        return loss_cls + loss_box, {
            "cls": loss_cls.detach(),
            "box": loss_box.detach(),
            "npos": total_pos.detach(),
        }
