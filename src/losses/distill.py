"""Knowledge distillation from a larger teacher detector into the student.

This is cheap and clean here because teacher and student share the head geometry: same
input size, same strides, so the two models emit predictions on *exactly the same anchor
points*. Distillation is then a per-point comparison with no matching step at all.

Two terms:

  cls  - BCE of the student's logits against the teacher's sigmoid scores. The teacher's
         soft scores carry which points are ambiguous, which a one-hot target cannot.
  box  - IoU loss between student and teacher boxes, applied only where the teacher is
         actually confident; copying the teacher's box on a point it scores at 0.01 is
         copying noise.
"""
import torch
import torch.nn.functional as F

from ..models.detector import dist2box
from .iou import iou_loss


def distill_loss(student_out, teacher_out, w_cls=1.0, w_box=1.0,
                 conf_thresh=0.1, box_kind="ciou", temperature=1.0):
    """-> (loss, parts dict). Both outs are the dicts returned by TinyDet.forward."""
    s_cls, s_dist = student_out["cls"], student_out["dist"]
    t_cls, t_dist = teacher_out["cls"].detach(), teacher_out["dist"].detach()
    if s_cls.shape != t_cls.shape:
        raise ValueError(f"teacher/student grids differ: {t_cls.shape} vs {s_cls.shape} - "
                         "distillation here assumes identical strides and input size")

    t_prob = torch.sigmoid(t_cls / temperature)
    l_cls = F.binary_cross_entropy_with_logits(s_cls, t_prob, reduction="none").mean()

    with torch.no_grad():
        conf = t_prob.max(-1).values                      # [B, N]
        mask = conf > conf_thresh
        n_pos = int(mask.sum())
    if n_pos:
        points = student_out["points"]
        B = s_dist.shape[0]
        pts = points[None, :, :].expand(B, -1, -1)
        s_box = dist2box(pts, s_dist)[mask]
        t_box = dist2box(pts, t_dist)[mask]
        w = conf[mask]
        l_box = (iou_loss(s_box, t_box, box_kind) * w).sum() / w.sum().clamp(min=1e-6)
    else:
        l_box = s_cls.new_zeros(())

    loss = w_cls * l_cls + w_box * l_box
    return loss, {"kd_cls": l_cls.detach(), "kd_box": l_box.detach(),
                  "kd_points": torch.as_tensor(float(n_pos))}
