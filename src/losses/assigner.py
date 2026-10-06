"""Label assignment - the component that actually decides whether tiny objects learn.

Two assigners, because switching between them is the headline ablation:

  center_prior  (FCOS-style)  : geometry only. A point is a candidate if it falls
                                inside the GT box and within `radius` strides of its
                                centre; the top-k closest candidates become positives.
                                Static: the same point is positive from epoch 0.

  tal           (TOOD-style)  : prediction-aware. Score each in-box point by
                                s^alpha * iou^beta (s = predicted class score, iou =
                                predicted-vs-GT IoU), take top-k per GT, and use the
                                normalised metric as a *soft* classification target.
                                This aligns classification confidence with localisation
                                quality, which is why it helps small objects: a 12x12 box
                                has ~1 well-placed point, and TAL finds the one the
                                network can actually regress from instead of the one
                                nearest the centre.

Both return the same tuple, so the criterion does not care which ran.
"""
import torch

from .iou import bbox_iou, pairwise_iou


def _empty(n_points, nc, device, dtype):
    return (torch.zeros(n_points, dtype=torch.bool, device=device),
            torch.zeros(n_points, dtype=torch.long, device=device),
            torch.zeros(n_points, 4, device=device, dtype=dtype),
            torch.zeros(n_points, nc, device=device, dtype=dtype))


def _resolve_conflicts(cand, metric):
    """cand [G,N] bool, metric [G,N] - a point claimed by several GTs goes to the best."""
    counts = cand.sum(0)
    multi = counts > 1
    if multi.any():
        masked = metric.masked_fill(~cand, -1.0)
        best = masked.argmax(0)                       # [N]
        keep = torch.zeros_like(cand)
        keep[best, torch.arange(cand.shape[1], device=cand.device)] = True
        cand = torch.where(multi[None, :], cand & keep, cand)
    return cand


def center_prior_assign(points, stride_pp, gt_boxes, gt_labels, num_classes,
                        topk=9, radius=2.5):
    N = points.shape[0]
    dtype = points.dtype
    G = gt_boxes.shape[0]
    if G == 0:
        return _empty(N, num_classes, points.device, dtype)

    px, py = points[:, 0][None, :], points[:, 1][None, :]
    gl, gt_, gr, gb = (gt_boxes[:, i][:, None] for i in range(4))
    inside_box = (px > gl) & (px < gr) & (py > gt_) & (py < gb)

    gcx, gcy = (gl + gr) / 2, (gt_ + gb) / 2
    r = radius * stride_pp[None, :]
    inside_center = (px - gcx).abs() < r
    inside_center &= (py - gcy).abs() < r
    cand = inside_box & inside_center
    if not cand.any():
        cand = inside_box
        if not cand.any():
            return _empty(N, num_classes, points.device, dtype)

    # closeness in stride units, so a stride-32 point is not punished for being far in pixels
    dist = ((px - gcx) ** 2 + (py - gcy) ** 2).sqrt() / stride_pp[None, :]
    metric = -dist
    metric = metric.masked_fill(~cand, -1e9)
    k = min(topk, N)
    _, idx = metric.topk(k, dim=1)
    sel = torch.zeros_like(cand)
    sel.scatter_(1, idx, True)
    cand = cand & sel

    # smaller GT wins a contested point: it has fewer candidates to spare
    areas = ((gr - gl) * (gb - gt_)).expand(-1, N)
    cand = _resolve_conflicts(cand, -areas)

    fg = cand.any(0)
    gi = cand.float().argmax(0)
    tgt_boxes = gt_boxes[gi]
    tgt_labels = gt_labels[gi]
    tgt_scores = torch.zeros(N, num_classes, device=points.device, dtype=dtype)
    tgt_scores[fg, tgt_labels[fg]] = 1.0
    return fg, gi, tgt_boxes, tgt_scores


def tal_assign(points, stride_pp, gt_boxes, gt_labels, pred_scores, pred_boxes,
               num_classes, topk=10, alpha=0.5, beta=6.0, eps=1e-9):
    N = points.shape[0]
    dtype = points.dtype
    G = gt_boxes.shape[0]
    if G == 0:
        return _empty(N, num_classes, points.device, dtype)

    px, py = points[:, 0][None, :], points[:, 1][None, :]
    gl, gt_, gr, gb = (gt_boxes[:, i][:, None] for i in range(4))
    inside_box = (px > gl) & (px < gr) & (py > gt_) & (py < gb)
    if not inside_box.any():
        return _empty(N, num_classes, points.device, dtype)

    with torch.no_grad():
        ious = pairwise_iou(gt_boxes, pred_boxes).clamp(min=0)          # [G,N]
        s = pred_scores.detach().sigmoid()[:, gt_labels].t()            # [G,N]
        metric = s.pow(alpha) * ious.pow(beta)
        metric = metric * inside_box

        k = min(topk, N)
        topv, topi = metric.topk(k, dim=1)
        # a GT with all-zero metric gets no positives from topk; fall back to best IoU
        valid_k = topv > eps
        sel = torch.zeros_like(inside_box)
        sel.scatter_(1, topi, valid_k)
        cand = sel & inside_box
        # A GT that won no candidate contributes *no gradient at all*, so it must be
        # rescued: take its best in-box point by IoU, breaking ties on centre distance
        # so the choice is still well defined when every prediction is garbage (which
        # is exactly the situation at step 0).
        dead = ~cand.any(1)
        if dead.any():
            gcx, gcy = (gl + gr) / 2, (gt_ + gb) / 2
            cdist = ((px - gcx) ** 2 + (py - gcy) ** 2).sqrt() / stride_pp[None, :]
            fb = (ious - 1e-3 * cdist).masked_fill(~inside_box, torch.finfo(dtype).min)
            fallback = fb.argmax(1)
            rescuable = dead & inside_box.any(1)
            idx = rescuable.nonzero(as_tuple=True)[0]
            cand[idx, fallback[idx]] = True

        cand = _resolve_conflicts(cand, ious)

        fg = cand.any(0)
        gi = cand.float().argmax(0)
        # normalise the metric per GT so each object contributes comparable weight,
        # then scale by that GT's best IoU -> soft target carries localisation quality
        pos_metric = metric * cand
        norm = pos_metric.amax(dim=1, keepdim=True).clamp(min=eps)
        iou_max = (ious * cand).amax(dim=1, keepdim=True)
        soft = (pos_metric / norm) * iou_max                            # [G,N]
        score_per_point = soft.sum(0)

    tgt_boxes = gt_boxes[gi]
    tgt_labels = gt_labels[gi]
    tgt_scores = torch.zeros(N, num_classes, device=points.device, dtype=dtype)
    tgt_scores[fg, tgt_labels[fg]] = score_per_point[fg].to(dtype)
    return fg, gi, tgt_boxes, tgt_scores
