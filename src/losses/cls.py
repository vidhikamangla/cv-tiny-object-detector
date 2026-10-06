"""Classification losses. All take soft targets in [0,1] so TAL's quality-aware
targets and plain one-hot both flow through the same code path."""
import torch
import torch.nn.functional as F


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none")


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        a_t = alpha * target + (1 - alpha) * (1 - target)
        loss = a_t * loss
    return loss


def varifocal_loss(logits, target, alpha=0.75, gamma=2.0):
    """Asymmetric: positives are weighted by their IoU-quality target, negatives by
    p^gamma. Unlike focal loss this does *not* down-weight high-quality positives,
    which is the point - a well-localised tiny box should keep contributing."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.where(target > 0, target, alpha * p.pow(gamma))
    return ce * weight


def get_cls_loss(name):
    return {"bce": bce_loss, "focal": focal_loss, "varifocal": varifocal_loss}[name]
