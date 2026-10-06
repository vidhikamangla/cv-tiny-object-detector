"""Grid outputs -> per-image detections. Kept out of the model so the exported ONNX
graph stays NMS-free and the deployment project can decode however it likes."""
import torch
from torchvision.ops import batched_nms

from .detector import dist2box


@torch.no_grad()
def decode(out, conf=0.01, iou=0.65, max_det=500, pre_nms=3000):
    """-> list of dicts with boxes [n,4] xyxy, scores [n], labels [n]"""
    cls, dist, points = out["cls"], out["dist"], out["points"]
    B, N, nc = cls.shape
    scores_all = cls.sigmoid()
    boxes_all = dist2box(points[None].expand(B, -1, -1), dist)
    results = []
    for b in range(B):
        s = scores_all[b]                                  # [N,nc]
        # flatten (point, class) pairs; one point may survive as two classes
        flat = s.reshape(-1)
        keep = flat > conf
        if keep.sum() > pre_nms:
            thresh = flat.topk(pre_nms).values[-1]
            keep = flat >= thresh
        idx = keep.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            results.append({"boxes": boxes_all.new_zeros((0, 4)),
                            "scores": boxes_all.new_zeros((0,)),
                            "labels": torch.zeros(0, dtype=torch.long, device=cls.device)})
            continue
        pidx = idx // nc
        cidx = idx % nc
        bx = boxes_all[b][pidx]
        sc = flat[idx]
        k = batched_nms(bx, sc, cidx, iou)[:max_det]
        results.append({"boxes": bx[k], "scores": sc[k], "labels": cidx[k]})
    return results
