"""Tiled inference: run the detector on overlapping crops, then merge.

The point: a 6-pixel car in a 1400x800 drone frame is sub-pixel by the time the image
is letterboxed to 640. Slicing the *original* resolution into 640 tiles keeps those
objects at their native size, at the cost of N forward passes per image.

Merging is greedy non-maximum *merging* rather than plain NMS across tiles: a vehicle
cut in half by a tile boundary produces two partial boxes that NMS would keep (low
mutual IoU), so we union boxes whose intersection covers most of the smaller one.
"""
import numpy as np
import torch
from torchvision.ops import batched_nms

from ..data.cache import letterbox
from ..models.decode import decode


def tile_origins(h, w, tile, overlap):
    step = max(1, int(round(tile * (1 - overlap))))
    xs = list(range(0, max(1, w - tile + 1), step))
    ys = list(range(0, max(1, h - tile + 1), step))
    if not xs or xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    if not ys or ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    return sorted(set(xs)), sorted(set(ys))


def _containment(a, b):
    """Fraction of the *smaller* box covered by the intersection. [N,M]"""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    ar = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    br = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(np.minimum(ar[:, None], br[None, :]), 1e-9)


def greedy_nmm(boxes, scores, labels, contain_thresh=0.7, iou_thresh=0.55):
    """Union overlapping boxes of the same class instead of discarding them."""
    if len(boxes) == 0:
        return boxes, scores, labels
    keep_b, keep_s, keep_l = [], [], []
    for cls in np.unique(labels):
        m = labels == cls
        b, s = boxes[m], scores[m]
        order = np.argsort(-s)
        b, s = b[order], s[order]
        used = np.zeros(len(b), bool)
        for i in range(len(b)):
            if used[i]:
                continue
            group = [i]
            c = _containment(b[i:i + 1], b)[0]
            lt = np.maximum(b[i, :2], b[:, :2]); rb = np.minimum(b[i, 2:], b[:, 2:])
            wh = np.clip(rb - lt, 0, None)
            inter = wh[:, 0] * wh[:, 1]
            ai = max((b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]), 1e-9)
            aj = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
            iou = inter / np.maximum(ai + aj - inter, 1e-9)
            for j in range(i + 1, len(b)):
                if not used[j] and (c[j] > contain_thresh or iou[j] > iou_thresh):
                    used[j] = True
                    group.append(j)
            g = b[group]
            keep_b.append([g[:, 0].min(), g[:, 1].min(), g[:, 2].max(), g[:, 3].max()])
            keep_s.append(float(s[group].max()))
            keep_l.append(int(cls))
            used[i] = True
    return (np.asarray(keep_b, np.float32).reshape(-1, 4),
            np.asarray(keep_s, np.float32), np.asarray(keep_l, np.int64))


@torch.no_grad()
def detect_tiled(model, image_bgr, device, mean, std, tile=640, overlap=0.2,
                 conf=0.01, iou=0.65, max_det=1000, include_full=True, batch=8):
    """image_bgr at ORIGINAL resolution -> (boxes xyxy, scores, labels) in that frame."""
    h, w = image_bgr.shape[:2]
    crops, offsets = [], []
    xs, ys = tile_origins(h, w, tile, overlap)
    for y in ys:
        for x in xs:
            c = image_bgr[y:y + tile, x:x + tile]
            if c.shape[0] != tile or c.shape[1] != tile:
                padded = np.full((tile, tile, 3), 114, np.uint8)
                padded[:c.shape[0], :c.shape[1]] = c
                c = padded
            crops.append(c); offsets.append((x, y, 1.0, 0, 0))
    if include_full:
        lb, r, pad = letterbox(image_bgr, tile)
        crops.append(lb); offsets.append((0, 0, r, pad[0], pad[1]))

    allb, alls, alll = [], [], []
    for start in range(0, len(crops), batch):
        chunk = crops[start:start + batch]
        t = torch.from_numpy(np.stack([c[:, :, ::-1].transpose(2, 0, 1) for c in chunk]))
        t = t.to(device)
        x = (t.float() / 255.0 - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
        res = decode(out, conf=conf, iou=iou, max_det=max_det)
        for k, r in enumerate(res):
            ox, oy, ratio, px, py = offsets[start + k]
            b = r["boxes"].float().cpu().numpy()
            if len(b) == 0:
                continue
            if ratio != 1.0:                      # the whole-image letterboxed pass
                b[:, [0, 2]] = (b[:, [0, 2]] - px) / ratio
                b[:, [1, 3]] = (b[:, [1, 3]] - py) / ratio
            else:
                b[:, [0, 2]] += ox
                b[:, [1, 3]] += oy
            allb.append(b)
            alls.append(r["scores"].float().cpu().numpy())
            alll.append(r["labels"].cpu().numpy())

    if not allb:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    boxes = np.concatenate(allb); scores = np.concatenate(alls); labels = np.concatenate(alll)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return greedy_nmm(boxes, scores, labels)
