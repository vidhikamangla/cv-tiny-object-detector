"""Evaluate SAHI-style tiled inference properly, i.e. at the ORIGINAL resolution.

The subtlety that makes this worth a separate file: the cached dataset is letterboxed to
640, so tiling *that* buys nothing - the small objects are already destroyed. Tiling only
helps if the tiles are cut from the original image, where a 6-pixel object is still
6 pixels rather than 2.

So this reads the original JPEGs, tiles them, and then maps the detections back into the
cached letterbox coordinate frame (the stored per-image ratio and padding) so they can be
scored against exactly the same ground truth as every other row in the ablation table.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import config as C
from ..data.dataset import VisDroneCache
from ..data.visdrone import list_split
from ..eval.coco_eval import build_gt, evaluate_coco
from ..eval.sahi import detect_tiled
from ..infer import detect_plain, load
from ..train import IMAGENET_MEAN, IMAGENET_STD


def main():
    ap = argparse.ArgumentParser(description="tiled vs single-pass evaluation at native resolution")
    C.add_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--split-root", required=True, help="VisDrone2019-DET-val directory")
    ap.add_argument("--mode", default="tiled", choices=["tiled", "single"])
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = C.load(a.config, a.sets)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    model = load(cfg, a.ckpt, a.weights, device)

    # the cache holds the letterbox ratio/pad per image, which is how we get back into
    # the coordinate frame the ground truth lives in
    ds = VisDroneCache(cfg.paths.val_cache, aug={}, train=False)
    gt_path = Path(cfg.paths.val_cache) / "coco_gt.json"
    if not gt_path.exists():
        build_gt(ds, gt_path)
    stem_to_idx = {str(s): i for i, s in enumerate(ds.stems)}

    pairs = list_split(Path(a.split_root))
    if a.limit:
        pairs = pairs[:a.limit]

    dets, t_total, n = [], 0.0, 0
    for imp, _ in pairs:
        idx = stem_to_idx.get(imp.stem)
        if idx is None:
            continue
        im = cv2.imread(str(imp))
        if im is None:
            continue
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        if a.mode == "tiled":
            b, s, l = detect_tiled(model, im, device, mean, std, tile=a.tile,
                                   overlap=a.overlap, conf=float(cfg.eval.conf),
                                   iou=float(cfg.eval.iou), max_det=int(cfg.eval.max_det))
        else:
            b, s, l = detect_plain(model, im, device, mean, std, a.tile,
                                   float(cfg.eval.conf), float(cfg.eval.iou),
                                   int(cfg.eval.max_det))
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_total += time.perf_counter() - t0
        n += 1

        # original pixels -> cached letterbox frame
        r = float(ds.ratio[idx]); px, py = ds.pad[idx]
        if len(b):
            b = b.copy()
            b[:, [0, 2]] = b[:, [0, 2]] * r + px
            b[:, [1, 3]] = b[:, [1, 3]] * r + py
        for (x0, y0, x1, y1), sc, lab in zip(b, s, l):
            dets.append({"image_id": int(idx), "category_id": int(lab) + 1,
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                         "score": float(sc)})

    metrics = evaluate_coco(str(gt_path), dets)
    metrics.update(mode=a.mode, tile=a.tile, overlap=a.overlap,
                   ms_per_img=round(1000 * t_total / max(1, n), 2),
                   n_images=n, n_dets=len(dets),
                   tag=a.tag or f"{Path(a.ckpt).parent.name}_{a.mode}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
