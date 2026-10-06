"""Evaluate a checkpoint (or EMA weights) on the cached val split."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import config as C
from ..data.dataset import VisDroneCache, collate
from ..data.visdrone import NUM_CLASSES
from ..models.decode import decode
from ..models.detector import TinyDet
from ..train import IMAGENET_MEAN, IMAGENET_STD, build_model, normalise
from .coco_eval import build_gt, evaluate_coco


def to_coco_dets(results, image_ids):
    dets = []
    for res, iid in zip(results, image_ids):
        b = res["boxes"].cpu().numpy()
        s = res["scores"].cpu().numpy()
        l = res["labels"].cpu().numpy()
        for (x0, y0, x1, y1), sc, lab in zip(b, s, l):
            dets.append({"image_id": int(iid), "category_id": int(lab) + 1,
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                         "score": float(sc)})
    return dets


@torch.no_grad()
def run(cfg, model, device, ds, conf=None, iou=None, max_det=None, timing=True):
    model.eval()
    ec = cfg.eval
    conf = ec.conf if conf is None else conf
    iou = ec.iou if iou is None else iou
    max_det = ec.max_det if max_det is None else max_det
    dl = DataLoader(ds, batch_size=int(ec.batch), shuffle=False, num_workers=8,
                    collate_fn=collate, pin_memory=True)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    dets, n_img, t_infer = [], 0, 0.0
    for batch in dl:
        img = batch["img"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(normalise(img, mean, std))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_infer += time.perf_counter() - t0
        res = decode(out, conf=float(conf), iou=float(iou), max_det=int(max_det))
        dets += to_coco_dets(res, batch["index"].tolist())
        n_img += img.shape[0]
    ms = 1000.0 * t_infer / max(1, n_img)
    return dets, ms


def main():
    ap = argparse.ArgumentParser(description="evaluate TinyDet on VisDrone val")
    C.add_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--out", default=None, help="metrics json path")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cfg = C.load(a.config, a.sets)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    subset = cfg.data.get("subset")
    ds = VisDroneCache(cfg.paths.val_cache, aug={}, train=False, subset=subset)
    gt_path = Path(cfg.paths.val_cache) / (f"coco_gt_{subset}.json" if subset else "coco_gt.json")
    if not gt_path.exists():
        build_gt(ds, gt_path)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg, device)
    sd = ck.get("ema") if a.weights == "ema" and ck.get("ema") else ck["model"]
    missing = model.load_state_dict(sd, strict=False)
    if missing.missing_keys:
        print("warning: missing keys", missing.missing_keys[:5])

    dets, ms = run(cfg, model, device, ds)
    metrics = evaluate_coco(str(gt_path), dets)
    metrics["ms_per_img"] = round(ms, 2)
    metrics["n_dets"] = len(dets)
    metrics["weights"] = a.weights
    metrics["tag"] = a.tag or cfg.run_name
    params = sum(p.numel() for p in model.parameters()) / 1e6
    metrics["params_M"] = round(params, 2)
    done = Path(a.ckpt).parent / "done.json"
    if done.exists():
        d = json.loads(done.read_text())
        metrics["epochs_completed"] = d.get("epochs_completed")
        metrics["train_minutes"] = round(d.get("train_seconds", 0) / 60.0, 1)
    out = a.out or str(Path(a.ckpt).parent / f"metrics_{a.weights}.json")
    Path(out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
