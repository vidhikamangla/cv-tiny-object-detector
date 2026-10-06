"""COCO mAP with VisDrone-appropriate size bins.

The standard COCO bins (small < 32^2) are useless here: on VisDrone almost
*everything* is "small", so the default AP_small hides all the signal. We add a
`tiny` bin at < 16^2 and split small as 16^2..32^2, which is where the ablation
differences actually show up.
"""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..data.visdrone import CLASSES

AREA_RANGES = {
    "all":   [0, 1e10],
    "tiny":  [0, 16 ** 2],
    "small": [16 ** 2, 32 ** 2],
    "med":   [32 ** 2, 96 ** 2],
    "large": [96 ** 2, 1e10],
}


def build_gt(dataset, path):
    """MOT-free COCO GT json from our cached dataset (letterbox coords)."""
    images, annotations = [], []
    ann_id = 1
    for k in range(len(dataset)):
        i = dataset.idx[k]
        _, b, l, _ = dataset.raw(i)
        images.append({"id": int(i), "file_name": str(dataset.stems[i]),
                       "width": dataset.size, "height": dataset.size})
        for box, lab in zip(b, l):
            w, h = float(box[2] - box[0]), float(box[3] - box[1])
            if w <= 0 or h <= 0:
                continue
            annotations.append({"id": ann_id, "image_id": int(i),
                                "category_id": int(lab) + 1,
                                "bbox": [float(box[0]), float(box[1]), w, h],
                                "area": w * h, "iscrowd": 0})
            ann_id += 1
    js = {"images": images, "annotations": annotations,
          "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
    Path(path).write_text(json.dumps(js))
    return path


def evaluate_coco(gt_json, dets, out_json=None):
    """dets: list of {image_id, bbox xywh, score, category_id}. -> metrics dict"""
    if out_json:
        Path(out_json).write_text(json.dumps(dets))
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(gt_json)
        if not dets:
            return {k: 0.0 for k in
                    ["AP", "AP50", "AP75", "AP_tiny", "AP_small", "AP_med", "AR100"]}
        dt = coco.loadRes(dets)
        metrics = {}
        for name, rng in AREA_RANGES.items():
            e = COCOeval(coco, dt, "bbox")
            e.params.areaRng = [rng]
            e.params.areaRngLbl = [name]
            e.params.maxDets = [100]
            # NB: no e.summarize() - it hard-codes maxDets[2] and only prints;
            # everything we need is already in e.eval after accumulate().
            e.evaluate(); e.accumulate()
            prec = e.eval["precision"]
            rec = e.eval["recall"]
            ap = float(np.mean(prec[prec > -1])) if (prec > -1).any() else 0.0
            ar = float(np.mean(rec[rec > -1])) if (rec > -1).any() else 0.0
            if name == "all":
                metrics["AP"] = ap
                metrics["AR100"] = ar
                p50 = prec[0]; p75 = prec[5]
                metrics["AP50"] = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else 0.0
                metrics["AP75"] = float(np.mean(p75[p75 > -1])) if (p75 > -1).any() else 0.0
            else:
                metrics[f"AP_{name}"] = ap
    return metrics
