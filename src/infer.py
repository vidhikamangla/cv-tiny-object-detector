"""Run TinyDet on an image or video, with or without tiling."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config as C
from .data.cache import letterbox
from .data.visdrone import CLASSES
from .eval.sahi import detect_tiled
from .models.decode import decode
from .train import IMAGENET_MEAN, IMAGENET_STD, build_model

PALETTE = [(66, 135, 245), (245, 130, 48), (60, 180, 75), (240, 50, 230),
           (70, 240, 240), (210, 245, 60), (250, 190, 190), (170, 110, 40),
           (255, 225, 25), (0, 128, 128)]


def load(cfg, ckpt, weights, device):
    model = build_model(cfg, device).eval()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("ema") if weights == "ema" and ck.get("ema") else ck["model"]
    model.load_state_dict(sd, strict=False)
    return model


@torch.no_grad()
def detect_plain(model, img, device, mean, std, imgsz, conf, iou, max_det):
    lb, r, pad = letterbox(img, imgsz)
    # ascontiguousarray: the ::-1 BGR->RGB flip leaves a negative stride, which
    # torch.from_numpy refuses
    t = torch.from_numpy(np.ascontiguousarray(
        lb[:, :, ::-1].transpose(2, 0, 1))).unsqueeze(0).to(device)
    x = (t.float() / 255.0 - mean) / std
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(x)
    res = decode(out, conf=conf, iou=iou, max_det=max_det)[0]
    b = res["boxes"].float().cpu().numpy()
    if len(b):
        b[:, [0, 2]] = (b[:, [0, 2]] - pad[0]) / r
        b[:, [1, 3]] = (b[:, [1, 3]] - pad[1]) / r
    return b, res["scores"].float().cpu().numpy(), res["labels"].cpu().numpy()


def draw(img, boxes, scores, labels, thickness=2, show_labels=True):
    out = img.copy()
    for (x0, y0, x1, y1), s, l in zip(boxes, scores, labels):
        c = PALETTE[int(l) % len(PALETTE)]
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), c, thickness)
        if show_labels:
            cv2.putText(out, f"{CLASSES[int(l)]} {s:.2f}", (int(x0), max(10, int(y0) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="run TinyDet on an image or video")
    C.add_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiles", action="store_true", help="SAHI-style tiled inference")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=600)
    a = ap.parse_args()
    cfg = C.load(a.config, a.sets)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    model = load(cfg, a.ckpt, a.weights, device)

    def run(img):
        if a.tiles:
            return detect_tiled(model, img, device, mean, std, tile=a.imgsz,
                                conf=a.conf, iou=float(cfg.eval.iou))
        return detect_plain(model, img, device, mean, std, a.imgsz, a.conf,
                            float(cfg.eval.iou), int(cfg.eval.max_det))

    src = Path(a.source)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(str(src))
        if img is None:
            raise SystemExit(f"could not read {src}")
        b, s, l = run(img)
        cv2.imwrite(a.out, draw(img, b, s, l))
        print(f"{len(b)} detections -> {a.out}")
        return

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer, n = None, 0
    while n < a.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        b, s, l = run(frame)
        vis = draw(frame, b, s, l)
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        n += 1
    cap.release()
    if writer:
        writer.release()
    print(f"{n} frames -> {a.out}")


if __name__ == "__main__":
    main()
