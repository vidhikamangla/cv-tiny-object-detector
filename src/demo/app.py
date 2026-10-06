"""Gradio demo: upload a drone frame, toggle tiled inference, see the difference.

The tiling toggle is the whole point of the demo - it makes the tiny-object problem
visible in one click instead of requiring the reader to trust a table.
"""
import argparse

import numpy as np
import torch

from .. import config as C
from ..data.visdrone import CLASSES
from ..eval.sahi import detect_tiled
from ..infer import detect_plain, draw, load
from ..train import IMAGENET_MEAN, IMAGENET_STD

STATE = {}


def predict(image, use_tiles, conf, imgsz):
    if image is None:
        return None, "upload an image"
    bgr = image[:, :, ::-1].copy()
    model, device, mean, std, cfg = (STATE[k] for k in ("model", "device", "mean", "std", "cfg"))
    import time
    t0 = time.perf_counter()
    if use_tiles:
        b, s, l = detect_tiled(model, bgr, device, mean, std, tile=int(imgsz),
                               conf=float(conf), iou=float(cfg.eval.iou))
    else:
        b, s, l = detect_plain(model, bgr, device, mean, std, int(imgsz), float(conf),
                               float(cfg.eval.iou), int(cfg.eval.max_det))
    ms = 1000 * (time.perf_counter() - t0)
    vis = draw(bgr, b, s, l)
    counts = {}
    for lab in l:
        counts[CLASSES[int(lab)]] = counts.get(CLASSES[int(lab)], 0) + 1
    summary = (f"{len(b)} detections in {ms:.0f} ms "
               f"({'tiled' if use_tiles else 'single pass'})\n" +
               ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    return vis[:, :, ::-1], summary


def main():
    ap = argparse.ArgumentParser(description="TinyDet gradio demo")
    C.add_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    cfg = C.load(a.config, a.sets)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    STATE.update(cfg=cfg, device=device,
                 mean=IMAGENET_MEAN.to(device), std=IMAGENET_STD.to(device),
                 model=load(cfg, a.ckpt, a.weights, device))

    import gradio as gr
    with gr.Blocks(title="TinyDet - small-object detection in drone imagery") as demo:
        gr.Markdown("## TinyDet\nAnchor-free detector trained on VisDrone2019-DET. "
                    "Turn on **tiled inference** to see what happens to the smallest objects.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="drone frame", type="numpy")
                tiles = gr.Checkbox(label="tiled inference (SAHI-style)", value=False)
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="confidence")
                imgsz = gr.Dropdown([640, 960], value=640, label="input / tile size")
                btn = gr.Button("detect", variant="primary")
            with gr.Column():
                out = gr.Image(label="detections")
                txt = gr.Textbox(label="summary", lines=3)
        btn.click(predict, [inp, tiles, conf, imgsz], [out, txt])
    demo.launch(server_name="0.0.0.0", server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
