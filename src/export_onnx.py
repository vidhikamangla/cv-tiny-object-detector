"""Export raw head outputs to ONNX. Decoding and NMS stay OUTSIDE the graph.

Why: the deployment benchmark wants to time the *network*, and a graph containing NMS
has data-dependent output shapes that block static shape inference, INT8 calibration
and fair batching. Keeping decode in Python also means the exported graph is identical
across precisions, so an accuracy delta can only come from the numerics.
"""
import argparse
from pathlib import Path

import torch

from . import config as C
from .data.visdrone import NUM_CLASSES
from .train import build_model


class ExportWrapper(torch.nn.Module):
    """Concatenates per-level maps into two fixed tensors: [B,N,nc] and [B,N,4]."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.export = True
        self.nc = model.num_classes

    def forward(self, x):
        cls_out, reg_out = self.model(x)
        B = x.shape[0]
        cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, self.nc) for c in cls_out], 1)
        reg = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4) for r in reg_out], 1)
        return cls, torch.nn.functional.softplus(reg)


def main():
    ap = argparse.ArgumentParser(description="export TinyDet to ONNX")
    C.add_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic-batch", action="store_true", default=True)
    ap.add_argument("--simplify", action="store_true", default=True)
    ap.add_argument("--half", action="store_true",
                    help="export an FP16 graph (TensorRT >=11 is strongly typed, "
                         "so precision must come from the ONNX itself)")
    a = ap.parse_args()
    cfg = C.load(a.config, a.sets)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # Architecture comes from the CHECKPOINT, not --config: a stride-4 or ResNet-50
    # checkpoint will not load into the default architecture.
    build_cfg = C.Cfg(dict(cfg))
    saved = ck.get("cfg") or {}
    if saved.get("model"):
        build_cfg["model"] = dict(saved["model"])
        if dict(saved["model"]) != dict(cfg.model):
            print(f"using the checkpoint's model config: {dict(saved['model'])}")
    model = build_model(build_cfg, torch.device("cpu")).eval()
    sd = ck.get("ema") if a.weights == "ema" and ck.get("ema") else ck["model"]
    model.load_state_dict(sd, strict=False)

    wrapper = ExportWrapper(model).eval()
    dummy = torch.zeros(1, 3, a.imgsz, a.imgsz)
    if a.half:
        wrapper = wrapper.half()
        dummy = dummy.half()
    with torch.no_grad():
        cls, reg = wrapper(dummy)
    print(f"output shapes: cls {tuple(cls.shape)}  reg {tuple(reg.shape)} "
          f"({cls.shape[1]} anchor points, {NUM_CLASSES} classes)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dyn = {"images": {0: "batch"}, "cls": {0: "batch"}, "dist": {0: "batch"}} if a.dynamic_batch else None
    torch.onnx.export(wrapper, dummy, str(out), opset_version=a.opset,
                      input_names=["images"], output_names=["cls", "dist"],
                      dynamic_axes=dyn, do_constant_folding=True)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")

    if a.simplify:
        try:
            import onnx
            from onnxsim import simplify
            m = onnx.load(str(out))
            ms, ok = simplify(m)
            if ok:
                onnx.save(ms, str(out))
                print(f"simplified -> {out.stat().st_size/1e6:.1f} MB")
            else:
                print("onnxsim could not validate the simplified graph - kept the original")
        except ImportError:
            print("onnxsim not installed - skipped")

    # parity: the ONNX graph must agree with PyTorch before anything is benchmarked
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        o = sess.run(None, {"images": dummy.numpy()})
        d_cls = float(np.abs(o[0] - cls.numpy()).max())
        d_reg = float(np.abs(o[1] - reg.numpy()).max())
        print(f"onnxruntime parity: max|dcls|={d_cls:.2e}  max|ddist|={d_reg:.2e}")
        assert d_cls < 1e-3 and d_reg < 1e-3, "ONNX output diverges from PyTorch"
    except ImportError:
        print("onnxruntime not installed - skipped parity check")


if __name__ == "__main__":
    main()
