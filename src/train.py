"""Training loop. Written by hand rather than driven by a framework, because being
able to explain every line of it is the point of this project.

Features: AMP (bf16 where supported), EMA, linear warmup + cosine decay, grad clipping,
per-epoch checkpoint + CSV log, resume, and a hard wall-clock budget so a run cannot
overrun the time it was given.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import config as C
from .data.dataset import VisDroneCache, collate
from .data.visdrone import NUM_CLASSES
from .ema import ModelEMA
from .losses.criterion import Criterion
from .losses.distill import distill_loss
from .models.detector import TinyDet
from .schedule import lr_at

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalise(img_u8, mean, std):
    return (img_u8.float() / 255.0 - mean) / std


def build_model(cfg, device):
    m = TinyDet(backbone=cfg.model.backbone, num_classes=NUM_CLASSES,
                out_ch=int(cfg.model.out_ch), p2=bool(cfg.model.get("p2", False)),
                pretrained=bool(cfg.model.get("pretrained", True)),
                head_stacks=int(cfg.model.get("head_stacks", 2)))
    return m.to(device)


def param_groups(model, weight_decay):
    """No weight decay on norms and biases - standard, and it matters on short runs."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser(description="train TinyDet")
    C.add_args(ap)
    ap.add_argument("--out", default=None, help="run dir (default outputs/runs/<run_name>)")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    cfg = C.load(a.config, a.sets)

    run_dir = Path(a.out or f"outputs/runs/{cfg.run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    device = torch.device(cfg.train.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(int(cfg.train.get("seed", 0)))

    ds = VisDroneCache(cfg.paths.train_cache, aug=dict(cfg.aug), train=True,
                       subset=cfg.data.get("subset"))
    nw = int(cfg.train.get("workers", 8))
    dl = DataLoader(ds, batch_size=int(cfg.train.batch), shuffle=True, num_workers=nw,
                    collate_fn=collate, pin_memory=True, drop_last=True,
                    persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)

    model = build_model(cfg, device)
    crit = Criterion(cfg, NUM_CLASSES)
    opt = torch.optim.AdamW(param_groups(model, float(cfg.train.weight_decay)),
                            lr=float(cfg.train.lr), betas=(0.9, 0.999))

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    ema = ModelEMA(model, decay=float(cfg.train.get("ema_decay", 0.9998)),
                   tau=int(cfg.train.get("ema_tau", 2000))) if cfg.train.get("ema", False) else None

    # optional teacher for knowledge distillation
    teacher, dcfg = None, cfg.get("distill", {})
    if dcfg and dcfg.get("teacher_ckpt"):
        tck = torch.load(dcfg["teacher_ckpt"], map_location="cpu", weights_only=False)
        tcfg = C.Cfg(tck.get("cfg") or {})
        teacher = TinyDet(backbone=tcfg.model.backbone, num_classes=NUM_CLASSES,
                          out_ch=int(tcfg.model.out_ch),
                          p2=bool(tcfg.model.get("p2", False)),
                          pretrained=False,
                          head_stacks=int(tcfg.model.get("head_stacks", 2))).to(device)
        teacher.load_state_dict(tck.get("ema") or tck["model"], strict=False)
        teacher.eval()
        for prm in teacher.parameters():
            prm.requires_grad_(False)
        print(f"distilling from {dcfg['teacher_ckpt']} "
              f"({tcfg.model.backbone}, w_cls={dcfg.get('w_cls',1.0)}, "
              f"w_box={dcfg.get('w_box',1.0)})")

    epochs = int(cfg.train.epochs)
    steps_per_epoch = len(dl)
    total_steps = epochs * steps_per_epoch
    warmup = min(int(cfg.train.get("warmup_steps", 500)), max(1, total_steps // 10))
    base_lr = float(cfg.train.lr)
    budget_s = float(cfg.train.get("budget_minutes", 0)) * 60.0

    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    start_epoch, gstep = 0, 0
    if a.resume and Path(a.resume).exists():
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1; gstep = ck.get("gstep", 0)
        if ema and ck.get("ema"):
            ema.ema.load_state_dict(ck["ema"])
        print(f"resumed from {a.resume} at epoch {start_epoch}")

    log_path = run_dir / "log.csv"
    new_log = not log_path.exists()
    logf = log_path.open("a", newline="")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(["epoch", "step", "loss", "cls", "box", "npos", "lr", "epoch_s", "elapsed_s"])

    t_start = time.time()
    stopped_early = False
    print(f"[{cfg.run_name}] {len(ds)} imgs, {steps_per_epoch} steps/epoch, "
          f"{epochs} epochs, amp={use_amp}/{amp_dtype}, budget={budget_s/60:.0f} min")

    for epoch in range(start_epoch, epochs):
        model.train()
        t_ep = time.time()
        agg = {"loss": 0.0, "cls": 0.0, "box": 0.0, "npos": 0.0, "n": 0}
        for batch in dl:
            lr = lr_at(gstep, total_steps, base_lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            img = batch["img"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                x = normalise(img, mean, std)
                out = model(x)
                loss, parts = crit(out, batch)
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(x)
                    kd, kparts = distill_loss(
                        out, t_out,
                        w_cls=float(dcfg.get("w_cls", 1.0)),
                        w_box=float(dcfg.get("w_box", 1.0)),
                        conf_thresh=float(dcfg.get("conf_thresh", 0.1)),
                        box_kind=cfg.loss.box,
                        temperature=float(dcfg.get("temperature", 1.0)))
                    loss = loss + kd
                    parts.update(kparts)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.get("clip", 10.0)))
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.get("clip", 10.0)))
                opt.step()
            if ema:
                ema.update(model)
            gstep += 1
            agg["loss"] += float(loss.detach()); agg["cls"] += float(parts["cls"])
            agg["box"] += float(parts["box"]); agg["npos"] += float(parts["npos"])
            agg["n"] += 1
            if not math.isfinite(agg["loss"]):
                raise SystemExit("loss became non-finite - stop and fix")
            if budget_s and (time.time() - t_start) > budget_s:
                stopped_early = True
                break

        n = max(1, agg["n"])
        ep_s = time.time() - t_ep
        writer.writerow([epoch, gstep, agg["loss"]/n, agg["cls"]/n, agg["box"]/n,
                         agg["npos"]/n, lr, round(ep_s, 1), round(time.time()-t_start, 1)])
        logf.flush()
        print(f"  ep {epoch}: loss {agg['loss']/n:.4f} (cls {agg['cls']/n:.4f} "
              f"box {agg['box']/n:.4f}) npos {agg['npos']/n:.0f} lr {lr:.2e} {ep_s:.0f}s")

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
              "gstep": gstep, "cfg": dict(cfg),
              "ema": ema.ema.state_dict() if ema else None}
        torch.save(ck, run_dir / "last.pt")
        if stopped_early:
            print(f"  wall-clock budget hit after epoch {epoch} - stopping as configured")
            break

    (run_dir / "done.json").write_text(json.dumps({
        "epochs_completed": epoch + 1, "requested_epochs": epochs,
        "stopped_early": stopped_early, "train_seconds": round(time.time() - t_start, 1),
        "run_name": cfg.run_name}, indent=2))
    logf.close()
    print(f"[{cfg.run_name}] done: {epoch+1}/{epochs} epochs in "
          f"{(time.time()-t_start)/60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
