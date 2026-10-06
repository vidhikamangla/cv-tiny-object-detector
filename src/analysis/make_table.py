"""Collect per-run metrics json into the ablation table + an AP-by-size chart."""
import argparse
import csv
import json
from pathlib import Path

LABELS = {
    "a0_baseline": "centre-prior assigner, BCE + GIoU, flip only, no EMA",
    "a1_aug_ema": "+ mosaic, HSV, scale jitter, EMA",
    "a2_tal_ciou": "+ TAL assigner, CIoU",
    "a3_p2_head": "+ stride-4 (P2) level",
    "a4_distilled": "A2 distilled from the ResNet-50 teacher",
    "teacher_r50": "ResNet-50 teacher (reference, not a student row)",
}
ROW_ID = {"a0_baseline": "A0", "a1_aug_ema": "A1", "a2_tal_ciou": "A2",
          "a3_p2_head": "A3", "a4_distilled": "A4", "teacher_r50": "T"}
COLS = ["AP", "AP50", "AP75", "AP_tiny", "AP_small", "AP_med", "AR100"]


def collect(runs_dir, extra=()):
    rows = []
    for name, label in LABELS.items():
        f = Path(runs_dir) / name / "metrics_best.json"
        if not f.exists():
            for alt in ("metrics_ema.json", "metrics_model.json"):
                if (Path(runs_dir) / name / alt).exists():
                    f = Path(runs_dir) / name / alt
                    break
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        rows.append({"row": ROW_ID.get(name, name), "change": label, **d})
    for path in extra:
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text())
            rows.append({"row": d.get("row", d.get("tag", p.stem)),
                         "change": d.get("change", d.get("tag", p.stem)), **d})
    return rows


def main():
    ap = argparse.ArgumentParser(description="build the detector ablation table")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--extra", nargs="*", default=[], help="extra metrics json files")
    ap.add_argument("--out-md", default="outputs/results/ablation.md")
    ap.add_argument("--out-csv", default="outputs/results/ablation.csv")
    ap.add_argument("--chart", default="outputs/figures/ap_by_size.png")
    ap.add_argument("--note", default=None)
    a = ap.parse_args()

    rows = collect(a.runs_dir, a.extra)
    if not rows:
        raise SystemExit(f"no metrics json found under {a.runs_dir}")

    keys = ["row", "change", "params_M", "epochs_completed", "train_minutes",
            "ms_per_img"] + COLS
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{100*v:.1f}" if abs(v) <= 1.0 else f"{v:.1f}"
        return str(v)

    header = ["Row", "Change", "Params M", "Epochs", "Train min", "ms/img"] + \
             [c.replace("AP_", "AP ") for c in COLS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [str(r.get("row", "")), str(r.get("change", "")),
                 fmt(r.get("params_M")), str(r.get("epochs_completed", "-")),
                 fmt(r.get("train_minutes")), fmt(r.get("ms_per_img"))]
        cells += [fmt(r.get(c)) for c in COLS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("AP columns are COCO mAP x100. Size bins are ours, not COCO's: "
                 "tiny < 16^2 px, small 16^2-32^2, med 32^2-96^2 - COCO's "
                 "`small < 32^2` puts almost all of VisDrone in one bucket and hides the effect.")
    if a.note:
        lines += ["", a.note]
    Path(a.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        bins = ["AP_tiny", "AP_small", "AP_med"]
        labels = [b.replace("AP_", "") for b in bins]
        x = np.arange(len(bins))
        width = 0.8 / max(1, len(rows))
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        for i, r in enumerate(rows):
            vals = [100 * float(r.get(b) or 0) for b in bins]
            ax.bar(x + i * width, vals, width, label=f"{r['row']}")
        ax.set_xticks(x + width * (len(rows) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("AP (%)")
        ax.set_title("AP by object size - VisDrone2019-DET val")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        Path(a.chart).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.chart)
        print(f"chart -> {a.chart}")
    except ImportError:
        print("matplotlib missing - skipped chart")


if __name__ == "__main__":
    main()
