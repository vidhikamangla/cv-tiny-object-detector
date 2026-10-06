# TinyDet — anchor-free small-object detection for drone imagery

A detector whose head, losses, assigners and training loop are written from scratch,
trained on **VisDrone2019-DET** to answer one question with numbers: *which design
choices actually help objects that are 6–30 pixels across?*

The training loop, the label assigners, the box losses, the augmentations and the tiled
inference are all implemented in this repo. The ImageNet backbone comes from `timm`, and
mAP comes from `pycocotools` — everything between those two is mine.

![single pass vs tiled inference](assets/tiling_comparison.jpg)

*The same frame and the same weights. Left: one 640 px pass, 47 detections. Right: SAHI-style tiled inference, 83 detections - the pedestrians in the far plaza only exist at native resolution.*


## Why this exists

Most "I trained a detector" projects run `yolo train` and report one number. That shows
you can drive a CLI. This project is organised as a **pre-registered ablation ladder**:
the rows were fixed before any of them ran, each row adds exactly one component, and
every row is reported — including the ones that did not help.

## Architecture

```
image 640x640
   │
   ├─ timm resnet18, features_only, out_indices (2,3,4)   strides  8 / 16 / 32
   │
   ├─ FPN top-down  +  PAN bottom-up          (128 ch)
   │     top-down injects semantics into the stride-8 map;
   │     bottom-up puts fine detail back into the deeper maps
   │
   └─ decoupled head, shared across levels, per-level learnable box scale
         ├─ cls: 10 logits per anchor point   (bias init to prior 0.01)
         └─ reg: softplus(ltrb) x stride      (distances in pixels)
                │
                └─ decode + batched NMS  (outside the graph, and outside the ONNX export)
```

Anchor-free, one point per feature-map cell, `ltrb` distance regression. The head is
shared across levels because the regression target is expressed in *stride units*, so
the same filters see the same apparent object size at every scale.

## The two assigners

Label assignment is where small-object detection is won or lost, so both options are
implemented in `src/losses/assigner.py`:

| | rule | character |
|---|---|---|
| **centre-prior** (FCOS-like) | a point is a candidate if it is inside the GT box *and* within `radius` strides of its centre; the top-k closest win | static — the same point is positive from epoch 0 |
| **TAL** (TOOD-like) | score every in-box point by `s^alpha * iou^beta`, take top-k per GT, and use the normalised metric as a **soft** classification target | prediction-aware — aligns confidence with localisation quality |

TAL is expected to help tiny boxes specifically: a 12×12 box covers roughly one
stride-8 cell, so "the point nearest the centre" and "the point the network can
actually regress from" are not the same point, and only TAL can tell the difference.

A subtlety that cost a real bug (caught by `tests/test_assigner.py`): a GT that wins no
candidate contributes **no gradient at all**. TAL therefore rescues such objects with
their best in-box point, breaking ties on centre distance so the choice is defined even
at step 0 when every prediction is garbage.

## Ignore regions

VisDrone marks crowds and unlabelled areas as category 0. Those points are dropped from
the classification loss rather than trained as background — otherwise the model is
actively taught to suppress real objects. This is why the parser reads the original
`annotations/*.txt` instead of the pre-converted YOLO labels, which discard them.

## Results

| Row | Change | Params M | Epochs | Train min | ms/img | AP | AP50 | AP75 | AP tiny | AP small | AP med | AR100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | centre-prior assigner, BCE + GIoU, flip only, no EMA | 12.8 | 30 | 50.1 | 3.5 | 9.8 | 20.3 | 8.7 | 4.0 | 14.2 | 24.7 | 20.8 |
| A1 | + mosaic, HSV, scale jitter, EMA | 12.8 | 30 | 42.3 | 3.5 | 13.1 | 25.7 | 11.9 | 5.1 | 19.2 | 31.9 | 26.8 |
| A2 | + TAL assigner, CIoU | 12.8 | 30 | 50.5 | 3.3 | 14.4 | 27.3 | 13.8 | 5.8 | 21.7 | 35.2 | 26.3 |

AP columns are COCO mAP x100. Size bins are ours, not COCO's: tiny < 16^2 px, small 16^2-32^2, med 32^2-96^2 - COCO's `small < 32^2` puts almost all of VisDrone in one bucket and hides the effect.

![AP by object size](outputs/figures/ap_by_size.png)

### Honest framing of the compute budget

All rows ran for **30 epochs at 640 px on one RTX A4000** (143 GPU-minutes total across the three rows), with the epoch count fixed in advance so the rows differ only in the component under test and not in how long they trained.

Published VisDrone results sit far higher (AP 25+) using ~1 500 px inputs, much longer schedules and heavier backbones. The numbers below are not competitive with those and are not meant to be: the deltas between rows are the result.

## Verification

The numbers are only worth reading because these checks pass first:

| check | why it matters |
|---|---|
| `pytest tests/test_iou.py` | our GIoU/DIoU/CIoU match `torchvision`'s reference losses to 1e-4 |
| `pytest tests/test_assigner.py` | assigner behaviour on hand-computed toy cases, including the no-gradient bug above |
| overfit 32 images | 200 epochs on 32 images drove the loss from 29.05 to 0.66 and reached **AP 69.8 / AP50 88.1 / AP_med 91.1** on those images. It does not reach AP 90 because AP_tiny caps at 51.6: with a stride-8 finest feature map, the smallest VisDrone boxes are simply not representable, which is itself the argument for the P2 level in `later.md`. |

```bash
pytest tests/ -q
python -m src.train --config configs/base.yaml --set data.subset=32 train.epochs=200
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./scripts/get_visdrone.sh data            # ~2 GB download, builds a 640px memmap cache
pytest tests/ -q

# one ablation row per GPU
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$i python -m src.train --config configs/ablations/a$i.yaml &
done; wait

python -m src.eval.evaluate --ckpt outputs/runs/a2_tal_ciou/last.pt
python -m src.analysis.make_table
python -m src.export_onnx --ckpt outputs/runs/a2_tal_ciou/last.pt --out outputs/tinydet.onnx
python -m src.demo.app --ckpt outputs/runs/a2_tal_ciou/last.pt        # gradio demo
```

Every script reads `configs/base.yaml`; an ablation row is a diff against it, and any
key can be overridden on the command line:

```bash
python -m src.train --set loss.assigner=tal loss.box=ciou train.epochs=30
```

## Layout

```
configs/base.yaml            single source of truth
configs/ablations/a*.yaml    one diff per ablation row
src/data/      visdrone.py (incl. ignore regions) · cache.py (memmap) · transforms.py (mosaic, HSV, affine)
src/models/    backbone · neck (FPN+PAN) · head · detector · decode
src/losses/    iou.py (GIoU/DIoU/CIoU) · cls.py (BCE/focal/varifocal) · assigner.py · criterion.py
src/eval/      coco_eval.py (custom size bins) · evaluate.py · sahi.py (tiled inference + NMM)
src/           train.py · ema.py · schedule.py · export_onnx.py · infer.py · demo/app.py
tests/         iou vs torchvision · assigner toy cases · overfit gate
```

## Notes and limitations

- Trained at 640 px on a fixed wall-clock budget, so the absolute AP is well below
  published VisDrone SOTA (which uses ~1500 px inputs and far longer schedules). The
  ablation *deltas* are the result, not the absolute numbers.
- No stride-4 (P2) level in these runs. It is the single most promising next step for
  `AP_tiny`, and it roughly doubles training time — see `later.md`.
- VisDrone is released for academic, non-commercial use; no images are redistributed
  here, only the download script.
- `ultralytics` appears in `requirements.txt` only for pretrained *comparison* weights
  and is AGPL-3.0; nothing in `src/` derives from it.

## License

Code: MIT (see `LICENSE`). Dataset and third-party weights keep their own licenses.
