#!/usr/bin/env bash
# Evaluate every finished ablation run (EMA and raw weights), keep the better one.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPU=${GPU:-1}
for run in a0_baseline a1_aug_ema a2_tal_ciou; do
  CK=outputs/runs/$run/last.pt
  [ -f "$CK" ] || { echo "skip $run (no checkpoint)"; continue; }
  BEST=""; BESTAP=-1
  for w in model ema; do
    OUT=outputs/runs/$run/metrics_$w.json
    CUDA_VISIBLE_DEVICES=$GPU $PY -m src.eval.evaluate --config configs/base.yaml \
      --ckpt "$CK" --weights $w --tag "$run" --out "$OUT" >/dev/null 2>&1 || continue
    AP=$($PY -c "import json;print(json.load(open('$OUT'))['AP'])" 2>/dev/null || echo -1)
    echo "  $run/$w -> AP $AP"
    GT=$($PY -c "print(1 if float('$AP')>float('$BESTAP') else 0)")
    [ "$GT" = "1" ] && { BEST=$OUT; BESTAP=$AP; }
  done
  [ -n "$BEST" ] && cp "$BEST" outputs/runs/$run/metrics_best.json && echo "  $run best: $BESTAP ($BEST)"
done
