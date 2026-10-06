#!/usr/bin/env bash
# Download VisDrone2019-DET and build the uint8 memmap cache.
# Mirror: ultralytics/assets releases - no registration, and it ships the ORIGINAL
# annotations/ txt files (needed for ignore regions, which the YOLO labels drop).
set -euo pipefail
DEST=${1:?usage: get_visdrone.sh <data-dir>}
SIZE=${SIZE:-640}
mkdir -p "$DEST/visdrone_zips" "$DEST/visdrone"
for s in train val; do
  Z="$DEST/visdrone_zips/VisDrone2019-DET-$s.zip"
  if [ ! -s "$Z" ]; then
    echo "downloading $s ..."
    curl -L --retry 3 --retry-delay 2 -o "$Z" \
      "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-$s.zip"
  fi
  [ -d "$DEST/visdrone/VisDrone2019-DET-$s" ] || unzip -q -o "$Z" -d "$DEST/visdrone/"
done
for s in val train; do
  python -m src.data.cache --split-root "$DEST/visdrone/VisDrone2019-DET-$s" \
    --out "$DEST/cache/${s}${SIZE}" --size "$SIZE"
done
