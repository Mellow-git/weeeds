#!/usr/bin/env bash
# End-to-end baseline dataset prep: download CropAndWeed -> YOLO pool -> 80/10/10 split
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "=== Step 1/3: Download CropAndWeed ==="
python -m src.training.download_cropandweed "$@"

echo ""
echo "=== Step 2/3: Convert to YOLO segmentation pool ==="
python -m src.training.convert_cropandweed

echo ""
echo "=== Step 3/3: Random 80/10/10 split ==="
python -m src.training.split_dataset

echo ""
echo "Baseline dataset ready at data/yolo_dataset/"
echo "Next: python -m src.training.train --config configs/train_config.yaml"
