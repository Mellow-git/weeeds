#!/usr/bin/env bash
# Download CropAndWeed (primary baseline dataset).
# CWFID is manual — see README and convert_cwfid.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== CropAndWeed (automated) ==="
python -m src.training.download_cropandweed "$@"

echo ""
echo "Next steps:"
echo "  python -m src.training.convert_cropandweed"
echo "  python -m src.training.split_dataset"
echo ""
echo "Or run the full chain:"
echo "  bash scripts/prepare_baseline_dataset.sh"
echo ""
echo "=== CWFID (manual, one-off) ==="
echo "  1. Download CWFID from the original publication/source"
echo "  2. Label/convert to YOLO segmentation format"
echo "  3. Place under data/external/CWFID/raw/{images,labels}/"
echo "  4. python -m src.training.convert_cwfid"
echo "  5. Merge pool into training data separately when ready"
