#!/usr/bin/env bash
# Download public crop/weed datasets into data/external/
# Run manually when you are ready to prepare training data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL="$ROOT/data/external"
mkdir -p "$EXTERNAL"

echo "=== Crop/Weed Dataset Download ==="
echo ""
echo "This script provides download instructions for common datasets."
echo "Automated downloads may require manual registration or git clone."
echo ""

# CropAndWeed dataset (GitHub)
CROPANDWEED_DIR="$EXTERNAL/CropAndWeed"
if [ ! -d "$CROPANDWEED_DIR" ]; then
  echo "Cloning CropAndWeed dataset..."
  git clone --depth 1 https://github.com/cropandweed/cropandweed-dataset.git "$CROPANDWEED_DIR" || {
    echo "WARNING: Could not clone CropAndWeed. Download manually from:"
    echo "  https://github.com/cropandweed/cropandweed-dataset"
  }
else
  echo "CropAndWeed already present at $CROPANDWEED_DIR"
fi

# CWFID — typically distributed via research portals
CWFID_DIR="$EXTERNAL/CWFID"
mkdir -p "$CWFID_DIR"
echo ""
echo "CWFID (Crop-Weed-Fruit Image Dataset):"
echo "  Download from the original publication/source and extract to:"
echo "  $CWFID_DIR"
echo ""
echo "After downloading, convert labels to YOLO segmentation format if needed,"
echo "then run prepare_dataset.py to merge into data/yolo_dataset/."
echo ""
echo "Example:"
echo "  python -m src.training.prepare_dataset \\"
echo "    --source data/external/cwfid/images data/external/cwfid/labels train cwfid_"
