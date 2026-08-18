# Crop/Weed Vision Pipeline

Automated crop vs weed instance segmentation for an SAE-style automated weeder.
Phone camera stream → laptop inference (RTX 3050, 6GB VRAM) → centroid output for actuation.

## Stack

- **Ultralytics YOLOv8n-seg** (segmentation)
- **OpenCV** (capture, preprocessing, visualization)
- **NumPy**, **PyYAML**, **pytest**

## Project layout

```
configs/          Pipeline, dataset, and training YAML configs
data/             Raw images, external datasets, YOLO-formatted splits
models/           Pretrained and fine-tuned weights
src/
  capture/        FrameSource (phone stream with reconnect)
  preprocessing/  Letterbox, CLAHE, ExG vegetation index
  postprocessing/ Detections, filtering, CentroidTracker, export
  inference/      Live pipeline (capture → model → overlay)
  training/       Dataset prep, train, evaluate, export (separate from inference)
scripts/          setup_env.sh, download_datasets.sh
tests/            pytest unit tests (no camera/model required)
```

## Quick start

```bash
# 1. Set up environment
bash scripts/setup_env.sh
source .venv/bin/activate

# 2. Place model weights (not auto-downloaded)
#    Copy yolov8n-seg.pt to models/pretrained/

# 3. Edit configs/pipeline_config.yaml (frame source URL, thresholds)

# 4. Run live pipeline
python -m src.inference.live_pipeline --config configs/pipeline_config.yaml

# 5. Run tests (no model needed)
pytest tests/ -v
```

## Training (separate from inference)

```bash
# Download datasets (manual step for some sources)
bash scripts/download_datasets.sh

# Merge and validate into YOLO format
python -m src.training.prepare_dataset \
  --source data/external/myset/images data/external/myset/labels train prefix_

# Train (writes to models/finetuned/<run_name>_<timestamp>/)
python -m src.training.train --config configs/train_config.yaml

# Evaluate with per-class metrics
python -m src.training.evaluate --weights models/finetuned/best.pt

# Export weights for inference
python -m src.training.export --weights models/finetuned/<run>/train/weights/best.pt
```

## Configuration

All tunable thresholds live in YAML — no code edits needed:

| Config | Purpose |
|--------|---------|
| `configs/pipeline_config.yaml` | Frame source, preprocessing, filtering, tracking, display |
| `configs/dataset.yaml` | YOLO dataset paths and class names |
| `configs/train_config.yaml` | Training hyperparameters (batch=8 default for 6GB VRAM) |

## Design notes

- **Ultralytics isolation**: only `from_ultralytics_result()` in postprocessing and the inference/training entry points import Ultralytics.
- **CentroidTracker**: greedy nearest-neighbour matching; can swap IDs when same-class detections are very close (documented limitation).
- **Training ≠ inference**: `src/training/` is never imported by `live_pipeline.py`.

## Hardware

Target: laptop with RTX 3050 (6GB VRAM). Lower `batch` in `train_config.yaml` to 4 if you hit OOM.
