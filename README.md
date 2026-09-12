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
scripts/          setup_env.ps1/.sh, prepare_baseline_dataset.ps1/.sh
tests/            pytest unit tests (no camera/model required)
```

## Quick start (Windows)

```powershell
# 1. Set up environment
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1

# 2. Place model weights (not auto-downloaded)
#    Copy yolov8n-seg.pt to models/pretrained/

# 3. Edit configs/pipeline_config.yaml (frame source URL, thresholds)

# 4. Run live pipeline
python -m src.inference.live_pipeline --config configs/pipeline_config.yaml

# 5. Run tests (no model needed)
pytest tests/ -v
```

On Linux/macOS, use `bash scripts/setup_env.sh` instead.

## Training baseline (CropAndWeed — automated)

```powershell
# Full chain: download -> YOLO conversion -> random 80/10/10 split
.\scripts\prepare_baseline_dataset.ps1

# Or step by step:
python -m src.training.download_cropandweed
python -m src.training.convert_cropandweed
python -m src.training.split_dataset --seed 42

# Train baseline (default augmentation, no class reweighting yet)
python -m src.training.train --config configs/train_config.yaml

# Evaluate per-class metrics before tuning anything else
python -m src.training.evaluate --weights models/finetuned/best.pt

# Export weights for inference
python -m src.training.export --weights models/finetuned/<run>/train/weights/best.pt
```

After training, inspect per-class precision/recall/mAP. Only add imbalance handling
(class weights, oversampling) if the confusion matrix shows it's needed.

## CWFID (manual, one-off — second dataset)

CWFID is not part of the automated baseline pipeline:

1. Download from the original publication/source
2. Label or convert to YOLO segmentation format (class 0=crop, 1=weed)
3. Place under `data/external/CWFID/raw/images/` and `.../labels/`
4. Run the one-off converter:

```powershell
python -m src.training.convert_cwfid
```

5. When ready to merge, combine the CWFID pool with CropAndWeed (e.g. copy into
   a combined pool and re-run `split_dataset`, or use `prepare_dataset.py` for
   explicit split assignment).

## Configuration

All tunable thresholds live in YAML — no code edits needed:

| Config | Purpose |
|--------|---------|
| `configs/pipeline_config.yaml` | Frame source, preprocessing, filtering, tracking, display |
| `configs/dataset.yaml` | YOLO dataset paths and class names |
| `configs/train_config.yaml` | Training hyperparameters (batch=8 default for 6GB VRAM) |

Set `display.show_exg_overlay: true` in `pipeline_config.yaml` to show an ExG
debug panel in the live view. ExG is auxiliary only — not fed to the model.

## Design decisions

- **CropAndWeed first**: automated download + CropOrWeed2 mapping + YOLO conversion
- **CWFID later**: manual download/labeling via `convert_cwfid.py`
- **Baseline before tuning**: default augmentation, no class reweighting until metrics say so
- **CentroidTracker**: simple nearest-neighbour tracker; upgrade to ByteTrack only if needed
- **Ultralytics isolation**: only `from_ultralytics_result()` in postprocessing and training/inference entry points import Ultralytics
- **Training ≠ inference**: `src/training/` is never imported by `live_pipeline.py`

## Hardware

Target: laptop with RTX 3050 (6GB VRAM). Lower `batch` in `train_config.yaml` to 4 if you hit OOM.
