# Cursor Task: Build Full Project Architecture Around Existing Pre/Postprocessing Code

## Context

I'm building a crop/weed vision pipeline for a student engineering project
(SAE-style automated weeder). Input is a phone camera stream, processing
happens on a laptop with an RTX 3050 (6GB VRAM). Model: YOLOv8n-seg,
fine-tuned via transfer learning from COCO weights on a crop/weed dataset
(starting from CWFID / CropAndWeedAndLeaf, supplemented with my own
phone-shot images).

I already have two working, tested modules: `preprocessing.py` and
`postprocessing.py` (attached / in repo root). Do not rewrite their logic
from scratch — review them first, then refactor them INTO a proper project
structure and extend them. Preserve all existing function signatures and
behavior unless you find and explain a concrete bug.

## What preprocessing.py already does
- `FrameSource`: cv2.VideoCapture wrapper with reconnect logic for a flaky
  phone WiFi stream
- `letterbox()` / `unletterbox_point()`: resize to model input size,
  preserving aspect ratio, with coordinate round-trip math
- `normalize_lighting()`: CLAHE-based lighting normalization
- `compute_exg()` / `compute_exg_exr()` / `vegetation_mask()`: colour-only
  vegetation index as a stand-in for a missing IR/multispectral channel
- `preprocess_frame()`: chains the above into one call, returns a
  `PreprocessResult` dataclass

## What postprocessing.py already does
- `Detection` dataclass: class, confidence, centroid, bbox, mask area
- `masks_to_detections()`: raw mask/box/class/score arrays → `Detection`
  objects, with letterbox coordinates unwound back to original frame space
- `from_ultralytics_result()`: the ONE function that touches the
  Ultralytics API directly — keep this isolation, don't leak Ultralytics
  types into other modules
- `filter_detections()`: confidence + area filtering
- `CentroidTracker`: nearest-neighbour matching + EMA smoothing across
  frames to reduce centroid jitter
- `draw_overlay()`: debug visualization
- `export_frame_result()` / `export_to_json()`: structured output

## Task 1 — Restructure into a real project

Create this directory layout, moving/splitting the existing code into it
without breaking any function:

```
project_root/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── pipeline_config.yaml       # frame source URL, target_size, thresholds, etc.
│   └── dataset.yaml               # YOLO dataset config (train/val paths, class names)
├── data/
│   ├── raw/                       # unlabeled phone-shot images go here
│   ├── external/                  # downloaded public datasets (CWFID etc.), gitignored
│   └── yolo_dataset/              # final merged, YOLO-formatted train/val/test split
├── models/
│   ├── pretrained/                # yolov8n-seg.pt etc.
│   └── finetuned/                 # your trained weights, versioned by date/experiment
├── src/
│   ├── __init__.py
│   ├── capture/
│   │   └── frame_source.py        # FrameSource class extracted here
│   ├── preprocessing/
│   │   └── preprocessing.py       # letterbox, CLAHE, ExG, preprocess_frame
│   ├── postprocessing/
│   │   └── postprocessing.py      # Detection, filtering, tracker, overlay, export
│   ├── inference/
│   │   └── live_pipeline.py       # ties capture -> preprocess -> model -> postprocess -> output
│   ├── training/                  # SEE TASK 2 — kept fully separate from inference code
│   └── utils/
│       └── logging_config.py
├── scripts/
│   ├── setup_env.sh               # venv + pip install -r requirements.txt
│   └── download_datasets.sh       # pulls CWFID / CropAndWeedAndLeaf
├── tests/
│   ├── test_preprocessing.py      # port the existing __main__ self-tests into real pytest tests
│   └── test_postprocessing.py
└── notebooks/                     # optional, for dataset exploration / debugging only
```

Requirements for this task:
- Add type hints and docstrings anywhere missing.
- Convert the `if __name__ == "__main__":` self-tests in both files into
  proper `pytest` tests under `tests/`, using the same synthetic-data
  approach (no camera/model dependency required to run tests).
- Add a `pipeline_config.yaml` that externalizes every magic number
  currently hardcoded in the two modules (target_size=640, min_conf=0.4,
  min_area_px=50, max_match_dist_px=40.0, smoothing_alpha=0.5, CLAHE
  clip_limit=2.0, etc.) so thresholds can be tuned without touching code.
- Write `src/inference/live_pipeline.py` as the glue script: loads config,
  opens `FrameSource`, loads a `.pt` model via Ultralytics, runs the loop
  (capture → preprocess → `model.predict()` → `from_ultralytics_result()`
  → `filter_detections()` → `CentroidTracker.update()` → `draw_overlay()`
  → display/export), with a clean shutdown on Ctrl+C.
- Do NOT put any Ultralytics import inside `preprocessing/` or
  `postprocessing/` beyond the existing isolated adapter function — keep
  that boundary intact.

## Task 2 — Separate training architecture

Build `src/training/` as a fully separate concern from `src/inference/`.
Training code should never be imported by the live pipeline, and vice
versa. Structure:

```
src/training/
├── __init__.py
├── prepare_dataset.py   # merges public dataset(s) + own labeled images into data/yolo_dataset/, validates label format
├── train.py              # wraps `ultralytics` training call, reads hyperparams from configs/train_config.yaml
├── evaluate.py            # runs validation/test split, reports precision/recall/mAP, dumps a report
└── export.py               # exports best weights to models/finetuned/, optionally to ONNX
```

Requirements:
- `configs/train_config.yaml` should hold: base weights (`yolov8n-seg.pt`),
  epochs, batch size, image size, learning rate, dataset yaml path, output
  dir. Batch size default should assume 6GB VRAM (start at 4-8, document
  that it may need lowering).
- `prepare_dataset.py` should validate that image/label pairs exist and
  that class indices in labels match `configs/dataset.yaml` before writing
  anything into `data/yolo_dataset/` — fail loudly on mismatch rather than
  silently training on bad data.
- `train.py` should be runnable as `python -m src.training.train --config configs/train_config.yaml` and should log to a timestamped run directory under `models/finetuned/<run_name>/`, keeping every run's weights and metrics separate rather than overwriting the last one.
- `evaluate.py` should specifically report per-class metrics (crop vs
  weed separately, not just aggregate mAP), since class imbalance is a
  known risk with this dataset.

## Task 3 — Improve the existing code (only where genuinely warranted)

While integrating, look for and fix (with an explanation of what and why,
don't change silently):
- Any place `CentroidTracker`'s greedy nearest-neighbour matching could
  misassign IDs when two same-class detections are close together —
  flag this as a known limitation in a code comment if not fixing it now.
- Confirm `letterbox()`'s coordinate math is correct for non-square
  source frames in all four aspect-ratio cases (wider-than-tall,
  taller-than-wide, and both padding directions) — add test cases for
  this in `tests/test_preprocessing.py`.
- Add basic error handling in `live_pipeline.py` for the case where
  `FrameSource.read()` returns `None` repeatedly (currently the caller
  must handle this — make sure it does, with a clear log message rather
  than a crash).

## Constraints

- Keep everything runnable on a laptop with 6GB VRAM — don't introduce
  dependencies or defaults that assume more.
- No cloud services, no paid APIs. Ultralytics + PyTorch + OpenCV +
  NumPy is the expected stack.
- Ask me before making any architectural decision not covered above
  (e.g. if you think a different tracker or a different config format
  would be better) rather than silently deciding.

## Deliverable

Once restructured, give me a short summary of what moved where, what
you added, and any open questions/tradeoffs you want me to weigh in on
before you proceed to writing the actual training run.
