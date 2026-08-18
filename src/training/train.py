"""
Fine-tune YOLOv8 segmentation on the crop/weed dataset.

Run: python -m src.training.train --config configs/train_config.yaml

Each run writes to models/finetuned/<run_name>_<timestamp>/ with weights
and metrics kept separate from prior runs.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.utils.config import load_yaml, resolve_project_path
from src.utils.logging_config import setup_logging

logger = setup_logging()


def train(config_path: Path, project_root: Path) -> Path:
    """Run Ultralytics training and return the run output directory."""
    from ultralytics import YOLO

    cfg = load_yaml(config_path)

    base_weights = resolve_project_path(cfg["base_weights"], project_root)
    if not base_weights.exists():
        raise FileNotFoundError(
            f"Base weights not found: {base_weights}. "
            "Download yolov8n-seg.pt to models/pretrained/ first."
        )

    dataset_yaml = resolve_project_path(cfg["dataset"], project_root)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")

    run_name = cfg.get("run_name", "crop_weed_seg")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = resolve_project_path(cfg.get("output_dir", "models/finetuned"), project_root)
    run_dir = output_base / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training run -> %s", run_dir)
    logger.info("Base weights: %s", base_weights)
    logger.info("Dataset: %s", dataset_yaml)

    model = YOLO(str(base_weights))
    results = model.train(
        data=str(dataset_yaml),
        epochs=cfg.get("epochs", 100),
        batch=cfg.get("batch", 8),
        imgsz=cfg.get("imgsz", 640),
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.01),
        patience=cfg.get("patience", 20),
        workers=cfg.get("workers", 4),
        device=cfg.get("device", "0"),
        project=str(run_dir),
        name="train",
        exist_ok=True,
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.7),
        hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 5.0),
        flipud=cfg.get("flipud", 0.0),
        fliplr=cfg.get("fliplr", 0.5),
        mosaic=cfg.get("mosaic", 1.0),
    )

    best_weights = run_dir / "train" / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best_weights)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train crop/weed YOLO segmentation model")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    config_path = resolve_project_path(args.config, project_root)

    try:
        train(config_path, project_root)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
