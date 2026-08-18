"""
Evaluate a trained YOLO segmentation model with per-class metrics.

Reports precision, recall, and mAP separately for each class (crop vs weed)
since class imbalance is a known risk with this dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.config import load_yaml, resolve_project_path
from src.utils.logging_config import setup_logging

logger = setup_logging()


def evaluate(
    weights_path: Path,
    dataset_config_path: Path,
    split: str = "val",
    output_report: Path | None = None,
) -> dict:
    """Run validation and return per-class metrics."""
    from ultralytics import YOLO

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    ds_cfg = load_yaml(dataset_config_path)
    class_names = {int(k): v for k, v in ds_cfg.get("names", {}).items()}

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(dataset_config_path), split=split)

    report: dict = {
        "weights": str(weights_path),
        "split": split,
        "aggregate": {},
        "per_class": {},
    }

    # Aggregate box metrics (Ultralytics Results object)
    if hasattr(metrics, "box"):
        box = metrics.box
        report["aggregate"] = {
            "mAP50": float(getattr(box, "map50", 0)),
            "mAP50-95": float(getattr(box, "map", 0)),
            "precision": float(getattr(box, "mp", 0)),
            "recall": float(getattr(box, "mr", 0)),
        }

        # Per-class: mp, mr, map50 indexed by class
        for cls_id, cls_name in class_names.items():
            p = float(box.p[cls_id]) if cls_id < len(box.p) else 0.0
            r = float(box.r[cls_id]) if cls_id < len(box.r) else 0.0
            ap50 = float(box.ap50[cls_id]) if cls_id < len(box.ap50) else 0.0
            ap = float(box.ap[cls_id]) if cls_id < len(box.ap) else 0.0
            report["per_class"][cls_name] = {
                "class_id": cls_id,
                "precision": p,
                "recall": r,
                "mAP50": ap50,
                "mAP50-95": ap,
            }

    # Segmentation metrics if available
    if hasattr(metrics, "seg"):
        seg = metrics.seg
        report["segmentation_aggregate"] = {
            "mAP50": float(getattr(seg, "map50", 0)),
            "mAP50-95": float(getattr(seg, "map", 0)),
        }
        for cls_id, cls_name in class_names.items():
            if cls_name not in report["per_class"]:
                report["per_class"][cls_name] = {"class_id": cls_id}
            if cls_id < len(seg.ap50):
                report["per_class"][cls_name]["seg_mAP50"] = float(seg.ap50[cls_id])
            if cls_id < len(seg.ap):
                report["per_class"][cls_name]["seg_mAP50-95"] = float(seg.ap[cls_id])

    logger.info("=== Aggregate metrics ===")
    for k, v in report.get("aggregate", {}).items():
        logger.info("  %s: %.4f", k, v)

    logger.info("=== Per-class metrics ===")
    for cls_name, cls_metrics in report["per_class"].items():
        logger.info("  %s:", cls_name)
        for k, v in cls_metrics.items():
            if k != "class_id" and isinstance(v, float):
                logger.info("    %s: %.4f", k, v)

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Report saved to %s", output_report)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate crop/weed model with per-class metrics")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
        help="YOLO dataset config",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="JSON report output path")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    weights = resolve_project_path(args.weights, project_root)
    dataset_cfg = resolve_project_path(args.dataset_config, project_root)
    output = resolve_project_path(args.output, project_root) if args.output else None

    try:
        evaluate(weights, dataset_cfg, split=args.split, output_report=output)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
