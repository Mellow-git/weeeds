"""
Export trained weights to models/finetuned/ and optionally to ONNX.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.utils.config import resolve_project_path
from src.utils.logging_config import setup_logging

logger = setup_logging()


def export_weights(
    weights_path: Path,
    dest_dir: Path,
    export_onnx: bool = False,
    onnx_imgsz: int = 640,
) -> Path:
    """Copy best weights to dest_dir and optionally export ONNX."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_pt = dest_dir / f"best_{timestamp}.pt"
    shutil.copy2(weights_path, dest_pt)
    logger.info("Exported weights -> %s", dest_pt)

    # Also maintain a stable symlink/copy name for pipeline config
    latest = dest_dir / "best.pt"
    shutil.copy2(weights_path, latest)
    logger.info("Updated latest weights -> %s", latest)

    if export_onnx:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        onnx_path = dest_dir / f"best_{timestamp}.onnx"
        model.export(format="onnx", imgsz=onnx_imgsz)
        # Ultralytics writes next to source weights; move if needed
        default_onnx = weights_path.with_suffix(".onnx")
        if default_onnx.exists() and default_onnx != onnx_path:
            shutil.move(str(default_onnx), str(onnx_path))
        logger.info("Exported ONNX -> %s", onnx_path)

    return dest_pt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export trained model weights")
    parser.add_argument("--weights", required=True, help="Path to source .pt weights")
    parser.add_argument(
        "--dest",
        default="models/finetuned",
        help="Destination directory",
    )
    parser.add_argument("--onnx", action="store_true", help="Also export ONNX")
    parser.add_argument("--imgsz", type=int, default=640, help="ONNX export image size")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    weights = resolve_project_path(args.weights, project_root)
    dest = resolve_project_path(args.dest, project_root)

    try:
        export_weights(weights, dest, export_onnx=args.onnx, onnx_imgsz=args.imgsz)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
