"""
One-off converter for manually downloaded CWFID data.

CWFID is not part of the automated baseline pipeline. After downloading and
labeling images yourself, place them under:

  data/external/CWFID/raw/images/
  data/external/CWFID/raw/labels/   # YOLO segmentation .txt files

Then run:

  python -m src.training.convert_cwfid --input data/external/CWFID/raw

This validates labels against configs/dataset.yaml, writes a flat pool under
data/external/CWFID/pool/, and optionally merges into the existing yolo_dataset
via split_dataset or prepare_dataset.

Expected label format: YOLOv8 segmentation (class_id x1 y1 x2 y2 ... xn yn),
with class 0=crop and class 1=weed matching configs/dataset.yaml.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.training.prepare_dataset import validate_source_dataset
from src.utils.config import load_yaml, resolve_project_path
from src.utils.logging_config import setup_logging

logger = setup_logging()


def convert_cwfid(
    raw_root: Path,
    output_pool: Path,
    dataset_config_path: Path,
    prefix: str = "cwfid_",
) -> int:
    """
    Validate and copy manually prepared CWFID labels into a flat YOLO pool.

    Does not download anything — run this only after manual acquisition.
    """
    images_dir = raw_root / "images"
    labels_dir = raw_root / "labels"

    ds_cfg = load_yaml(dataset_config_path)
    valid_classes = {int(k) for k in ds_cfg.get("names", {}).keys()}

    pairs = validate_source_dataset(images_dir, labels_dir, valid_classes)

    out_images = output_pool / "images"
    out_labels = output_pool / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        stem = f"{prefix}{img_path.stem}"
        shutil.copy2(img_path, out_images / f"{stem}{img_path.suffix.lower()}")
        shutil.copy2(lbl_path, out_labels / f"{stem}.txt")

    logger.info(
        "CWFID: copied %d validated pairs -> %s. "
        "Merge into training set with split_dataset or prepare_dataset.",
        len(pairs),
        output_pool,
    )
    return len(pairs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-off CWFID validation/copy (manual download required)"
    )
    parser.add_argument(
        "--input",
        default="data/external/CWFID/raw",
        help="Directory with images/ and labels/ subfolders",
    )
    parser.add_argument(
        "--output",
        default="data/external/CWFID/pool",
        help="Output flat YOLO pool",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
        help="Class name validation config",
    )
    parser.add_argument(
        "--prefix",
        default="cwfid_",
        help="Filename prefix to avoid collisions when merging datasets",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    raw_root = resolve_project_path(args.input, project_root)
    output_pool = resolve_project_path(args.output, project_root)
    dataset_config_path = resolve_project_path(args.dataset_config, project_root)

    try:
        convert_cwfid(raw_root, output_pool, dataset_config_path, prefix=args.prefix)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("CWFID conversion failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
