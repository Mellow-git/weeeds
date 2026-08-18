"""
Merge and validate datasets into YOLO format under data/yolo_dataset/.

Validates that every image has a matching label file and that class
indices in labels match configs/dataset.yaml before writing anything.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable

import yaml

from src.utils.config import load_yaml, resolve_project_path
from src.utils.logging_config import setup_logging

logger = setup_logging()


def _parse_yolo_label(path: Path) -> list[tuple[int, list[float]]]:
    """Parse a YOLO segmentation label file. Returns list of (class_id, coords)."""
    instances: list[tuple[int, list[float]]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return instances
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid label line in {path}: {line!r}")
        cls_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        instances.append((cls_id, coords))
    return instances


def validate_label_classes(label_path: Path, valid_classes: set[int]) -> None:
    """Raise ValueError if any class index in the label is not in valid_classes."""
    for cls_id, _ in _parse_yolo_label(label_path):
        if cls_id not in valid_classes:
            raise ValueError(
                f"Class index {cls_id} in {label_path} not in dataset config "
                f"({sorted(valid_classes)}). Fix labels before training."
            )


def find_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> list[tuple[Path, Path]]:
    """Return (image, label) pairs; raise if an image has no label."""
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in extensions:
            continue
        rel = img_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")
        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for image {img_path} — expected {label_path}"
            )
        pairs.append((img_path, label_path))
    return pairs


def validate_source_dataset(
    images_dir: Path,
    labels_dir: Path,
    valid_classes: set[int],
) -> list[tuple[Path, Path]]:
    """Validate all pairs in a source dataset directory."""
    pairs = find_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        raise ValueError(f"No image/label pairs found under {images_dir}")
    for _, label_path in pairs:
        validate_label_classes(label_path, valid_classes)
    logger.info("Validated %d pairs in %s", len(pairs), images_dir)
    return pairs


def copy_pairs(
    pairs: Iterable[tuple[Path, Path]],
    dest_images: Path,
    dest_labels: Path,
    prefix: str = "",
) -> int:
    """Copy validated pairs into destination train/val/test folders."""
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path, label_path in pairs:
        stem = f"{prefix}{img_path.stem}" if prefix else img_path.stem
        out_img = dest_images / f"{stem}{img_path.suffix.lower()}"
        out_lbl = dest_labels / f"{stem}.txt"
        shutil.copy2(img_path, out_img)
        shutil.copy2(label_path, out_lbl)
        count += 1
    return count


def prepare_dataset(
    dataset_config_path: Path,
    sources: list[dict],
    output_root: Path,
    project_root: Path,
) -> None:
    """
    Merge multiple source datasets into data/yolo_dataset/.

    Each source dict: {"images": "path/to/images", "labels": "path/to/labels",
    "split": "train"|"val"|"test", "prefix": "optional_"}

    Fails loudly on missing labels or class index mismatches.
    """
    ds_cfg = load_yaml(dataset_config_path)
    names = ds_cfg.get("names", {})
    valid_classes = set(int(k) for k in names.keys())

    for split in ("train", "val", "test"):
        split_img_key = ds_cfg.get(split, f"images/{split}")
        # YOLO dataset.yaml uses relative paths like images/train
        dest_images = output_root / split_img_key
        dest_labels = output_root / "labels" / split
        if dest_images.exists():
            shutil.rmtree(dest_images)
        if dest_labels.exists():
            shutil.rmtree(dest_labels)

    total = 0
    for src in sources:
        split = src["split"]
        images_dir = resolve_project_path(src["images"], project_root)
        labels_dir = resolve_project_path(src["labels"], project_root)
        prefix = src.get("prefix", "")

        pairs = validate_source_dataset(images_dir, labels_dir, valid_classes)

        split_key = ds_cfg.get(split, f"images/{split}")
        dest_images = output_root / split_key
        dest_labels = output_root / "labels" / split
        n = copy_pairs(pairs, dest_images, dest_labels, prefix=prefix)
        total += n
        logger.info("Copied %d pairs -> %s", n, split)

    logger.info("Dataset preparation complete: %d total images -> %s", total, output_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from source folders")
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
        help="YOLO dataset config with class names",
    )
    parser.add_argument(
        "--output",
        default="data/yolo_dataset",
        help="Output directory for merged YOLO dataset",
    )
    parser.add_argument(
        "--source",
        action="append",
        nargs=4,
        metavar=("IMAGES", "LABELS", "SPLIT", "PREFIX"),
        help="Source: images_dir labels_dir split(train|val|test) prefix",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    dataset_config_path = resolve_project_path(args.dataset_config, project_root)
    output_root = resolve_project_path(args.output, project_root)

    if not args.source:
        logger.error(
            "No --source specified. Example:\n"
            "  python -m src.training.prepare_dataset \\\n"
            "    --source data/external/cwfid/images data/external/cwfid/labels train cwfid_ \\\n"
            "    --source data/raw/labeled/images data/raw/labeled/labels train phone_"
        )
        return 1

    sources = [
        {"images": s[0], "labels": s[1], "split": s[2], "prefix": s[3]}
        for s in args.source
    ]

    try:
        prepare_dataset(dataset_config_path, sources, output_root, project_root)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Dataset validation failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
