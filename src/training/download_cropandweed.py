"""
Download CropAndWeed images/annotations and map them to the CropOrWeed2 variant.

Run: python -m src.training.download_cropandweed --output data/external/CropAndWeed/data

This wraps the official dataset tar downloads (see cropandweed-dataset/cnw/setup.py)
and writes mapped bboxes/labelIds under CropOrWeed2 without requiring a git clone.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from src.training.cropandweed_mapping import (
    BACKGROUND_CLASS_ID,
    map_source_label,
)
from src.utils.config import resolve_project_path
from src.utils.logging_config import setup_logging

logger = setup_logging()

DOWNLOAD_BASE = "https://vitro-testing.com/wp-content/uploads/2022/12"
ARCHIVES = [
    "cropandweed_annotations",
    "cropandweed_images1of4",
    "cropandweed_images2of4",
    "cropandweed_images3of4",
    "cropandweed_images4of4",
]


def _download_and_extract(url: str, dest: Path) -> None:
    """Stream-download a tar archive and extract into dest."""
    logger.info("Downloading %s", url)
    dest.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        with tarfile.open(fileobj=response, mode="r|") as archive:
            archive.extractall(dest)


def _normalize_layout(data_root: Path) -> None:
    """
    Match the layout produced by the official setup.py script.

    After extraction, bboxes live at data_root/bboxes/CropAndWeed/*.csv and
    labelIds at data_root/labelIds/CropAndWeed/*.png.
    """
    loose_bboxes = data_root / "bboxes"
    nested = data_root / "CropAndWeed"
    if nested.is_dir() and not (loose_bboxes / "CropAndWeed").is_dir():
        target = loose_bboxes / "CropAndWeed"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(nested), str(target))

    images_parent = data_root.parent / "images"
    if images_parent.is_dir() and not (data_root / "images").is_dir():
        shutil.move(str(images_parent), str(data_root / "images"))


def map_bboxes(source_dir: Path, target_dir: Path) -> int:
    """Map CropAndWeed CSV bboxes to CropOrWeed2 (training set — no Vegetation fallback)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    mapped_files = 0

    for csv_path in sorted(source_dir.glob("*.csv")):
        rows: list[dict[str, str]] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(
                handle,
                fieldnames=["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"],
            )
            for row in reader:
                mapped = map_source_label(int(row["label_id"]))
                if mapped is None:
                    continue
                row["label_id"] = str(mapped)
                rows.append(row)

        if not rows:
            continue

        out_path = target_dir / csv_path.name
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
            writer.writerows(rows)
        mapped_files += 1

    logger.info("Mapped %d bbox files -> %s", mapped_files, target_dir)
    return mapped_files


def map_label_ids(source_dir: Path, target_dir: Path) -> int:
    """Map CropAndWeed semantic masks to CropOrWeed2."""
    target_dir.mkdir(parents=True, exist_ok=True)
    mapped_files = 0

    for mask_path in sorted(source_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning("Skipping unreadable mask: %s", mask_path)
            continue

        output = np.full_like(mask, BACKGROUND_CLASS_ID)
        include = False
        for source_id in np.unique(mask):
            mapped = map_source_label(int(source_id))
            if mapped is not None:
                output[mask == source_id] = mapped
                include = True
            else:
                output[mask == source_id] = BACKGROUND_CLASS_ID

        if not include:
            continue

        cv2.imwrite(str(target_dir / mask_path.name), output)
        mapped_files += 1

    logger.info("Mapped %d labelId masks -> %s", mapped_files, target_dir)
    return mapped_files


def download_cropandweed(output_root: Path, skip_download: bool = False) -> Path:
    """
    Download CropAndWeed and map annotations to CropOrWeed2.

    Returns the data root containing images/, bboxes/CropOrWeed2/, labelIds/CropOrWeed2/.
    """
    data_root = output_root
    data_root.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        for name in ARCHIVES:
            url = f"{DOWNLOAD_BASE}/{name}.tar"
            _download_and_extract(url, data_root)
        _normalize_layout(data_root)
    else:
        logger.info("Skipping download; using existing data at %s", data_root)

    bboxes_src = data_root / "bboxes" / "CropAndWeed"
    labelids_src = data_root / "labelIds" / "CropAndWeed"
    if not bboxes_src.is_dir() or not labelids_src.is_dir():
        raise FileNotFoundError(
            f"Expected CropAndWeed annotations under {bboxes_src} and {labelids_src}. "
            "Re-run without --skip-download."
        )

    map_bboxes(bboxes_src, data_root / "bboxes" / "CropOrWeed2")
    map_label_ids(labelids_src, data_root / "labelIds" / "CropOrWeed2")

    images_dir = data_root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    logger.info("CropAndWeed ready at %s (%d images)", data_root, len(list(images_dir.glob("*.jpg"))))
    return data_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and map CropAndWeed to CropOrWeed2")
    parser.add_argument(
        "--output",
        default="data/external/CropAndWeed/data",
        help="Directory for downloaded images and mapped annotations",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip tar download; only (re)map existing extracted data",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    output_root = resolve_project_path(args.output, project_root)

    try:
        download_cropandweed(output_root, skip_download=args.skip_download)
    except (FileNotFoundError, tarfile.TarError, OSError) as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
