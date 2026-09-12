"""
Convert mapped CropOrWeed2 annotations to YOLO segmentation labels.

Reads bbox CSVs + semantic masks, extracts instance polygons, and writes a
flat image/label pool suitable for random splitting.

Run after download_cropandweed.py:
  python -m src.training.convert_cropandweed
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import resolve_project_path
from src.utils.logging_config import setup_logging

logger = setup_logging()

BBOX_FIELDS = ["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"]


def _bbox_fallback_polygon(
    left: int, top: int, right: int, bottom: int, img_w: int, img_h: int
) -> list[tuple[float, float]]:
    """Rectangle polygon normalized to [0, 1] as a fallback when mask contour fails."""
    xs = [left, right, right, left]
    ys = [top, bottom, bottom, top]
    return [(x / img_w, y / img_h) for x, y in zip(xs, ys)]


def _mask_instance_polygon(
    mask: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    class_id: int,
    img_w: int,
    img_h: int,
    epsilon_ratio: float = 0.002,
) -> list[tuple[float, float]] | None:
    """Extract the largest contour for class_id within the bbox ROI."""
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)
    if right <= left or bottom <= top:
        return None

    roi = mask[top:bottom, left:right]
    binary = (roi == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None

    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None

    points: list[tuple[float, float]] = []
    for pt in approx.reshape(-1, 2):
        x = (left + float(pt[0])) / img_w
        y = (top + float(pt[1])) / img_h
        points.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)))
    return points


def _points_to_yolo_line(class_id: int, points: list[tuple[float, float]]) -> str:
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{class_id} {coords}"


def convert_image(
    image_path: Path,
    bbox_path: Path,
    mask_path: Path,
    output_images: Path,
    output_labels: Path,
) -> int:
    """Convert one CropOrWeed2 image and return the number of instances written."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    img_h, img_w = image.shape[:2]
    lines: list[str] = []

    with bbox_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, fieldnames=BBOX_FIELDS)
        for row in reader:
            class_id = int(row["label_id"])
            left, top, right, bottom = (int(float(row[k])) for k in ("left", "top", "right", "bottom"))

            points = _mask_instance_polygon(mask, left, top, right, bottom, class_id, img_w, img_h)
            if points is None:
                points = _bbox_fallback_polygon(left, top, right, bottom, img_w, img_h)

            lines.append(_points_to_yolo_line(class_id, points))

    stem = image_path.stem
    out_img = output_images / f"{stem}{image_path.suffix.lower()}"
    out_lbl = output_labels / f"{stem}.txt"

    cv2.imwrite(str(out_img), image)
    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def convert_cropandweed(
    data_root: Path,
    output_pool: Path,
    variant: str = "CropOrWeed2",
) -> int:
    """
    Convert all mapped CropAndWeed images to YOLO segmentation format.

    Writes flat pool/images/ and pool/labels/ (no train/val split yet).
    """
    images_dir = data_root / "images"
    bboxes_dir = data_root / "bboxes" / variant
    masks_dir = data_root / "labelIds" / variant

    for path, label in (
        (images_dir, "images"),
        (bboxes_dir, f"bboxes/{variant}"),
        (masks_dir, f"labelIds/{variant}"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label} directory: {path}")

    out_images = output_pool / "images"
    out_labels = output_pool / "labels"
    if output_pool.exists():
        for sub in (out_images, out_labels):
            if sub.exists():
                for child in sub.iterdir():
                    if child.is_file():
                        child.unlink()

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for image_path in sorted(images_dir.glob("*.jpg")):
        bbox_path = bboxes_dir / f"{image_path.stem}.csv"
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not bbox_path.exists() or not mask_path.exists():
            skipped += 1
            continue

        convert_image(image_path, bbox_path, mask_path, out_images, out_labels)
        converted += 1

    logger.info(
        "Converted %d images to YOLO pool at %s (%d skipped — missing bbox/mask)",
        converted,
        output_pool,
        skipped,
    )
    return converted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert CropOrWeed2 to YOLO segmentation pool")
    parser.add_argument(
        "--input",
        default="data/external/CropAndWeed/data",
        help="CropAndWeed data root (images + mapped CropOrWeed2 annotations)",
    )
    parser.add_argument(
        "--output",
        default="data/external/cropandweed_yolo/pool",
        help="Output directory for flat image/label pool",
    )
    parser.add_argument(
        "--variant",
        default="CropOrWeed2",
        help="Mapped dataset variant subdirectory name",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    data_root = resolve_project_path(args.input, project_root)
    output_pool = resolve_project_path(args.output, project_root)

    try:
        convert_cropandweed(data_root, output_pool, variant=args.variant)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
