"""
Random train/val/test split for a flat YOLO image/label pool.

Default: 80/10/10 with a fixed seed for reproducibility.

Run after convert_cropandweed.py:
  python -m src.training.split_dataset --input data/external/cropandweed_yolo/pool
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from src.utils.config import resolve_project_path
from src.utils.logging_config import setup_logging

logger = setup_logging()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_pairs(pool_root: Path) -> list[tuple[Path, Path]]:
    """Return sorted (image, label) pairs from pool/images and pool/labels."""
    images_dir = pool_root / "images"
    labels_dir = pool_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected pool/images and pool/labels under {pool_root}"
        )

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {img_path.name}: {label_path}")
        pairs.append((img_path, label_path))

    if not pairs:
        raise ValueError(f"No image/label pairs found in {pool_root}")

    return pairs


def split_dataset(
    pool_root: Path,
    output_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """
    Copy pool pairs into data/yolo_dataset/images/{train,val,test} layout.

    Returns counts per split.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    pairs = _collect_pairs(pool_root)
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Assign remainder to test so every image is used exactly once.
    n_test = n - n_train - n_val

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    assert len(splits["test"]) == n_test

    for split in ("train", "val", "test"):
        img_dest = output_root / "images" / split
        lbl_dest = output_root / "labels" / split
        if img_dest.exists():
            shutil.rmtree(img_dest)
        if lbl_dest.exists():
            shutil.rmtree(lbl_dest)
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split, items in splits.items():
        for img_path, lbl_path in items:
            out_img = output_root / "images" / split / img_path.name
            out_lbl = output_root / "labels" / split / lbl_path.name
            shutil.copy2(img_path, out_img)
            shutil.copy2(lbl_path, out_lbl)
        counts[split] = len(items)
        logger.info("Split %-5s: %4d images (%.1f%%)", split, counts[split], 100 * counts[split] / n)

    logger.info(
        "Split complete: %d total images -> %s (seed=%d)",
        n,
        output_root,
        seed,
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Random 80/10/10 split for YOLO dataset pool")
    parser.add_argument(
        "--input",
        default="data/external/cropandweed_yolo/pool",
        help="Flat pool with images/ and labels/ subdirectories",
    )
    parser.add_argument(
        "--output",
        default="data/yolo_dataset",
        help="Output YOLO dataset root (images/train, labels/train, ...)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction")
    parser.add_argument("--val", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    pool_root = resolve_project_path(args.input, project_root)
    output_root = resolve_project_path(args.output, project_root)

    try:
        split_dataset(
            pool_root,
            output_root,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
