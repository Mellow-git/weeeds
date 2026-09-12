"""Tests for dataset splitting."""

from pathlib import Path

import pytest

from src.training.split_dataset import split_dataset


def _make_pool(root: Path, count: int) -> Path:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(count):
        (images / f"img_{i:03d}.jpg").write_bytes(b"fake")
        (labels / f"img_{i:03d}.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
    return root


class TestSplitDataset:
    def test_80_10_10_counts(self, tmp_path: Path):
        pool = _make_pool(tmp_path / "pool", 100)
        out = tmp_path / "yolo"
        counts = split_dataset(pool, out, seed=42)
        assert counts == {"train": 80, "val": 10, "test": 10}
        assert len(list((out / "images" / "train").glob("*.jpg"))) == 80
        assert len(list((out / "images" / "val").glob("*.jpg"))) == 10
        assert len(list((out / "images" / "test").glob("*.jpg"))) == 10

    def test_reproducible_with_seed(self, tmp_path: Path):
        pool = _make_pool(tmp_path / "pool", 20)
        out_a = tmp_path / "out_a"
        out_b = tmp_path / "out_b"
        split_dataset(pool, out_a, seed=7)
        split_dataset(pool, out_b, seed=7)
        train_a = sorted(p.name for p in (out_a / "images" / "train").glob("*.jpg"))
        train_b = sorted(p.name for p in (out_b / "images" / "train").glob("*.jpg"))
        assert train_a == train_b

    def test_ratios_must_sum_to_one(self, tmp_path: Path):
        pool = _make_pool(tmp_path / "pool", 10)
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_dataset(pool, tmp_path / "out", train_ratio=0.5, val_ratio=0.3, test_ratio=0.1)
