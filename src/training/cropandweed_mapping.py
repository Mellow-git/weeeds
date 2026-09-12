"""Label mappings for CropAndWeed -> binary crop/weed (CropOrWeed2 variant)."""

from __future__ import annotations

# Source label IDs mapped to crop (0) in the official CropOrWeed2 variant.
CROP_SOURCE_IDS: frozenset[int] = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 94, 24, 18, 13, 26, 27, 15}
)

# Source label IDs mapped to weed (1).
WEED_SOURCE_IDS: frozenset[int] = frozenset(
    {
        31, 48, 62, 65, 68, 69, 74, 75, 81, 84, 86, 32, 29, 33, 37, 49, 30, 44,
        66, 87, 89, 91, 61, 79, 34, 41, 52, 35, 36, 78, 38, 39, 71, 72, 88, 42, 45,
        70, 47, 51, 54, 58, 60, 80, 83, 96, 22, 63, 85, 56, 57, 64, 77, 50, 59, 67,
        76,
    }
)

# Matches configs/dataset.yaml: 0=crop, 1=weed
SOURCE_TO_BINARY: dict[int, int] = {
    **{label_id: 0 for label_id in CROP_SOURCE_IDS},
    **{label_id: 1 for label_id in WEED_SOURCE_IDS},
}

BINARY_CLASS_COUNT = 2
BACKGROUND_CLASS_ID = BINARY_CLASS_COUNT  # unmapped pixels in semantic masks


def map_source_label(source_label_id: int) -> int | None:
    """Return 0 (crop), 1 (weed), or None if the source label should be skipped."""
    return SOURCE_TO_BINARY.get(source_label_id)
