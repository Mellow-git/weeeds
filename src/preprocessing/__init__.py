"""Image preprocessing: letterbox, CLAHE, vegetation indices."""

from src.preprocessing.preprocessing import (
    PreprocessResult,
    compute_exg,
    compute_exg_exr,
    letterbox,
    normalize_lighting,
    preprocess_frame,
    unletterbox_point,
    vegetation_mask,
)

__all__ = [
    "PreprocessResult",
    "compute_exg",
    "compute_exg_exr",
    "letterbox",
    "normalize_lighting",
    "preprocess_frame",
    "unletterbox_point",
    "vegetation_mask",
]
