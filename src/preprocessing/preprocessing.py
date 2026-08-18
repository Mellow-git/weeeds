"""
Preprocessing between frame capture and model inference.

Handles resizing/letterboxing, lighting normalization (CLAHE), and
vegetation-index computation (ExG, ExG-ExR) as a cheap stand-in for
the infrared/multispectral channel used in commercial systems.

All array-processing functions take/return plain numpy arrays so they're
independently testable without a live camera or a loaded model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Resizing / letterboxing
# ---------------------------------------------------------------------------


def letterbox(
    frame: np.ndarray,
    target_size: int = 640,
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize a frame to a square target_size x target_size canvas while
    preserving aspect ratio, padding the rest.

    Returns (letterboxed_frame, scale_ratio, (pad_x, pad_y)) so model
    output coordinates can be mapped back to the original frame.
    """
    h, w = frame.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, (pad_x, pad_y)


def unletterbox_point(
    x: float, y: float, scale: float, pad: tuple[int, int]
) -> tuple[float, float]:
    """Map an (x, y) point in letterboxed-frame coordinates back to the original frame."""
    pad_x, pad_y = pad
    orig_x = (x - pad_x) / scale
    orig_y = (y - pad_y) / scale
    return orig_x, orig_y


# ---------------------------------------------------------------------------
# Lighting normalization
# ---------------------------------------------------------------------------


def normalize_lighting(
    frame_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE on the L channel in LAB space.

    Helps flatten out auto-exposure variation between training and demo
    conditions.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_channel)

    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Vegetation index (stand-in for IR/multispectral channel)
# ---------------------------------------------------------------------------


def compute_exg(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Excess Green Index: ExG = 2G - R - B on normalized channels.

    Returns float32 H x W array, roughly in [-2, 2]. High values indicate
    likely vegetation. Does NOT distinguish crop vs weed.
    """
    frame_f = frame_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(frame_f)
    return 2 * g - r - b


def compute_exg_exr(frame_bgr: np.ndarray) -> np.ndarray:
    """ExG - ExR refinement that further suppresses soil/background reflectance."""
    frame_f = frame_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(frame_f)
    exg = 2 * g - r - b
    exr = 1.4 * r - g
    return exg - exr


def vegetation_mask(frame_bgr: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Binary uint8 mask (0/255) of likely-vegetation pixels using ExG."""
    exg = compute_exg(frame_bgr)
    return (exg > threshold).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@dataclass
class PreprocessResult:
    """Output of the full preprocessing chain for one frame."""

    model_input: np.ndarray  # letterboxed, lighting-normalized BGR frame
    original: np.ndarray  # untouched original frame
    scale: float  # for mapping model coords back to original
    pad: tuple[int, int]
    exg_map: np.ndarray  # auxiliary vegetation-index map (original resolution)


def preprocess_frame(
    frame_bgr: np.ndarray,
    target_size: int = 640,
    apply_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> PreprocessResult:
    """
    Full preprocessing pipeline for a single frame.

    Normalizes lighting on the original-resolution frame first, computes
    the vegetation index, then letterboxes for model input.
    """
    working = (
        normalize_lighting(
            frame_bgr,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )
        if apply_clahe
        else frame_bgr.copy()
    )
    exg_map = compute_exg(working)
    model_input, scale, pad = letterbox(working, target_size=target_size, pad_color=pad_color)

    return PreprocessResult(
        model_input=model_input,
        original=frame_bgr,
        scale=scale,
        pad=pad,
        exg_map=exg_map,
    )
