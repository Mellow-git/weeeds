"""Tests for preprocessing module."""

import numpy as np
import pytest

from src.preprocessing.preprocessing import (
    compute_exg,
    letterbox,
    preprocess_frame,
    unletterbox_point,
)


class TestPreprocessFrame:
    """Port of the original __main__ self-test."""

    def test_preprocess_frame_synthetic(self):
        synthetic = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocess_frame(synthetic)
        assert result.model_input.shape == (640, 640, 3)
        assert result.exg_map.shape == (480, 640)
        assert result.exg_map.min() >= -2.0
        assert result.exg_map.max() <= 2.0
        assert result.scale > 0
        assert len(result.pad) == 2


class TestLetterboxRoundTrip:
    """Verify coordinate math for all four aspect-ratio / padding cases."""

    TARGET = 640

    @pytest.mark.parametrize(
        "h,w,expected_pad_x,expected_pad_y",
        [
            # Wider-than-tall (landscape): vertical padding
            (480, 640, 0, 80),
            # Taller-than-wide (portrait): horizontal padding
            (640, 480, 80, 0),
            # Square: no padding
            (640, 640, 0, 0),
            # Wide 16:9-ish: vertical padding only
            (600, 800, 0, 40),
            # Tall 9:16-ish: horizontal padding only
            (800, 600, 80, 0),
        ],
    )
    def test_letterbox_padding(self, h, w, expected_pad_x, expected_pad_y):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        _, scale, (pad_x, pad_y) = letterbox(frame, target_size=self.TARGET)
        assert pad_x == expected_pad_x
        assert pad_y == expected_pad_y
        assert scale == pytest.approx(min(self.TARGET / h, self.TARGET / w), rel=1e-3)

    @pytest.mark.parametrize(
        "h,w",
        [(480, 640), (640, 480), (800, 600), (600, 800)],
    )
    def test_unletterbox_center_roundtrip(self, h, w):
        """Center of letterboxed content maps back to center of original frame."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        _, scale, pad = letterbox(frame, target_size=self.TARGET)
        pad_x, pad_y = pad
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        cx_model = pad_x + new_w / 2.0
        cy_model = pad_y + new_h / 2.0
        cx_orig, cy_orig = unletterbox_point(cx_model, cy_model, scale, pad)
        assert cx_orig == pytest.approx(w / 2.0, abs=1.0)
        assert cy_orig == pytest.approx(h / 2.0, abs=1.0)

    def test_letterbox_output_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out, _, _ = letterbox(frame, target_size=640)
        assert out.shape == (640, 640, 3)


class TestVegetationIndex:
    def test_exg_green_pixel_positive(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 1] = 255
        exg = compute_exg(frame)
        assert exg.mean() > 0
