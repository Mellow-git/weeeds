"""Tests for postprocessing module."""

import numpy as np

from src.postprocessing.postprocessing import (
    CentroidTracker,
    filter_detections,
    masks_to_detections,
)


def _synthetic_detections():
    """Same synthetic setup as the original __main__ self-test."""
    H, W = 640, 640
    masks = np.zeros((2, H, W), dtype=np.uint8)
    masks[0, 100:150, 100:150] = 1
    masks[1, 300:340, 400:460] = 1

    boxes = np.array(
        [[100, 100, 150, 150], [400, 300, 460, 340]],
        dtype=np.float32,
    )
    classes = np.array([0, 1])
    scores = np.array([0.91, 0.77])
    class_names = {0: "crop", 1: "weed"}
    scale = 640 / 480
    pad = (80, 0)
    return masks, boxes, classes, scores, class_names, scale, pad


class TestMasksToDetections:
    def test_two_instances(self):
        masks, boxes, classes, scores, class_names, scale, pad = _synthetic_detections()
        dets = masks_to_detections(masks, boxes, classes, scores, class_names, scale, pad)
        assert len(dets) == 2
        assert dets[0].class_name == "crop"
        assert dets[1].class_name == "weed"
        assert dets[0].mask_area_px == 50 * 50

    def test_empty_masks(self):
        masks = np.zeros((0, 640, 640), dtype=np.uint8)
        boxes = np.zeros((0, 4), dtype=np.float32)
        dets = masks_to_detections(
            masks, boxes, np.array([]), np.array([]), {0: "crop"}, 1.0, (0, 0)
        )
        assert dets == []


class TestFiltering:
    def test_filter_confidence_and_area(self):
        masks, boxes, classes, scores, class_names, scale, pad = _synthetic_detections()
        dets = masks_to_detections(masks, boxes, classes, scores, class_names, scale, pad)
        filtered = filter_detections(dets, min_conf=0.5, min_area_px=100)
        assert len(filtered) == 2

        filtered_strict = filter_detections(dets, min_conf=0.95, min_area_px=100)
        assert len(filtered_strict) == 0


class TestCentroidTracker:
    def test_tracking_smoothing_across_frames(self):
        masks, boxes, classes, scores, class_names, scale, pad = _synthetic_detections()
        dets = masks_to_detections(masks, boxes, classes, scores, class_names, scale, pad)
        filtered = filter_detections(dets, min_conf=0.5, min_area_px=100)

        tracker = CentroidTracker(smoothing_alpha=0.5)
        tracked1 = tracker.update(filtered)
        assert all(d.track_id is not None for d in tracked1)
        ids_frame1 = {d.track_id for d in tracked1}

        H, W = 640, 640
        masks2 = np.zeros((2, H, W), dtype=np.uint8)
        masks2[0, 102:152, 98:148] = 1
        masks2[1, 298:338, 402:462] = 1
        dets2 = masks_to_detections(masks2, boxes, classes, scores, class_names, scale, pad)
        tracked2 = tracker.update(filter_detections(dets2, min_conf=0.5, min_area_px=100))

        ids_frame2 = {d.track_id for d in tracked2}
        assert ids_frame1 == ids_frame2
        assert len(tracked2) == 2
