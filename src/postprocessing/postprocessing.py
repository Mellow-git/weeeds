"""
Postprocessing between model output and usable detections.

Parses raw mask/box/class arrays, filters by confidence and area, tracks
centroids across frames, and exports structured results.

Nothing here imports ultralytics except the isolated adapter
``from_ultralytics_result``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """Single instance segmentation detection in original frame coordinates."""

    class_id: int
    class_name: str
    confidence: float
    centroid_xy: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]
    mask_area_px: int
    track_id: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["centroid_xy"] = list(d["centroid_xy"])
        d["bbox_xyxy"] = list(d["bbox_xyxy"])
        return d


# ---------------------------------------------------------------------------
# Adapter: raw model outputs -> Detection objects
# ---------------------------------------------------------------------------


def masks_to_detections(
    masks: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    class_names: dict[int, str],
    scale: float,
    pad: tuple[int, int],
) -> list[Detection]:
    """
    Convert raw per-instance arrays into Detection objects.

    Centroids and boxes are mapped back into original frame coordinates.
    """
    detections: list[Detection] = []
    pad_x, pad_y = pad

    n = masks.shape[0] if masks is not None else 0
    for i in range(n):
        mask = masks[i]
        area = int(mask.sum())
        if area == 0:
            continue

        ys, xs = np.nonzero(mask)
        cx_model = float(xs.mean())
        cy_model = float(ys.mean())

        cx = (cx_model - pad_x) / scale
        cy = (cy_model - pad_y) / scale

        x1, y1, x2, y2 = boxes_xyxy[i]
        bbox_orig = (
            (x1 - pad_x) / scale,
            (y1 - pad_y) / scale,
            (x2 - pad_x) / scale,
            (y2 - pad_y) / scale,
        )

        cls_id = int(classes[i])
        detections.append(
            Detection(
                class_id=cls_id,
                class_name=class_names.get(cls_id, f"class_{cls_id}"),
                confidence=float(scores[i]),
                centroid_xy=(cx, cy),
                bbox_xyxy=bbox_orig,
                mask_area_px=area,
            )
        )
    return detections


def from_ultralytics_result(
    result,
    class_names: dict[int, str],
    scale: float,
    pad: tuple[int, int],
) -> list[Detection]:
    """
    Thin adapter for an Ultralytics ``Results`` object.

    This is the ONLY function in this module that touches the ultralytics API.
    """
    if result.masks is None:
        return []
    masks = result.masks.data.cpu().numpy().astype(np.uint8)
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    return masks_to_detections(masks, boxes, classes, scores, class_names, scale, pad)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_by_confidence(detections: list[Detection], min_conf: float = 0.4) -> list[Detection]:
    return [d for d in detections if d.confidence >= min_conf]


def filter_by_area(
    detections: list[Detection],
    min_area_px: int = 50,
    max_area_px: Optional[int] = None,
) -> list[Detection]:
    out = []
    for d in detections:
        if d.mask_area_px < min_area_px:
            continue
        if max_area_px is not None and d.mask_area_px > max_area_px:
            continue
        out.append(d)
    return out


def filter_detections(
    detections: list[Detection],
    min_conf: float = 0.4,
    min_area_px: int = 50,
    max_area_px: Optional[int] = None,
) -> list[Detection]:
    """Chain confidence and area filters."""
    d = filter_by_confidence(detections, min_conf)
    return filter_by_area(d, min_area_px, max_area_px)


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------


class CentroidTracker:
    """
    Minimal nearest-neighbour tracker with exponential moving average smoothing.

    Intentionally simple — good enough for a slow-moving demo rig. Swap in
    ByteTrack (via ``model.track()``) for fast motion or many overlapping
    instances.

    Known limitation: greedy nearest-neighbour matching can misassign track
    IDs when two same-class detections are closer than ``max_match_dist_px``
    to each other's previous positions (e.g. two weeds side by side). A
    global assignment (Hungarian algorithm) or a proper multi-object tracker
    would fix this but is out of scope for the current prototype.
    """

    def __init__(
        self,
        max_match_dist_px: float = 40.0,
        smoothing_alpha: float = 0.5,
        max_missed_frames: int = 5,
    ) -> None:
        self.max_match_dist_px = max_match_dist_px
        self.alpha = smoothing_alpha
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched_track_ids = set(self._tracks.keys())
        smoothed: list[Detection] = []

        for det in detections:
            best_id, best_dist = None, self.max_match_dist_px
            for tid in unmatched_track_ids:
                track = self._tracks[tid]
                if track["class_id"] != det.class_id:
                    continue
                dist = _euclidean(track["centroid"], det.centroid_xy)
                if dist < best_dist:
                    best_id, best_dist = tid, dist

            if best_id is not None:
                track = self._tracks[best_id]
                sx = self.alpha * det.centroid_xy[0] + (1 - self.alpha) * track["centroid"][0]
                sy = self.alpha * det.centroid_xy[1] + (1 - self.alpha) * track["centroid"][1]
                track["centroid"] = (sx, sy)
                track["missed"] = 0
                unmatched_track_ids.discard(best_id)
                det.centroid_xy = (sx, sy)
                det.track_id = best_id
            else:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = {
                    "centroid": det.centroid_xy,
                    "class_id": det.class_id,
                    "missed": 0,
                }
                det.track_id = new_id

            smoothed.append(det)

        for tid in list(unmatched_track_ids):
            self._tracks[tid]["missed"] += 1
            if self._tracks[tid]["missed"] > self.max_missed_frames:
                del self._tracks[tid]

        return smoothed


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "crop": (60, 200, 60),
    "weed": (40, 40, 220),
}


def draw_overlay(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bbox, centroid, and label for each detection. Returns a new frame."""
    out = frame_bgr.copy()
    for d in detections:
        color = CLASS_COLORS.get(d.class_name, (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
        cx, cy = [int(v) for v in d.centroid_xy]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (cx, cy), 5, color, -1)

        label = f"{d.class_name} {d.confidence:.2f}"
        if d.track_id is not None:
            label += f" #{d.track_id}"
        cv2.putText(
            out, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_frame_result(
    detections: list[Detection],
    frame_index: int,
    timestamp: Optional[float] = None,
) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp": timestamp if timestamp is not None else time.time(),
        "detections": [d.to_dict() for d in detections],
    }


def export_to_json(detections: list[Detection], frame_index: int, path: str) -> None:
    payload = export_frame_result(detections, frame_index)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
