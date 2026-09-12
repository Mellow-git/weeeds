"""Live inference pipeline: capture -> preprocess -> model -> postprocess."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.capture.frame_source import FrameSource, FrameSourceConfig
from src.postprocessing.postprocessing import (
    CentroidTracker,
    draw_overlay,
    export_to_json,
    filter_detections,
    from_ultralytics_result,
)
from src.preprocessing.preprocessing import preprocess_frame
from src.utils.config import load_yaml, resolve_project_path
from src.utils.logging_config import setup_logging

logger = setup_logging()


def _blend_exg_debug_panel(frame_bgr: np.ndarray, exg_map: np.ndarray, panel_scale: float = 0.25) -> np.ndarray:
    """
    Blend a small ExG heatmap in the bottom-right corner for debug visualization.

    ExG is auxiliary only — not fed to the model.
    """
    out = frame_bgr.copy()
    exg_norm = np.clip((exg_map + 1.0) / 3.0, 0.0, 1.0)
    exg_u8 = (exg_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(exg_u8, cv2.COLORMAP_VIRIDIS)

    h, w = frame_bgr.shape[:2]
    panel_h = max(1, int(h * panel_scale))
    panel_w = max(1, int(w * panel_scale))
    panel = cv2.resize(heatmap, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

    y0 = h - panel_h - 10
    x0 = w - panel_w - 10
    out[y0 : y0 + panel_h, x0 : x0 + panel_w] = cv2.addWeighted(
        out[y0 : y0 + panel_h, x0 : x0 + panel_w],
        0.35,
        panel,
        0.65,
        0,
    )
    cv2.putText(out, "ExG", (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


class LivePipeline:
    """Orchestrates the full real-time crop/weed detection loop."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self._running = True

        fs_cfg = config["frame_source"]
        source = fs_cfg["source"]
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.frame_source = FrameSource(
            FrameSourceConfig(
                source=source,
                reconnect_delay_s=fs_cfg.get("reconnect_delay_s", 1.0),
                max_reconnect_attempts=fs_cfg.get("max_reconnect_attempts", 10),
                read_timeout_s=fs_cfg.get("read_timeout_s", 2.0),
            )
        )

        self.tracker = CentroidTracker(
            max_match_dist_px=config["tracking"]["max_match_dist_px"],
            smoothing_alpha=config["tracking"]["smoothing_alpha"],
            max_missed_frames=config["tracking"]["max_missed_frames"],
        )

        self.class_names = {int(k): v for k, v in config["class_names"].items()}
        self.model = self._load_model(config, project_root)

        export_cfg = config.get("export", {})
        if export_cfg.get("enabled"):
            out_dir = resolve_project_path(export_cfg["output_dir"], project_root)
            out_dir.mkdir(parents=True, exist_ok=True)
            self.export_dir = out_dir
        else:
            self.export_dir = None

        self.max_consecutive_failures = fs_cfg.get("max_consecutive_failures", 30)
        self._null_frame_streak = 0

    def _load_model(self, config: dict[str, Any], project_root: Path):
        """Load YOLO weights from config; does not auto-download."""
        from ultralytics import YOLO

        model_cfg = config["model"]
        weights = resolve_project_path(model_cfg["weights"], project_root)
        fallback = resolve_project_path(model_cfg["pretrained_fallback"], project_root)

        path = weights if weights.exists() else fallback
        if not path.exists():
            raise FileNotFoundError(
                f"No model weights found at {weights} or {fallback}. "
                "Place yolov8n-seg.pt in models/pretrained/ or train a model first."
            )

        logger.info("Loading model from %s", path)
        return YOLO(str(path))

    def _handle_sigint(self, signum, frame) -> None:
        logger.info("Shutdown requested (signal %s)", signum)
        self._running = False

    def run(self) -> None:
        """Run the main capture-inference loop until interrupted."""
        signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

        pp_cfg = self.config["preprocessing"]
        po_cfg = self.config["postprocessing"]
        model_cfg = self.config["model"]
        display_cfg = self.config.get("display", {})
        export_cfg = self.config.get("export", {})

        pad_color = tuple(pp_cfg.get("pad_color", [114, 114, 114]))
        clahe_grid = tuple(pp_cfg.get("clahe_tile_grid_size", [8, 8]))
        max_area = po_cfg.get("max_area_px")

        frame_index = 0
        logger.info("Starting live pipeline loop")

        try:
            while self._running:
                frame = self.frame_source.read()
                if frame is None:
                    self._null_frame_streak += 1
                    if self._null_frame_streak >= self.max_consecutive_failures:
                        logger.error(
                            "No frames received for %d consecutive cycles — "
                            "check camera/stream URL (%s) and network connection",
                            self._null_frame_streak,
                            self.config["frame_source"]["source"],
                        )
                        self._null_frame_streak = 0
                    time.sleep(0.05)
                    continue

                self._null_frame_streak = 0

                result = preprocess_frame(
                    frame,
                    target_size=pp_cfg["target_size"],
                    apply_clahe=pp_cfg.get("apply_clahe", True),
                    clahe_clip_limit=pp_cfg.get("clahe_clip_limit", 2.0),
                    clahe_tile_grid_size=clahe_grid,
                    pad_color=pad_color,
                )

                predictions = self.model.predict(
                    result.model_input,
                    conf=model_cfg.get("conf", 0.25),
                    device=model_cfg.get("device", "0"),
                    verbose=False,
                )

                detections = from_ultralytics_result(
                    predictions[0],
                    self.class_names,
                    result.scale,
                    result.pad,
                )
                detections = filter_detections(
                    detections,
                    min_conf=po_cfg["min_conf"],
                    min_area_px=po_cfg["min_area_px"],
                    max_area_px=max_area,
                )
                detections = self.tracker.update(detections)

                overlay = draw_overlay(result.original, detections)
                if display_cfg.get("show_exg_overlay", False):
                    overlay = _blend_exg_debug_panel(overlay, result.exg_map)

                if self.export_dir and frame_index % export_cfg.get("json_every_n_frames", 30) == 0:
                    out_path = self.export_dir / f"frame_{frame_index:06d}.json"
                    export_to_json(detections, frame_index, str(out_path))

                if display_cfg.get("show_window", True):
                    cv2.imshow(display_cfg.get("window_name", "Crop/Weed Detection"), overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit key pressed")
                        break

                frame_index += 1

        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Release camera and destroy OpenCV windows."""
        logger.info("Shutting down pipeline")
        self.frame_source.release()
        cv2.destroyAllWindows()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run live crop/weed detection pipeline")
    parser.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Path to pipeline YAML config",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    config_path = resolve_project_path(args.config, project_root)
    config = load_yaml(config_path)

    log_level = config.get("logging", {}).get("level", "INFO")
    global logger
    logger = setup_logging(log_level)

    try:
        pipeline = LivePipeline(config, project_root)
        pipeline.run()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
