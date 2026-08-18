"""
Frame acquisition from phone WiFi stream or USB camera.

FrameSource wraps cv2.VideoCapture with reconnect logic, because phone
streams over WiFi drop far more often than a plugged-in USB camera.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameSourceConfig:
    """Configuration for a video frame source."""

    source: Union[str, int]  # e.g. "http://192.168.1.23:8080/video" or 0 for USB cam
    reconnect_delay_s: float = 1.0
    max_reconnect_attempts: int = 10
    read_timeout_s: float = 2.0


class FrameSource:
    """
    Wraps cv2.VideoCapture with reconnect logic for a phone stream.

    Phone-over-WiFi streams (IP Webcam etc.) drop far more often than a
    plugged-in USB camera. A bare VideoCapture loop will just start
    returning ret=False forever once the connection drops — this class
    detects that and attempts to reopen the stream instead of silently
    feeding stale/garbage frames into the model.
    """

    def __init__(self, config: FrameSourceConfig) -> None:
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0
        self._connect()

    def _connect(self) -> bool:
        if self._cap is not None:
            self._cap.release()
        logger.info("Connecting to frame source: %s", self.config.source)
        self._cap = cv2.VideoCapture(self.config.source)
        # Keep the internal buffer small so we get the *latest* frame,
        # not a queued-up backlog of stale ones (matters a lot for a
        # laggy WiFi stream feeding a real-time loop).
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = self._cap.isOpened()
        if not ok:
            logger.warning("Failed to open frame source: %s", self.config.source)
        return ok

    def read(self) -> Optional[np.ndarray]:
        """
        Return a BGR frame, or None if the source is currently unreachable.

        Caller should treat None as "skip this cycle", not "crash".
        """
        if self._cap is None or not self._cap.isOpened():
            if not self._reconnect():
                return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._consecutive_failures += 1
            logger.warning(
                "Frame read failed (%d consecutive failures)",
                self._consecutive_failures,
            )
            if self._consecutive_failures >= 3:
                self._reconnect()
            return None

        self._consecutive_failures = 0
        return frame

    def _reconnect(self) -> bool:
        for attempt in range(1, self.config.max_reconnect_attempts + 1):
            logger.info(
                "Reconnect attempt %d/%d",
                attempt,
                self.config.max_reconnect_attempts,
            )
            if self._connect():
                self._consecutive_failures = 0
                return True
            time.sleep(self.config.reconnect_delay_s)
        logger.error("Exhausted reconnect attempts for %s", self.config.source)
        return False

    def release(self) -> None:
        """Release the underlying VideoCapture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
