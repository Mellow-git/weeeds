"""Shared logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """
    Configure root logging with a consistent format and return a logger.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR).
        name: Optional logger name; defaults to root logger behavior.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(name)
