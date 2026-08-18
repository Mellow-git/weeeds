"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as a dict."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_project_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path relative to the project root (parent of ``src/``)."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = Path(base) if base else Path(__file__).resolve().parents[2]
    return (root / p).resolve()
