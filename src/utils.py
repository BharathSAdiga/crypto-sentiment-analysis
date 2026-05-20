"""Shared utilities for paths, logging, and filesystem setup."""

from __future__ import annotations

import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent console logging for scripts and modules."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories() -> None:
    """Create project output directories if they do not already exist."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, CHARTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def relative_to_project(path: Path) -> str:
    """Return a readable path relative to the project root when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
