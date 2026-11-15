"""Shared paths and raw JSON file discovery for the test suite."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILES = sorted(RAW_DIR.glob("*_raw.json"))
