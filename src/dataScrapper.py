"""
dataScrapper.py
===============
Backward-compatible entry point that delegates to `src.main`.
Prefer running `python -m src.main` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exporter import ExportedFiles, MatchDataExporter
from src.main import DEFAULT_MATCH_URL, configure_logging, run_pipeline
from src.parser import MatchDataParser, ParsedMatchDataset
from src.scraper import WhoScoredScraper, scrape_whoscored_match_json

__all__ = [
    "WhoScoredScraper",
    "scrape_whoscored_match_json",
    "MatchDataParser",
    "ParsedMatchDataset",
    "MatchDataExporter",
    "ExportedFiles",
    "run_pipeline",
]

if __name__ == "__main__":
    configure_logging()
    run_pipeline(urls=[DEFAULT_MATCH_URL], output_dir="output", headless=False)
