"""
dataScrapper.py
===============
Backward-compatible entry point that delegates to the modular WhoScored scraping pipeline:
- `scraper.py`  (`WhoScoredScraper`)   : Fetches full `matchCentreData` JSON via Selenium.
- `parser.py`   (`MatchDataParser`)    : Parses ALL match events, player lineups, and match metadata.
- `exporter.py` (`MatchDataExporter`)  : Exports structured CSV files + raw JSON backups to `output/`.
- `main.py`     (`run_pipeline`)       : CLI & batch runner.

Run either:
    python main.py
or:
    python dataScrapper.py
"""

from __future__ import annotations

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
    # Replace with one or more WhoScored Match Centre Live URLs
    MATCH_URL = DEFAULT_MATCH_URL
    run_pipeline(urls=[MATCH_URL], output_dir="output", headless=False)
