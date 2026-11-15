"""Football Match Analytics package."""

from __future__ import annotations

from src.exporter import ExportedFiles, MatchDataExporter
from src.models import ParsedMatchDataset
from src.parser import parse_match
from src.scraper import WhoScoredScraper, WhoScoredScraperError

__all__ = [
    "ExportedFiles",
    "MatchDataExporter",
    "ParsedMatchDataset",
    "WhoScoredScraper",
    "WhoScoredScraperError",
    "parse_match",
]
