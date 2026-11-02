"""
test_scraper_exporter.py
========================
Unit and integration tests for the Web -> Raw JSON boundary (`WhoScoredScraper`) and
the Dataset -> CSV/Parquet boundary (`MatchDataExporter`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.exporter import MatchDataExporter
from src.parser import MatchDataParser
from src.scraper import WhoScoredScraper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILES = sorted(RAW_DIR.glob("*_raw.json"))


def test_scraper_html_fallback_preserves_lookup_tables() -> None:
    """
    Checks that `WhoScoredScraper.extract_all_from_html` recovers `matchCentreData` plus
    every auxiliary dictionary in `SAVED_PAGE_ARG_MAP` just like the primary JS path.
    """
    html = """
    <script>
    require.config.params["args"] = {
        matchCentreData: {"score": "2 : 1", "events": [{"id": 1}]},
        matchCentreEventTypeJson: {"pass": 1, "goal": 16},
        formationIdNameMappings: {"2": "442"}
    };
    </script>
    """
    url = "https://www.whoscored.com/Matches/999/Live/Test"
    extracted = WhoScoredScraper.extract_all_from_html(html, match_url=url)
    assert extracted is not None
    assert extracted["matchId"] == 999
    assert extracted["_sourceUrl"] == url
    assert extracted["_matchCentreEventTypeJson"] == {"pass": 1, "goal": 16}
    assert extracted["_formationIdNameMappings"] == {"2": "442"}


def test_scraper_detects_unsaved_page_args() -> None:
    """Checks that any new `require.config.params['args']` key is flagged as unsaved."""
    page_args = [
        {"key": "matchCentreData", "type": "object", "size": 20, "preview": "{...}"},
        {"key": "matchCentreEventTypeJson", "type": "object", "size": 50, "preview": "{...}"},
        {"key": "formationIdNameMappings", "type": "object", "size": 25, "preview": "{...}"},
        {"key": "newWhoScoredWidgetData", "type": "object", "size": 5, "preview": "{...}"},
    ]
    unsaved = WhoScoredScraper.find_unsaved_page_args(page_args)
    assert [u["key"] for u in unsaved] == ["newWhoScoredWidgetData"]


def test_exporter_raw_json_overwrite_and_reuse(tmp_path: Path) -> None:
    """Checks that `save_raw_json(overwrite=True)` updates existing files and `export_all(raw_json_path=...)` skips redundant writes."""
    exporter = MatchDataExporter(
        output_dir=tmp_path / "output",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        export_parquet=False,
        export_csv=False,
    )
    payload_v1 = {"matchId": 101, "score": "1 : 0", "home": {"name": "A"}, "away": {"name": "B"}}
    payload_v2 = {"matchId": 101, "score": "2 : 1", "home": {"name": "A"}, "away": {"name": "B"}}

    raw_path = exporter.save_raw_json(payload_v1)
    assert exporter.load_raw_json(raw_path)["score"] == "1 : 0"

    exporter.save_raw_json(payload_v2, overwrite=False)
    assert exporter.load_raw_json(raw_path)["score"] == "1 : 0"

    exporter.save_raw_json(payload_v2, overwrite=True)
    assert exporter.load_raw_json(raw_path)["score"] == "2 : 1"

    dataset = MatchDataParser().parse_all(payload_v2)
    existing_path = tmp_path / "custom_raw.json"
    exported = exporter.export_all(payload_v2, dataset, raw_json_path=existing_path)
    assert exported.raw_json_path == existing_path
    assert not existing_path.exists()


@pytest.mark.skipif(not RAW_FILES, reason=f"No raw JSON files in {RAW_DIR}")
def test_exporter_round_trip_csv_and_parquet(tmp_path: Path) -> None:
    """
    Checks that `MatchDataExporter.export_all` writes every non-empty table to both CSV
    and Parquet and that reloading those files preserves row counts and column schemas.
    """
    raw = json.loads(RAW_FILES[0].read_text(encoding="utf-8"))
    dataset = MatchDataParser().parse_all(raw)
    exporter = MatchDataExporter(
        output_dir=tmp_path / "output",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        export_parquet=True,
        export_csv=True,
    )
    exported = exporter.export_all(raw, dataset)

    non_empty_tables = {name: df for name, df in dataset.tables().items() if not df.empty}
    assert set(exported.csv_files.keys()) == set(non_empty_tables.keys())
    assert set(exported.parquet_files.keys()) == set(non_empty_tables.keys())

    for name, df in non_empty_tables.items():
        csv_df = pd.read_csv(exported.csv_files[name])
        parquet_df = pd.read_parquet(exported.parquet_files[name])
        assert list(csv_df.columns) == list(df.columns)
        assert list(parquet_df.columns) == list(df.columns)
        assert len(csv_df) == len(df)
        assert len(parquet_df) == len(df)

