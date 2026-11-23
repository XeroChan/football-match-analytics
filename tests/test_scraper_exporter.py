"""
test_scraper_exporter.py
========================
Unit and integration tests for the Web -> Raw JSON boundary (`WhoScoredScraper`) and
the Dataset -> CSV boundary (`MatchDataExporter`).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.exporter import MatchDataExporter
from src.parser import parse_match
from src.scraper import WhoScoredScraper, _FastFirefoxService
from tests.conftest import RAW_DIR, RAW_FILES


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
    extracted = WhoScoredScraper.extract_all_from_html(html, match_id=999)
    assert extracted is not None
    assert extracted["matchId"] == 999
    assert extracted["_sourceUrl"] == "https://www.whoscored.com/matches/999/live/"
    assert extracted["_matchCentreEventTypeJson"] == {"pass": 1, "goal": 16}
    assert extracted["_formationIdNameMappings"] == {"2": "442"}


def test_scraper_builds_match_url() -> None:
    """Checks that `WhoScoredScraper.build_match_url` builds `/matches/<id>/live/`."""
    assert (
        WhoScoredScraper.build_match_url(1729376)
        == "https://www.whoscored.com/matches/1729376/live/"
    )
    assert (
        WhoScoredScraper.build_match_url(1729401)
        == "https://www.whoscored.com/matches/1729401/live/"
    )


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


def test_scraper_builds_firefox_options() -> None:
    """Checks Firefox options: eager load, prefs, optional UA override, and headless flag."""
    visible_opts = WhoScoredScraper(headless=False)._build_firefox_options()
    assert visible_opts.page_load_strategy == "eager"
    assert visible_opts.preferences["dom.webdriver.enabled"] is False
    assert visible_opts.preferences["useAutomationExtension"] is False
    assert "general.useragent.override" not in visible_opts.preferences
    assert "-headless" not in visible_opts.arguments

    headless_opts = WhoScoredScraper(
        headless=True, user_agent="CustomAgent/1.0",
    )._build_firefox_options()
    assert "-headless" in headless_opts.arguments
    assert headless_opts.preferences["general.useragent.override"] == "CustomAgent/1.0"


def test_fast_firefox_service_ipv4_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks IPv4 host args, IPv4 service URL, and zero-network shutdown on Windows."""
    service = _FastFirefoxService(port=44444)
    args = service.command_line_args()
    assert "--host" in args
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert service.service_url == "http://127.0.0.1:44444"

    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("send_remote_shutdown_command should not make network calls")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    service.send_remote_shutdown_command()


def test_find_cached_geckodriver_picks_newest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that `_find_cached_geckodriver` returns None if empty and picks newest mtime."""
    monkeypatch.delenv("SE_GECKODRIVER", raising=False)
    monkeypatch.setenv("SE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    assert WhoScoredScraper._find_cached_geckodriver() is None

    binary_name = "geckodriver.exe" if os.name == "nt" else "geckodriver"
    older = tmp_path / "geckodriver" / "win64" / "0.35.0" / binary_name
    newer = tmp_path / "geckodriver" / "win64" / "0.36.0" / binary_name
    older.parent.mkdir(parents=True, exist_ok=True)
    newer.parent.mkdir(parents=True, exist_ok=True)
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    assert WhoScoredScraper._find_cached_geckodriver() == str(newer)


def test_exporter_raw_json_overwrite_and_reuse(tmp_path: Path) -> None:
    """
    Checks that `save_raw_json(overwrite=True)` updates existing files and
    `export_all(raw_json_path=...)` skips redundant writes.
    """
    exporter = MatchDataExporter(output_dir=tmp_path / "output", raw_dir=tmp_path / "raw")
    payload_v1 = {"matchId": 101, "score": "1 : 0", "home": {"name": "A"}, "away": {"name": "B"}}
    payload_v2 = {"matchId": 101, "score": "2 : 1", "home": {"name": "A"}, "away": {"name": "B"}}

    raw_path = exporter.save_raw_json(payload_v1)
    assert exporter.load_raw_json(raw_path)["score"] == "1 : 0"

    exporter.save_raw_json(payload_v2, overwrite=False)
    assert exporter.load_raw_json(raw_path)["score"] == "1 : 0"

    exporter.save_raw_json(payload_v2, overwrite=True)
    assert exporter.load_raw_json(raw_path)["score"] == "2 : 1"

    dataset = parse_match(payload_v2)
    existing_path = tmp_path / "custom_raw.json"
    exported = exporter.export_all(payload_v2, dataset, raw_json_path=existing_path)
    assert exported.raw_json_path == existing_path
    assert not existing_path.exists()


@pytest.mark.skipif(not RAW_FILES, reason=f"No raw JSON files in {RAW_DIR}")
def test_exporter_round_trip_csv(tmp_path: Path) -> None:
    """
    Checks that `MatchDataExporter.export_all` writes every non-empty table to CSV
    and that reloading those files preserves row counts and column schemas.
    """
    raw = json.loads(RAW_FILES[0].read_text(encoding="utf-8"))
    dataset = parse_match(raw)
    exporter = MatchDataExporter(output_dir=tmp_path / "output", raw_dir=tmp_path / "raw")
    exported = exporter.export_all(raw, dataset)

    non_empty_tables = {name: df for name, df in dataset.tables().items() if not df.empty}
    assert set(exported.csv_files.keys()) == set(non_empty_tables.keys())

    for name, df in non_empty_tables.items():
        csv_df = pd.read_csv(exported.csv_files[name])
        assert list(csv_df.columns) == list(df.columns)
        assert len(csv_df) == len(df)


@pytest.mark.skipif(len(RAW_FILES) < 2, reason=f"Need at least 2 raw JSON files in {RAW_DIR}")
def test_exporter_combined_catalog_and_tables(tmp_path: Path) -> None:
    """
    Checks that `MatchDataExporter.export_combined` joins multiple `ParsedMatchDataset`
    instances and writes combined CSVs into a separate `output/combined/` catalog.
    """
    datasets = [parse_match(json.loads(f.read_text(encoding="utf-8"))) for f in RAW_FILES[:2]]
    exporter = MatchDataExporter(output_dir=tmp_path / "output", raw_dir=tmp_path / "raw")
    combined_ds, exported = exporter.export_combined(datasets)

    assert exporter.combined_dir == tmp_path / "output" / "combined"

    non_empty_tables = {name: df for name, df in combined_ds.tables().items() if not df.empty}
    assert set(exported.csv_files.keys()) == set(non_empty_tables.keys())

    for name, df in non_empty_tables.items():
        csv_path = exported.csv_files[name]
        assert csv_path.parent == exporter.combined_dir
        assert csv_path.name == f"combined_{name}.csv"
        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) == len(df)

    expected_events = sum(len(ds.events_df) for ds in datasets)
    assert len(combined_ds.events_df) == expected_events
    assert len(combined_ds.metadata_df) == 2
