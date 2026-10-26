"""
main.py
=======
Main CLI and pipeline orchestrator for scraping WhoScored Match Centre data
and exporting structured datasets (`events`, `players`, `formations`, `metadata`,
`player_stats`, `team_stats`, `shot_zones`, `minute_mapping`, `lookups`, `common_events`)
in Parquet (`data/processed/`) and CSV (`output/`) formats alongside raw JSON backups.

Usage examples:
    1. Scrape (or load from local cache) the default match URL:
       python -m src.main

    2. Scrape one or more custom WhoScored Live match URLs:
       python -m src.main --url "https://www.whoscored.com/Matches/1821123/Live/..."

    3. Re-generate Parquet & CSV files offline from saved raw JSON backups:
       python -m src.main --from-json "data/raw/match_1821123_Southampton_vs_Aston_Villa_raw.json"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure package imports work whether executed via `python -m src.main` or `python src/main.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exporter import ExportedFiles, MatchDataExporter
from src.parser import MatchDataParser, ParsedMatchDataset
from src.scraper import WhoScoredScraper, WhoScoredScraperError

logger = logging.getLogger(__name__)

DEFAULT_MATCH_URL = "https://www.whoscored.com/Matches/1821123/Live/England-Premier-League"


def configure_logging(verbose: bool = False) -> None:
    """Configures structured console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_dataset_summary(dataset: ParsedMatchDataset, exported: ExportedFiles) -> None:
    """Prints a human-readable overview of the parsed match and exported files."""
    meta = dataset.metadata_df.iloc[0] if not dataset.metadata_df.empty else {}
    home_team = meta.get("home_team_name", "Home")
    away_team = meta.get("away_team_name", "Away")
    score = meta.get("score", "N/A")

    print("\n" + "=" * 72)
    print(f"Match Summary: {home_team} vs {away_team} (Score: {score})")
    print("=" * 72)
    print(f"  Match ID         : {dataset.match_id}")
    print(f"  Total Events     : {len(dataset.events_df):,}")
    print(f"  Total Players    : {len(dataset.players_df):,}")
    print(f"  Formation Slots  : {len(dataset.formations_df):,}")
    print(f"  Player Stat Rows : {len(dataset.player_stats_df):,}")
    print(f"  Team Stat Rows   : {len(dataset.team_stats_df):,}")

    if not dataset.events_df.empty and "event_type" in dataset.events_df.columns:
        print("\nTop 10 Event Types Captured:")
        counts = dataset.events_df["event_type"].value_counts(dropna=False).head(10)
        for ev_type, count in counts.items():
            print(f"  - {str(ev_type):<22}: {count:>5}")

    print("\nExported Files:")
    print(f"  [Raw JSON]       {exported.raw_json_path.resolve()}")
    print(f"  [Events CSV]     {exported.events_csv_path.resolve()}")
    print(f"  [Players CSV]    {exported.players_csv_path.resolve()}")
    if exported.formations_csv_path:
        print(f"  [Formations CSV] {exported.formations_csv_path.resolve()}")
    print(f"  [Metadata CSV]   {exported.metadata_csv_path.resolve()}")
    for table_name, csv_path in exported.extra_csv_paths.items():
        label = f"[{table_name} CSV]"
        print(f"  {label:<16} {csv_path.resolve()}")
    if exported.parquet_files:
        parquet_dir = next(iter(exported.parquet_files.values())).parent.resolve()
        print(f"  [Parquet Dir]    {parquet_dir} ({len(exported.parquet_files)} tables)")
    print("=" * 72 + "\n")


def process_match_payload(
    raw_data: dict[str, Any],
    parser: MatchDataParser,
    exporter: MatchDataExporter,
) -> tuple[ParsedMatchDataset, ExportedFiles]:
    """Parses a raw `matchCentreData` dictionary and exports all configured formats."""
    dataset = parser.parse_all(raw_data)
    exported = exporter.export_all(raw_data, dataset)
    print_dataset_summary(dataset, exported)
    return dataset, exported


def run_pipeline(
    urls: Sequence[str] | None = None,
    from_json_files: Sequence[str | Path] | None = None,
    output_dir: str | Path = "output",
    processed_dir: str | Path = "data/processed",
    headless: bool = False,
    force_scrape: bool = False,
) -> list[tuple[ParsedMatchDataset, ExportedFiles]]:
    """
    Runs the end-to-end scraping, parsing, and export pipeline.
    Automatically reuses cached `data/raw/match_<id>_*.json` files unless `force_scrape=True`.
    """
    parser = MatchDataParser()
    exporter = MatchDataExporter(output_dir=output_dir, processed_dir=processed_dir)
    results: list[tuple[ParsedMatchDataset, ExportedFiles]] = []

    if from_json_files:
        for json_file in from_json_files:
            raw_data = exporter.load_raw_json(json_file)
            results.append(process_match_payload(raw_data, parser, exporter))

    urls_to_scrape: list[str] = []
    for url in urls or []:
        match_id = WhoScoredScraper.extract_match_id_from_url(url)
        cached_json = None if force_scrape else exporter.find_cached_raw_json(match_id)
        if cached_json is not None:
            logger.info("Using cached raw JSON for matchId=%s: %s", match_id, cached_json)
            raw_data = exporter.load_raw_json(cached_json)
            results.append(process_match_payload(raw_data, parser, exporter))
        else:
            urls_to_scrape.append(url)

    if urls_to_scrape:
        with WhoScoredScraper(headless=headless) as scraper:
            for url in urls_to_scrape:
                try:
                    raw_data = scraper.scrape_match(url)
                    results.append(process_match_payload(raw_data, parser, exporter))
                except WhoScoredScraperError as exc:
                    logger.error("Failed to scrape %s: %s", url, exc)

    return results


def parse_cli_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="Scrape WhoScored match data and export Parquet + CSV tables for analytics."
    )
    cli.add_argument(
        "--url",
        "-u",
        nargs="+",
        help=f"One or more WhoScored Live match URLs (default: {DEFAULT_MATCH_URL}).",
    )
    cli.add_argument(
        "--from-json",
        "-j",
        nargs="+",
        help="Parse one or more existing raw match JSON files offline instead of launching Chrome.",
    )
    cli.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory where CSV files will be saved (default: 'output').",
    )
    cli.add_argument(
        "--processed-dir",
        "-p",
        default="data/processed",
        help="Directory where Parquet files will be saved (default: 'data/processed').",
    )
    cli.add_argument(
        "--force-scrape",
        action="store_true",
        help="Re-scrape URLs via Selenium even if a raw JSON backup already exists in data/raw/.",
    )
    cli.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (visible mode is recommended to avoid Cloudflare).",
    )
    cli.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return cli.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    configure_logging(verbose=args.verbose)

    target_urls = args.url
    if not target_urls and not args.from_json:
        target_urls = [DEFAULT_MATCH_URL]

    run_pipeline(
        urls=target_urls,
        from_json_files=args.from_json,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        headless=args.headless,
        force_scrape=args.force_scrape,
    )
