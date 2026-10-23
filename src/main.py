"""
main.py
=======
Main CLI and pipeline orchestrator for scraping WhoScored Match Centre data
and exporting comprehensive CSV datasets (`events`, `players`, `metadata`) + raw JSON backups.

Usage examples:
    1. Scrape the default match URL configured below:
       python main.py

    2. Scrape one or more custom WhoScored Live match URLs:
       python main.py --url "https://www.whoscored.com/Matches/1821123/Live/England-Premier-League"

    3. Re-generate CSVs offline from a previously saved raw JSON backup (no browser needed):
       python main.py --from-json "output/match_1821123_raw.json"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from src.exporter import ExportedFiles, MatchDataExporter
from src.parser import MatchDataParser, ParsedMatchDataset
from src.scraper import WhoScoredScraper

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
    """Prints a human-readable overview of the scraped match and exported CSV files."""
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

    if not dataset.events_df.empty and "event_type" in dataset.events_df.columns:
        print("\nTop 10 Event Types Captured:")
        counts = dataset.events_df["event_type"].value_counts(dropna=False).head(10)
        for ev_type, count in counts.items():
            print(f"  - {str(ev_type):<22}: {count:>5}")

    print("\nExported Files for Data Exploration:")
    print(f"  [Events CSV]   {exported.events_csv_path.resolve()}")
    print(f"  [Players CSV]  {exported.players_csv_path.resolve()}")
    print(f"  [Metadata CSV] {exported.metadata_csv_path.resolve()}")
    print(f"  [Raw JSON]     {exported.raw_json_path.resolve()}")
    print("=" * 72 + "\n")


def process_match_payload(
    raw_data: dict[str, Any],
    parser: MatchDataParser,
    exporter: MatchDataExporter,
) -> tuple[ParsedMatchDataset, ExportedFiles]:
    """Parses a raw `matchCentreData` dictionary and exports all CSV and JSON files."""
    dataset = parser.parse_all(raw_data)
    exported = exporter.export_all(raw_data, dataset)
    print_dataset_summary(dataset, exported)
    return dataset, exported


def run_pipeline(
    urls: Sequence[str] | None = None,
    from_json_files: Sequence[str | Path] | None = None,
    output_dir: str | Path = "output",
    headless: bool = False,
) -> list[tuple[ParsedMatchDataset, ExportedFiles]]:
    """
    Runs the end-to-end scraping, parsing, and CSV export pipeline.

    Args:
        urls: List of WhoScored `/Live/` match URLs to scrape.
        from_json_files: Optional list of already downloaded raw JSON files to parse offline.
        output_dir: Target folder for exported CSV and JSON files.
        headless: Whether to run Chrome in headless mode.

    Returns:
        List of `(ParsedMatchDataset, ExportedFiles)` tuples for each processed match.
    """
    parser = MatchDataParser()
    exporter = MatchDataExporter(output_dir=output_dir)
    results: list[tuple[ParsedMatchDataset, ExportedFiles]] = []

    # 1. Process any offline JSON files first (no browser required)
    if from_json_files:
        for json_file in from_json_files:
            raw_data = exporter.load_raw_json(json_file)
            results.append(process_match_payload(raw_data, parser, exporter))

    # 2. Scrape live URLs in a single managed browser session
    if urls:
        with WhoScoredScraper(headless=headless) as scraper:
            for url in urls:
                raw_data = scraper.scrape_match(url)
                results.append(process_match_payload(raw_data, parser, exporter))

    return results


def parse_cli_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="Scrape WhoScored match data and export complete CSV files for data exploration."
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
        help="Directory where CSV and JSON files will be saved (default: 'output').",
    )
    cli.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (note: visible mode is less likely to trigger Cloudflare).",
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
        headless=args.headless,
    )

