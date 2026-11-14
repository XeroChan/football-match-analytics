"""
main.py
=======
Main CLI and pipeline orchestrator for scraping WhoScored Match Centre data
and exporting structured datasets (`events`, `players`, `formations`, `metadata`,
`player_stats`, `team_stats`, `shot_zones`, `minute_mapping`, `lookups`, `common_events`)
in Parquet (`data/processed/`) and CSV (`output/`) formats alongside raw JSON backups,
plus combined multi-match datasets in `output/combined/` and `data/processed/combined/`.

Usage examples:
    1. Scrape (or load from local cache) the configured array of match URLs (`DEFAULT_MATCH_URLS`):
       python -m src.main

    2. Scrape one or more custom WhoScored Live match URLs:
       python -m src.main --url "https://www.whoscored.com/Matches/1821123/Live/..." "https://..."

    3. Re-generate Parquet, per-match CSV, and combined CSV files offline from saved raw JSON backups:
       python -m src.main --from-json data/raw/*_raw.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from src.exporter import ExportedFiles, MatchDataExporter
from src.models import ParsedMatchDataset
from src.parser import MatchDataParser
from src.scraper import WhoScoredScraper

logger = logging.getLogger(__name__)

# Array of WhoScored Live match URLs to scrape/process by default when running `python -m src.main`.
DEFAULT_MATCH_URLS: list[str] = [
    "https://www.whoscored.com/matches/1821123/live/england-premier-league-2024-2025-southampton-aston_villa",
    "https://www.whoscored.com/matches/1983593/live/england-premier-league-2026-2027-chelsea-bournemouth",
    "https://www.whoscored.com/matches/1983603/live/england-premier-league-2026-2027-liverpool-manchester-city",
    "https://www.whoscored.com/matches/1993972/live/spain-laliga-2026-2027-barcelona-deportivo-alaves",
    "https://www.whoscored.com/matches/1993980/live/spain-laliga-2026-2027-levante-atletico-madrid",
    "https://www.whoscored.com/matches/1993983/live/spain-laliga-2026-2027-racing-santander-real-madrid",
    "https://www.whoscored.com/matches/1995426/live/germany-bundesliga-2026-2027-werder-bremen-paderborn",
    "https://www.whoscored.com/matches/1998572/live/international-fifa-world-cup-2026-switzerland-colombia",
    "https://www.whoscored.com/matches/2002449/live/international-fifa-world-cup-2026-france-spain",
    "https://www.whoscored.com/matches/2029242/live/europe-champions-league-2026-2027-viking-bayern-munich",
    "https://www.whoscored.com/matches/1991112/live/russia-premier-league-2026-2027-krylya-sovetov-samara-fc-orenburg",
    "https://www.whoscored.com/matches/1991103/live/russia-premier-league-2026-2027-rubin-kazan-fakel",
    "https://www.whoscored.com/matches/1999627/live/turkey-super-lig-2026-2027-eyupspor-kocaelispor",
    "https://www.whoscored.com/matches/1985905/live/belgium-jupiler-pro-league-2026-2027-kortrijk-kv-mechelen",
    "https://www.whoscored.com/matches/1995758/live/germany-2-bundesliga-2026-2027-arminia-bielefeld-eintracht-braunschweig",
    "https://www.whoscored.com/matches/1995450/live/germany-bundesliga-2026-2027-bayern-munich-borussia-dortmund",
    "https://www.whoscored.com/matches/1995458/live/germany-bundesliga-2026-2027-union-berlin-fc-koln",
    "https://www.whoscored.com/matches/1995418/live/germany-bundesliga-2026-2027-elversberg-augsburg",
    "https://www.whoscored.com/matches/1995411/live/germany-bundesliga-2026-2027-bayern-munich-rb-leipzig",
    "https://www.whoscored.com/matches/1995424/live/germany-bundesliga-2026-2027-union-berlin-borussia-dortmund",
    "https://www.whoscored.com/matches/1993932/live/spain-laliga-2026-2027-celta-vigo-real-betis",
    "https://www.whoscored.com/matches/1993934/live/spain-laliga-2026-2027-barcelona-real-madrid",
    "https://www.whoscored.com/matches/1994185/live/spain-laliga-2026-2027-getafe-rayo-vallecano",
    "https://www.whoscored.com/matches/1994189/live/spain-laliga-2026-2027-valencia-athletic-club",
    "https://www.whoscored.com/matches/1984218/live/france-ligue-1-2026-2027-le-havre-paris-saint-germain",
    "https://www.whoscored.com/matches/1984096/live/france-ligue-1-2026-2027-paris-fc-monaco",
    "https://www.whoscored.com/matches/1984184/live/france-ligue-1-2026-2027-auxerre-le-mans",
    "https://www.whoscored.com/matches/1984143/live/france-ligue-1-2026-2027-angers-marseille",
    "https://www.whoscored.com/matches/1956076/live/brazil-brasileirão-2026-vitoria-athletico-paranaense",
    "https://www.whoscored.com/matches/1956089/live/brazil-brasileirão-2026-bahia-sao-paulo",
    "https://www.whoscored.com/matches/1956054/live/brazil-brasileirão-2026-gremio-cruzeiro",
    "https://www.whoscored.com/matches/1956062/live/brazil-brasileirão-2026-vasco-da-gama-corinthians",
    "https://www.whoscored.com/matches/1953075/live/usa-major-league-soccer-2026-inter-miami-cf-fc-cincinnati",
    "https://www.whoscored.com/matches/1953076/live/usa-major-league-soccer-2026-new-england-revolution-red-bull-new-york",
    "https://www.whoscored.com/matches/1953095/live/usa-major-league-soccer-2026-la-galaxy-austin-fc",
    "https://www.whoscored.com/matches/1953064/live/usa-major-league-soccer-2026-orlando-city-new-york-city-fc",
    "https://www.whoscored.com/matches/1953094/live/usa-major-league-soccer-2026-sporting-kansas-city-fc-dallas",

]


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
    meta: dict[Any, Any] = (
        dataset.metadata_df.iloc[0].to_dict() if not dataset.metadata_df.empty else {}
    )
    home_team = meta.get("home_team_name", "Home")
    away_team = meta.get("away_team_name", "Away")
    score = meta.get("score", "N/A")

    print("\n" + "=" * 72)
    print(f"Match Summary: {home_team} vs {away_team} (Score: {score})")
    print("=" * 72)
    print(f"  Match ID         : {dataset.match_id}")
    for table_name, df in dataset.tables().items():
        if not df.empty:
            print(f"  {table_name:<16} : {len(df):,} rows ({len(df.columns)} cols)")

    if not dataset.events_df.empty and "event_type" in dataset.events_df.columns:
        print("\nTop 10 Event Types Captured:")
        counts = dataset.events_df["event_type"].value_counts(dropna=False).head(10)
        for ev_type, count in counts.items():
            print(f"  - {str(ev_type):<22}: {count:>5}")

    print("\nExported Files:")
    print(f"  [Raw JSON]       {exported.raw_json_path.resolve()}")
    for table_name, csv_path in exported.csv_files.items():
        label = f"[{table_name} CSV]"
        print(f"  {label:<16} {csv_path.resolve()}")
    if exported.parquet_files:
        parquet_dir = next(iter(exported.parquet_files.values())).parent.resolve()
        print(f"  [Parquet Dir]    {parquet_dir} ({len(exported.parquet_files)} tables)")
    print("=" * 72 + "\n")


def print_combined_summary(
    match_count: int,
    combined_dataset: ParsedMatchDataset,
    exported: ExportedFiles,
) -> None:
    """Prints a summary of the combined multi-match dataset and exported combined files."""
    print("\n" + "=" * 72)
    print(f"Combined Dataset Summary ({match_count} matches joined)")
    print("=" * 72)
    for table_name, df in combined_dataset.tables().items():
        if not df.empty:
            print(f"  {table_name:<16} : {len(df):,} rows ({len(df.columns)} cols)")

    print("\nExported Combined Files:")
    for table_name, csv_path in exported.csv_files.items():
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
    raw_json_path: Path | None = None,
    overwrite_raw_json: bool = False,
) -> tuple[ParsedMatchDataset, ExportedFiles]:
    """Parses a raw `matchCentreData` dictionary and exports all configured formats."""
    dataset = parser.parse_all(raw_data)
    exported = exporter.export_all(
        raw_data,
        dataset,
        raw_json_path=raw_json_path,
        overwrite_raw_json=overwrite_raw_json,
    )
    print_dataset_summary(dataset, exported)
    return dataset, exported


def _normalize_url_list(urls: str | Sequence[str] | None) -> list[str]:
    """Normalizes a single URL string or sequence of URLs into a flat list of non-empty URLs."""
    if urls is None:
        return []
    if isinstance(urls, str):
        stripped = urls.strip()
        return [stripped] if stripped else []
    return [u.strip() for u in urls if u.strip()]


def run_pipeline(
    urls: str | Sequence[str] | None = None,
    from_json_files: Sequence[str | Path] | None = None,
    output_dir: str | Path = "output",
    processed_dir: str | Path = "data/processed",
    combined_dir: str | Path | None = None,
    export_combined: bool = True,
    headless: bool = False,
    force_scrape: bool = False,
) -> list[tuple[ParsedMatchDataset, ExportedFiles]]:
    """
    Runs the end-to-end scraping, parsing, and export pipeline for one or an array of URLs.
    Automatically reuses cached `data/raw/match_<id>_*.json` files unless `force_scrape=True`,
    and exports joined multi-match tables to `output/combined/` when `export_combined=True`.
    """
    parser = MatchDataParser()
    exporter = MatchDataExporter(
        output_dir=output_dir,
        processed_dir=processed_dir,
        combined_dir=combined_dir,
    )
    results: list[tuple[ParsedMatchDataset, ExportedFiles]] = []

    if from_json_files:
        for json_file in from_json_files:
            json_path = Path(json_file)
            raw_data = exporter.load_raw_json(json_path)
            results.append(
                process_match_payload(raw_data, parser, exporter, raw_json_path=json_path)
            )

    normalized_urls = _normalize_url_list(urls)
    urls_to_scrape: list[str] = []
    for url in normalized_urls:
        match_id = WhoScoredScraper.extract_match_id_from_url(url)
        cached_json = None if force_scrape else exporter.find_cached_raw_json(match_id)
        if cached_json is not None:
            logger.info("Using cached raw JSON for matchId=%s: %s", match_id, cached_json)
            raw_data = exporter.load_raw_json(cached_json)
            results.append(
                process_match_payload(raw_data, parser, exporter, raw_json_path=cached_json)
            )
        else:
            urls_to_scrape.append(url)

    if urls_to_scrape:
        with WhoScoredScraper(headless=headless) as scraper:
            raw_payloads = scraper.scrape_matches(urls_to_scrape)
            for raw_data in raw_payloads:
                results.append(
                    process_match_payload(
                        raw_data,
                        parser,
                        exporter,
                        overwrite_raw_json=force_scrape,
                    )
                )

    if export_combined and results:
        datasets = [ds for ds, _ in results]
        combined_ds, combined_exported = exporter.export_combined(datasets)
        print_combined_summary(len(datasets), combined_ds, combined_exported)

    return results


def parse_cli_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="Scrape WhoScored match data and export Parquet + CSV tables for analytics."
    )
    cli.add_argument(
        "--url",
        "-u",
        nargs="+",
        help="One or more WhoScored Live match URLs (defaults to DEFAULT_MATCH_URLS).",
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
        help="Directory where per-match CSV files will be saved (default: 'output').",
    )
    cli.add_argument(
        "--combined-dir",
        "-c",
        default=None,
        help="Directory where combined CSV files will be saved (default: '<output-dir>/combined').",
    )
    cli.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip generating combined CSV/Parquet tables across all processed matches.",
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
        target_urls = DEFAULT_MATCH_URLS

    run_pipeline(
        urls=target_urls,
        from_json_files=args.from_json,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        combined_dir=args.combined_dir,
        export_combined=not args.no_combined,
        headless=args.headless,
        force_scrape=args.force_scrape,
    )
