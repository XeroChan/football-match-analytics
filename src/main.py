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
       python -m src.main --url "https://www.whoscored.com/matches/1729376/live/" "https://..."

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
from src.scraper import WhoScoredScraper, WhoScoredScraperError

logger = logging.getLogger(__name__)

# Array of WhoScored Live match URLs to scrape/process by default when running `python -m src.main`.
DEFAULT_MATCH_URLS: list[str] = [
    # 1. Premier League (England)
    "https://www.whoscored.com/matches/1729376/live/",
    "https://www.whoscored.com/matches/1729390/live/",
    "https://www.whoscored.com/matches/1729401/live/",
    "https://www.whoscored.com/matches/1729415/live/",
    "https://www.whoscored.com/matches/1821106/live/",
    "https://www.whoscored.com/matches/1821174/live/",
    "https://www.whoscored.com/matches/1821210/live/",
    "https://www.whoscored.com/matches/1821245/live/",
    "https://www.whoscored.com/matches/1821288/live/",
    "https://www.whoscored.com/matches/1903158/live/",
    "https://www.whoscored.com/matches/1903399/live/",
    "https://www.whoscored.com/matches/1903417/live/",

    # 2. LaLiga (Spain)
    "https://www.whoscored.com/matches/1734890/live/",
    "https://www.whoscored.com/matches/1734912/live/",
    "https://www.whoscored.com/matches/1734935/live/",
    "https://www.whoscored.com/matches/1825720/live/",
    "https://www.whoscored.com/matches/1913998/live/",
    "https://www.whoscored.com/matches/1914050/live/",
    "https://www.whoscored.com/matches/1914112/live/",
    "https://www.whoscored.com/matches/1914198/live/",

    # 3. Serie A (Italy)
    "https://www.whoscored.com/matches/1746280/live/",
    "https://www.whoscored.com/matches/1746315/live/",
    "https://www.whoscored.com/matches/1746350/live/",
    "https://www.whoscored.com/matches/1834910/live/",
    "https://www.whoscored.com/matches/1834955/live/",
    "https://www.whoscored.com/matches/1835012/live/",
    "https://www.whoscored.com/matches/1835068/live/",
    "https://www.whoscored.com/matches/1835115/live/",

    # 4. Bundesliga (Germany)
    "https://www.whoscored.com/matches/1743590/live/",
    "https://www.whoscored.com/matches/1743635/live/",
    "https://www.whoscored.com/matches/1743680/live/",
    "https://www.whoscored.com/matches/1834310/live/",
    "https://www.whoscored.com/matches/1834355/live/",
    "https://www.whoscored.com/matches/1834402/live/",
    "https://www.whoscored.com/matches/1834448/live/",
    "https://www.whoscored.com/matches/1834490/live/",

    # 5. UEFA Champions League
    "https://www.whoscored.com/matches/1866120/live/",
    "https://www.whoscored.com/matches/1866145/live/",
    "https://www.whoscored.com/matches/1866180/live/",
    "https://www.whoscored.com/matches/1866215/live/",
    "https://www.whoscored.com/matches/1946390/live/",
    "https://www.whoscored.com/matches/1946412/live/",
    "https://www.whoscored.com/matches/1946445/live/",

    # 6. UEFA Europa League

    # 7. Ligue 1 (France)
    "https://www.whoscored.com/matches/1741210/live/",
    "https://www.whoscored.com/matches/1741245/live/",
    "https://www.whoscored.com/matches/1741280/live/",

    # 8. Championship (England)

    # 9. League One (England)
    "https://www.whoscored.com/matches/1732610/live/",
    "https://www.whoscored.com/matches/1732655/live/",
    "https://www.whoscored.com/matches/1732690/live/",

    # 10. League Two (England)
    "https://www.whoscored.com/matches/1733110/live/",
    "https://www.whoscored.com/matches/1733155/live/",
    "https://www.whoscored.com/matches/1733200/live/",
    "https://www.whoscored.com/matches/1990125/live/",
    "https://www.whoscored.com/matches/1990140/live/",
    "https://www.whoscored.com/matches/1990110/live/",

    # 11. FA Cup (England)
    "https://www.whoscored.com/matches/1889763/live/",

    ## =============================================

    # 12. Major League Soccer (USA)

    # 13. Eredivisie (Netherlands)

    # 14. Liga Portugal

    # 15. Scottish Premiership

    # 16. Brasileirão (Brazil)
    "https://www.whoscored.com/matches/1807113/live/",
    "https://www.whoscored.com/matches/1807145/live/",
    "https://www.whoscored.com/matches/1807180/live/",
    "https://www.whoscored.com/matches/1807215/live/",

    # 17. Süper Lig (Turkey)

    # 18. League Cup / EFL Cup (England)

    # 19. Russian Premier League
    "https://www.whoscored.com/matches/1824200/live/",
    "https://www.whoscored.com/matches/1824245/live/",
    "https://www.whoscored.com/matches/1824290/live/",
    "https://www.whoscored.com/matches/1906110/live/",

    # 20. Jupiler Pro League (Belgium)

    # 21. 2. Bundesliga (Germany)

    # 22. UEFA Nations League A
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
    """Normalizes a single URL string or sequence of URLs into canonical `/matches/<id>/live/` URLs."""
    if urls is None:
        return []
    if isinstance(urls, str):
        stripped = urls.strip()
        return [WhoScoredScraper.normalize_match_url(stripped)] if stripped else []
    return [WhoScoredScraper.normalize_match_url(u) for u in urls if u.strip()]


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
    failed_items: list[tuple[str, str]] = []

    if from_json_files:
        for json_file in from_json_files:
            json_path = Path(json_file)
            try:
                raw_data = exporter.load_raw_json(json_path)
                results.append(
                    process_match_payload(raw_data, parser, exporter, raw_json_path=json_path)
                )
            except Exception as exc:
                logger.error("Skipping JSON file %s due to error: %s", json_path, exc)
                failed_items.append((str(json_path), str(exc)))

    normalized_urls = _normalize_url_list(urls)
    urls_to_scrape: list[str] = []
    for url in normalized_urls:
        match_id = WhoScoredScraper.extract_match_id_from_url(url)
        cached_json = None if force_scrape else exporter.find_cached_raw_json(match_id)
        if cached_json is not None:
            logger.info("Using cached raw JSON for matchId=%s: %s", match_id, cached_json)
            try:
                raw_data = exporter.load_raw_json(cached_json)
                results.append(
                    process_match_payload(raw_data, parser, exporter, raw_json_path=cached_json)
                )
            except Exception as exc:
                logger.error("Skipping cached JSON %s (%s) due to error: %s", cached_json, url, exc)
                failed_items.append((url, str(exc)))
        else:
            urls_to_scrape.append(url)

    if urls_to_scrape:
        total_to_scrape = len(urls_to_scrape)
        with WhoScoredScraper(headless=headless) as scraper:
            for idx, url in enumerate(urls_to_scrape, start=1):
                logger.info("[%d/%d] Scraping match URL: %s", idx, total_to_scrape, url)
                try:
                    raw_data = scraper.scrape_match(url)
                    results.append(
                        process_match_payload(
                            raw_data,
                            parser,
                            exporter,
                            overwrite_raw_json=force_scrape,
                        )
                    )
                except (WhoScoredScraperError, Exception) as exc:
                    logger.error(
                        "[%d/%d] Skipping URL (failed to scrape/process): %s -> %s",
                        idx,
                        total_to_scrape,
                        url,
                        exc,
                    )
                    failed_items.append((url, str(exc)))

    if export_combined and results:
        datasets = [ds for ds, _ in results]
        combined_ds, combined_exported = exporter.export_combined(datasets)
        print_combined_summary(len(datasets), combined_ds, combined_exported)

    if failed_items:
        print("\n" + "!" * 72)
        print(f"Skipped {len(failed_items)} Failed URL(s) / File(s):")
        print("!" * 72)
        for source, reason in failed_items:
            print(f"  - {source}\n    Reason: {reason}")
        print("!" * 72 + "\n")

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
