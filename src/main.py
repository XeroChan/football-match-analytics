"""
main.py
=======
Main CLI and pipeline orchestrator for scraping WhoScored Match Centre data
and exporting structured datasets (`events`, `players`, `formations`, `metadata`,
`player_stats`, `team_stats`, `shot_zones`, `minute_mapping`, `lookups`, `common_events`)
as CSV files (`output/`) alongside raw JSON backups (`data/raw/`),
plus combined multi-match datasets in `output/combined/`.

Usage examples:
    1. Scrape (or load from local cache) the configured array of match IDs (`DEFAULT_MATCH_IDS`):
       python -m src.main

    2. Scrape one or more custom WhoScored match IDs:
       python -m src.main --match-id 1729376 1729390

    3. Re-generate per-match and combined CSV files offline from saved raw JSONs:
       python -m src.main --from-json data/raw/*_raw.json
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.exporter import ExportedFiles, MatchDataExporter
from src.match_ids import DEFAULT_MATCH_IDS
from src.models import ParsedMatchDataset
from src.parser import parse_match
from src.scraper import WhoScoredScraper, WhoScoredScraperError

logger = logging.getLogger(__name__)


def print_summary(
    dataset: ParsedMatchDataset,
    exported: ExportedFiles,
    title: str | None = None,
) -> None:
    """Prints a human-readable overview of a parsed dataset and its exported files."""
    if title is None:
        meta: dict[Any, Any] = (
            dataset.metadata_df.iloc[0].to_dict() if not dataset.metadata_df.empty else {}
        )
        home = meta.get("home_team_name", "Home")
        away = meta.get("away_team_name", "Away")
        title = f"Match Summary: {home} vs {away} (Score: {meta.get('score', 'N/A')})"

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if exported.raw_json_path is not None:
        print(f"  Match ID         : {dataset.match_id}")
    for table_name, df in dataset.tables().items():
        if not df.empty:
            print(f"  {table_name:<16} : {len(df):,} rows ({len(df.columns)} cols)")

    if (
        exported.raw_json_path is not None
        and not dataset.events_df.empty
        and "event_type" in dataset.events_df.columns
    ):
        print("\nTop 10 Event Types Captured:")
        counts = dataset.events_df["event_type"].value_counts(dropna=False).head(10)
        for ev_type, count in counts.items():
            print(f"  - {ev_type!s:<22}: {count:>5}")

    print("\nExported Files:")
    if exported.raw_json_path is not None:
        print(f"  [Raw JSON]       {exported.raw_json_path.resolve()}")
    for table_name, csv_path in exported.csv_files.items():
        label = f"[{table_name} CSV]"
        print(f"  {label:<16} {csv_path.resolve()}")
    print("=" * 72 + "\n")


def run_pipeline(
    match_ids: int | Sequence[int] | None = None,
    from_json_files: Sequence[str | Path] | None = None,
    output_dir: str | Path = "output",
    combined_dir: str | Path | None = None,
    export_combined: bool = True,
    headless: bool = False,
    force_scrape: bool = False,
) -> list[tuple[ParsedMatchDataset, ExportedFiles]]:
    """
    Runs the end-to-end scraping, parsing, and export pipeline for one or an array of match IDs.
    Automatically reuses cached `data/raw/match_<id>_*.json` files unless `force_scrape=True`,
    and exports joined multi-match tables to `output/combined/` when `export_combined=True`.
    """
    exporter = MatchDataExporter(output_dir=output_dir, combined_dir=combined_dir)
    results: list[tuple[ParsedMatchDataset, ExportedFiles]] = []
    failed_items: list[tuple[str, str]] = []

    json_paths: list[Path] = [Path(p) for p in (from_json_files or [])]
    normalized_ids: list[int] = (
        [match_ids]
        if isinstance(match_ids, int)
        else [int(mid) for mid in (match_ids or [])]
    )
    ids_to_scrape: list[int] = []
    for match_id in normalized_ids:
        cached_json = None if force_scrape else exporter.find_cached_raw_json(match_id)
        if cached_json is not None:
            logger.info("Using cached raw JSON for matchId=%s: %s", match_id, cached_json)
            json_paths.append(cached_json)
        else:
            ids_to_scrape.append(match_id)

    for json_path in json_paths:
        try:
            raw_data = exporter.load_raw_json(json_path)
            dataset = parse_match(raw_data)
            exported = exporter.export_all(raw_data, dataset, raw_json_path=json_path)
            print_summary(dataset, exported)
            results.append((dataset, exported))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("Skipping JSON file %s due to error: %s", json_path, exc)
            failed_items.append((str(json_path), str(exc)))

    if ids_to_scrape:
        total = len(ids_to_scrape)
        with WhoScoredScraper(headless=headless) as scraper:
            for idx, match_id in enumerate(ids_to_scrape, start=1):
                logger.info("[%d/%d] Scraping matchId=%s", idx, total, match_id)
                try:
                    raw_data = scraper.scrape_match(match_id)
                    dataset = parse_match(raw_data)
                    exported = exporter.export_all(
                        raw_data, dataset, overwrite_raw_json=force_scrape,
                    )
                    print_summary(dataset, exported)
                    results.append((dataset, exported))
                except (WhoScoredScraperError, OSError, ValueError, KeyError, TypeError) as exc:
                    logger.error(
                        "[%d/%d] Skipping matchId=%s (failed to scrape/process): %s",
                        idx,
                        total,
                        match_id,
                        exc,
                    )
                    failed_items.append((f"matchId={match_id}", str(exc)))

    if export_combined and results:
        datasets = [ds for ds, _ in results]
        combined_ds, combined_exported = exporter.export_combined(datasets)
        print_summary(
            combined_ds,
            combined_exported,
            title=f"Combined Dataset Summary ({len(datasets)} matches joined)",
        )

    if failed_items:
        print("\n" + "!" * 72)
        print(f"Skipped {len(failed_items)} Failed Match(es) / File(s):")
        print("!" * 72)
        for source, reason in failed_items:
            print(f"  - {source}\n    Reason: {reason}")
        print("!" * 72 + "\n")

    return results


def parse_cli_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="Scrape WhoScored match data and export CSV tables for analytics.",
    )
    cli.add_argument(
        "--match-id",
        "-m",
        type=int,
        nargs="+",
        help="One or more WhoScored numeric match IDs (defaults to DEFAULT_MATCH_IDS).",
    )
    cli.add_argument(
        "--from-json",
        "-j",
        nargs="+",
        help=(
            "Parse one or more existing raw match JSON files offline "
            "instead of launching Firefox."
        ),
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
        help="Skip generating combined CSV tables across all processed matches.",
    )
    cli.add_argument(
        "--force-scrape",
        action="store_true",
        help="Re-scrape via Selenium even if a raw JSON backup already exists in data/raw/.",
    )
    cli.add_argument(
        "--headless",
        action="store_true",
        help="Run Firefox in headless mode (visible mode is recommended to avoid Cloudflare).",
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
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    target_match_ids = args.match_id
    if not target_match_ids and not args.from_json:
        target_match_ids = DEFAULT_MATCH_IDS

    run_pipeline(
        match_ids=target_match_ids,
        from_json_files=args.from_json,
        output_dir=args.output_dir,
        combined_dir=args.combined_dir,
        export_combined=not args.no_combined,
        headless=args.headless,
        force_scrape=args.force_scrape,
    )
