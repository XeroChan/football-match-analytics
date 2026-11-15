"""
exporter.py
===========
Handles persisting raw WhoScored JSON payloads and structured Pandas DataFrames
to disk (Parquet in `data/processed/` and CSV in `output/`) with deterministic filenames.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.models import ExportedFiles, ParsedMatchDataset

__all__ = ["ExportedFiles", "MatchDataExporter"]

logger = logging.getLogger(__name__)


class MatchDataExporter:
    """
    Exports the raw `matchCentreData` JSON to `raw_dir`, Parquet tables to `processed_dir`,
    and CSV tables (`events`, `players`, `formations`, `metadata`, `player_stats`,
    `team_stats`, `shot_zones`, `minute_mapping`, `lookups`, `common_events`) to `output_dir`,
    plus multi-match combined datasets to `combined_dir` (`output/combined/`).
    """

    def __init__(
        self,
        output_dir: str | Path = "output",
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        combined_dir: str | Path | None = None,
        combined_processed_dir: str | Path | None = None,
        export_parquet: bool = True,
        export_csv: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.combined_dir = (
            Path(combined_dir) if combined_dir is not None else self.output_dir / "combined"
        )
        self.combined_processed_dir = (
            Path(combined_processed_dir)
            if combined_processed_dir is not None
            else self.processed_dir / "combined"
        )
        self.export_parquet = export_parquet
        self.export_csv = export_csv

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.export_csv:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.export_parquet:
            self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slugify(value: Any) -> str:
        """Creates a filesystem-safe slug from team names or identifiers."""
        if not value:
            return ""
        text = re.sub(r"[^\w\s-]", "", str(value)).strip()
        return re.sub(r"[-\s]+", "_", text)

    def _build_file_prefix(self, match_data: dict[str, Any]) -> str:
        """Builds a human-readable file prefix (e.g. `match_1821123_Southampton_vs_Aston_Villa`)."""
        match_id = match_data.get("matchId") or "unknown"
        home_name = self._slugify((match_data.get("home") or {}).get("name"))
        away_name = self._slugify((match_data.get("away") or {}).get("name"))

        if home_name and away_name:
            return f"match_{match_id}_{home_name}_vs_{away_name}"
        return f"match_{match_id}"

    def find_cached_raw_json(self, match_id: Any) -> Path | None:
        """Finds an existing raw JSON file in `raw_dir` for the given `match_id`, if present."""
        if not match_id:
            return None
        matches = sorted(self.raw_dir.glob(f"match_{match_id}*_raw.json"))
        return matches[0] if matches else None

    def save_raw_json(
        self,
        match_data: dict[str, Any],
        prefix: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Saves the complete raw `matchCentreData` dictionary to a UTF-8 JSON file."""
        file_prefix = prefix or self._build_file_prefix(match_data)
        json_path = self.raw_dir / f"{file_prefix}_raw.json"
        if overwrite or not json_path.exists():
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(match_data, f, ensure_ascii=False, indent=2)
            logger.info("Saved raw match JSON backup to: %s", json_path)
        else:
            logger.debug("Raw match JSON backup already exists at: %s", json_path)
        return json_path

    @staticmethod
    def load_raw_json(json_path: str | Path) -> dict[str, Any]:
        """Loads a previously saved raw match JSON file from disk."""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_table(
        self,
        df: pd.DataFrame,
        prefix: str,
        table_name: str,
        csv_files: dict[str, Path],
        parquet_files: dict[str, Path],
        csv_dir: Path | None = None,
        parquet_dir: Path | None = None,
    ) -> None:
        """Writes a non-empty DataFrame to CSV and/or Parquet."""
        if df.empty:
            return

        target_csv_dir = csv_dir if csv_dir is not None else self.output_dir
        target_parquet_dir = parquet_dir if parquet_dir is not None else self.processed_dir

        if self.export_csv:
            target_csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = target_csv_dir / f"{prefix}_{table_name}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            csv_files[table_name] = csv_path
            logger.info(
                "Saved %s CSV (%d rows, %d cols) to: %s",
                table_name,
                len(df),
                len(df.columns),
                csv_path,
            )

        if self.export_parquet:
            target_parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = target_parquet_dir / f"{prefix}_{table_name}.parquet"
            try:
                df.to_parquet(parquet_path, index=False)
                parquet_files[table_name] = parquet_path
                logger.info("Saved %s Parquet to: %s", table_name, parquet_path)
            except (OSError, ImportError, ValueError) as exc:
                logger.warning("Skipping Parquet export for %s (%s)", table_name, exc)

    def export_all(
        self,
        match_data: dict[str, Any],
        dataset: ParsedMatchDataset,
        raw_json_path: str | Path | None = None,
        overwrite_raw_json: bool = False,
    ) -> ExportedFiles:
        """
        Saves the raw JSON backup (unless an existing `raw_json_path` is provided)
        and exports all non-empty structured tables to CSV and Parquet.
        """
        prefix = self._build_file_prefix(match_data)
        if raw_json_path is not None:
            saved_raw_path = Path(raw_json_path)
        else:
            saved_raw_path = self.save_raw_json(
                match_data, prefix=prefix, overwrite=overwrite_raw_json,
            )

        csv_files: dict[str, Path] = {}
        parquet_files: dict[str, Path] = {}
        for table_name, df in dataset.tables().items():
            self._write_table(df, prefix, table_name, csv_files, parquet_files)

        return ExportedFiles(
            raw_json_path=saved_raw_path,
            csv_files=csv_files,
            parquet_files=parquet_files,
        )

    def export_combined(
        self,
        datasets: Sequence[ParsedMatchDataset],
        prefix: str = "combined",
    ) -> tuple[ParsedMatchDataset, ExportedFiles]:
        """
        Combines multiple `ParsedMatchDataset` instances into one dataset and exports
        all non-empty tables to `self.combined_dir` (CSV) and
        `self.combined_processed_dir` (Parquet).
        """
        combined_dataset = ParsedMatchDataset.combine(datasets)
        csv_files: dict[str, Path] = {}
        parquet_files: dict[str, Path] = {}

        for table_name, df in combined_dataset.tables().items():
            self._write_table(
                df,
                prefix,
                table_name,
                csv_files,
                parquet_files,
                csv_dir=self.combined_dir,
                parquet_dir=self.combined_processed_dir,
            )

        exported = ExportedFiles(
            raw_json_path=self.combined_dir,
            csv_files=csv_files,
            parquet_files=parquet_files,
        )
        return combined_dataset, exported
