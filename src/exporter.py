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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from src.parser import ParsedMatchDataset

logger = logging.getLogger(__name__)


@dataclass
class ExportedFiles:
    """Paths of all files generated for a single match export."""

    raw_json_path: Path
    events_csv_path: Path
    players_csv_path: Path
    metadata_csv_path: Path
    formations_csv_path: Optional[Path] = None
    extra_csv_paths: dict[str, Path] = field(default_factory=dict)
    parquet_files: dict[str, Path] = field(default_factory=dict)


class MatchDataExporter:
    """
    Exports the raw `matchCentreData` JSON to `raw_dir`, Parquet tables to `processed_dir`,
    and CSV tables (`events`, `players`, `formations`, `metadata`, `player_stats`,
    `team_stats`, `shot_zones`, `minute_mapping`, `lookups`, `common_events`) to `output_dir`.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "output",
        raw_dir: Union[str, Path] = "data/raw",
        processed_dir: Union[str, Path] = "data/processed",
        export_parquet: bool = True,
        export_csv: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
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
        """Builds a human-readable file prefix such as `match_1821123_Southampton_vs_Aston_Villa`."""
        match_id = match_data.get("matchId") or "unknown"
        home_name = self._slugify((match_data.get("home") or {}).get("name"))
        away_name = self._slugify((match_data.get("away") or {}).get("name"))

        if home_name and away_name:
            return f"match_{match_id}_{home_name}_vs_{away_name}"
        return f"match_{match_id}"

    def find_cached_raw_json(self, match_id: Any) -> Optional[Path]:
        """Finds an existing raw JSON file in `raw_dir` for the given `match_id`, if present."""
        if not match_id:
            return None
        matches = sorted(self.raw_dir.glob(f"match_{match_id}*_raw.json"))
        return matches[0] if matches else None

    def save_raw_json(self, match_data: dict[str, Any], prefix: str | None = None) -> Path:
        """Saves the complete raw `matchCentreData` dictionary to a UTF-8 JSON file."""
        file_prefix = prefix or self._build_file_prefix(match_data)
        json_path = self.raw_dir / f"{file_prefix}_raw.json"
        if not json_path.exists():
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(match_data, f, ensure_ascii=False, indent=2)
            logger.info("Saved raw match JSON backup to: %s", json_path)
        else:
            logger.debug("Raw match JSON backup already exists at: %s", json_path)
        return json_path

    @staticmethod
    def load_raw_json(json_path: Union[str, Path]) -> dict[str, Any]:
        """Loads a previously saved raw match JSON file from disk."""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_table(self, df: pd.DataFrame, prefix: str, table_name: str, parquet_files: dict[str, Path]) -> Path:
        """Writes a single DataFrame to CSV and/or Parquet."""
        csv_path = self.output_dir / f"{prefix}_{table_name}.csv"
        if self.export_csv and not df.empty:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info(
                "Saved %s CSV (%d rows, %d cols) to: %s",
                table_name,
                len(df),
                len(df.columns),
                csv_path,
            )

        if self.export_parquet and not df.empty:
            parquet_path = self.processed_dir / f"{prefix}_{table_name}.parquet"
            try:
                df.to_parquet(parquet_path, index=False)
                parquet_files[table_name] = parquet_path
                logger.info("Saved %s Parquet to: %s", table_name, parquet_path)
            except Exception as exc:
                logger.warning("Skipping Parquet export for %s (%s)", table_name, exc)

        return csv_path

    def export_all(
        self,
        match_data: dict[str, Any],
        dataset: ParsedMatchDataset,
    ) -> ExportedFiles:
        """Saves the raw JSON backup and exports all structured tables to CSV and Parquet."""
        prefix = self._build_file_prefix(match_data)
        raw_json_path = self.save_raw_json(match_data, prefix=prefix)

        parquet_files: dict[str, Path] = {}
        events_csv_path = self._write_table(dataset.events_df, prefix, "events", parquet_files)
        players_csv_path = self._write_table(dataset.players_df, prefix, "players", parquet_files)
        metadata_csv_path = self._write_table(dataset.metadata_df, prefix, "metadata", parquet_files)
        formations_csv_path = self._write_table(
            dataset.formations_df, prefix, "formations", parquet_files
        )

        extra_tables = {
            "player_stats": dataset.player_stats_df,
            "team_stats": dataset.team_stats_df,
            "shot_zones": dataset.shot_zones_df,
            "minute_mapping": dataset.minute_mapping_df,
            "lookups": dataset.lookups_df,
            "common_events": dataset.common_events_df,
        }
        extra_csv_paths = {
            name: self._write_table(df, prefix, name, parquet_files)
            for name, df in extra_tables.items()
            if not df.empty
        }

        return ExportedFiles(
            raw_json_path=raw_json_path,
            events_csv_path=events_csv_path,
            players_csv_path=players_csv_path,
            metadata_csv_path=metadata_csv_path,
            formations_csv_path=formations_csv_path,
            extra_csv_paths=extra_csv_paths,
            parquet_files=parquet_files,
        )
