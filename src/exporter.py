"""
exporter.py
===========
Handles persisting raw WhoScored JSON payloads and structured Pandas DataFrames
to disk (CSV and JSON) with clean, deterministic filenames for data exploration.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

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


class MatchDataExporter:
    """
    Exports raw `matchCentreData` JSON and parsed DataFrames (`events`, `players`, `metadata`)
    into an output directory.
    """

    def __init__(self, output_dir: Union[str, Path] = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slugify(value: Any) -> str:
        """Creates a filesystem-safe slug from team names or identifiers."""
        if not value:
            return ""
        text = re.sub(r"[^\w\s-]", "", str(value)).strip()
        return re.sub(r"[-\s]+", "_", text)

    def _build_file_prefix(self, match_data: dict[str, Any]) -> str:
        """
        Builds a human-readable file prefix such as `match_1821123_Bournemouth_vs_Sunderland`
        or falls back to `match_<id>`.
        """
        match_id = match_data.get("matchId") or "unknown"
        home_name = self._slugify((match_data.get("home") or {}).get("name"))
        away_name = self._slugify((match_data.get("away") or {}).get("name"))

        if home_name and away_name:
            return f"match_{match_id}_{home_name}_vs_{away_name}"
        return f"match_{match_id}"

    def save_raw_json(self, match_data: dict[str, Any], prefix: str | None = None) -> Path:
        """
        Saves the complete raw `matchCentreData` dictionary to a UTF-8 JSON file.
        Keeping this backup allows re-running or customizing `MatchDataParser` offline
        without re-scraping WhoScored.
        """
        file_prefix = prefix or self._build_file_prefix(match_data)
        json_path = self.output_dir / f"{file_prefix}_raw.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(match_data, f, ensure_ascii=False, indent=2)
        logger.info("Saved raw match JSON backup to: %s", json_path)
        return json_path

    @staticmethod
    def load_raw_json(json_path: Union[str, Path]) -> dict[str, Any]:
        """Loads a previously saved raw match JSON file from disk."""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def export_all(
        self,
        match_data: dict[str, Any],
        dataset: ParsedMatchDataset,
    ) -> ExportedFiles:
        """
        Saves the raw JSON backup and exports `events`, `players`, and `metadata` CSV files.

        Args:
            match_data: Raw `matchCentreData` dictionary.
            dataset: `ParsedMatchDataset` returned by `MatchDataParser.parse_all()`.

        Returns:
            `ExportedFiles` containing the paths to all created files.
        """
        prefix = self._build_file_prefix(match_data)

        raw_json_path = self.save_raw_json(match_data, prefix=prefix)

        events_csv_path = self.output_dir / f"{prefix}_events.csv"
        players_csv_path = self.output_dir / f"{prefix}_players.csv"
        metadata_csv_path = self.output_dir / f"{prefix}_metadata.csv"

        # Use utf-8-sig so Excel and Pandas both open special player characters (accents, umlauts) cleanly
        dataset.events_df.to_csv(events_csv_path, index=False, encoding="utf-8-sig")
        logger.info(
            "Saved all match events CSV (%d rows, %d columns) to: %s",
            len(dataset.events_df),
            len(dataset.events_df.columns),
            events_csv_path,
        )

        dataset.players_df.to_csv(players_csv_path, index=False, encoding="utf-8-sig")
        logger.info(
            "Saved players/lineups CSV (%d rows, %d columns) to: %s",
            len(dataset.players_df),
            len(dataset.players_df.columns),
            players_csv_path,
        )

        dataset.metadata_df.to_csv(metadata_csv_path, index=False, encoding="utf-8-sig")
        logger.info("Saved match metadata CSV to: %s", metadata_csv_path)

        return ExportedFiles(
            raw_json_path=raw_json_path,
            events_csv_path=events_csv_path,
            players_csv_path=players_csv_path,
            metadata_csv_path=metadata_csv_path,
        )

