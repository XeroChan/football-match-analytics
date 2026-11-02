"""
models.py
=========
Data containers for parsed match tables (`ParsedMatchDataset`) and exported file paths
(`ExportedFiles`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ParsedMatchDataset:
    """Container holding all structured tables extracted from a single match."""

    match_id: Any
    metadata_df: pd.DataFrame
    players_df: pd.DataFrame
    events_df: pd.DataFrame
    formations_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_stats_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_stats_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    shot_zones_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    minute_mapping_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    lookups_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    common_events_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Returns all match tables keyed by their canonical export name."""
        return {
            "events": self.events_df,
            "players": self.players_df,
            "metadata": self.metadata_df,
            "formations": self.formations_df,
            "player_stats": self.player_stats_df,
            "team_stats": self.team_stats_df,
            "shot_zones": self.shot_zones_df,
            "minute_mapping": self.minute_mapping_df,
            "lookups": self.lookups_df,
            "common_events": self.common_events_df,
        }


@dataclass
class ExportedFiles:
    """Paths of all files generated for a single match export."""

    raw_json_path: Path
    csv_files: dict[str, Path] = field(default_factory=dict)
    parquet_files: dict[str, Path] = field(default_factory=dict)

