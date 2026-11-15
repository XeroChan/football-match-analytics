"""
models.py
=========
Data containers for parsed match tables (`ParsedMatchDataset`) and exported file paths
(`ExportedFiles`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def _concat_frames(
    frames: list[pd.DataFrame],
    deduplicate_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Concatenates non-empty DataFrames, optionally deduplicating on key columns."""
    non_empty = [df for df in frames if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    if deduplicate_cols and all(col in combined.columns for col in deduplicate_cols):
        combined = combined.drop_duplicates(
            subset=list(deduplicate_cols),
            inplace=False,
            ignore_index=True,
        )
    return combined


@dataclass
class ParsedMatchDataset:
    """Container holding all structured tables extracted from a single match or combined matches."""

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

    @classmethod
    def combine(cls, datasets: Sequence[ParsedMatchDataset]) -> ParsedMatchDataset:
        """
        Combines a sequence of `ParsedMatchDataset` objects into a single multi-match
        `ParsedMatchDataset` by concatenating each corresponding table.
        """
        if not datasets:
            return cls(
                match_id="combined",
                metadata_df=pd.DataFrame(),
                players_df=pd.DataFrame(),
                events_df=pd.DataFrame(),
            )

        match_ids = [ds.match_id for ds in datasets if ds.match_id is not None]
        return cls(
            match_id=f"combined_{len(datasets)}_matches" if match_ids else "combined",
            metadata_df=_concat_frames([ds.metadata_df for ds in datasets]),
            players_df=_concat_frames([ds.players_df for ds in datasets]),
            events_df=_concat_frames([ds.events_df for ds in datasets]),
            formations_df=_concat_frames([ds.formations_df for ds in datasets]),
            player_stats_df=_concat_frames([ds.player_stats_df for ds in datasets]),
            team_stats_df=_concat_frames([ds.team_stats_df for ds in datasets]),
            shot_zones_df=_concat_frames([ds.shot_zones_df for ds in datasets]),
            minute_mapping_df=_concat_frames([ds.minute_mapping_df for ds in datasets]),
            lookups_df=_concat_frames(
                [ds.lookups_df for ds in datasets],
                deduplicate_cols=("lookup_type", "lookup_id", "name"),
            ),
            common_events_df=_concat_frames([ds.common_events_df for ds in datasets]),
        )


@dataclass
class ExportedFiles:
    """Paths of all files generated for a single match (or combined) export."""

    raw_json_path: Path | None = None
    csv_files: dict[str, Path] = field(default_factory=dict)
