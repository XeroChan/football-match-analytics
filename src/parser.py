"""
parser.py
=========
High-level `MatchDataParser` facade that orchestrates all sub-parsers in `src.parsers`
to transform raw WhoScored `matchCentreData` JSON into a `ParsedMatchDataset` of 10
structured Pandas DataFrames.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.models import ParsedMatchDataset
from src.parsers import (
    parse_common_events,
    parse_events,
    parse_formations,
    parse_lookups,
    parse_match_metadata,
    parse_minute_mapping,
    parse_player_stats,
    parse_players,
    parse_shot_zones,
    parse_team_stats,
)

__all__ = ["MatchDataParser", "ParsedMatchDataset"]


class MatchDataParser:
    """
    Stateless parser that converts WhoScored's nested `matchCentreData` dictionary
    into tabular Pandas DataFrames.
    """

    def __init__(self, include_raw_json_columns: bool = True) -> None:
        self.include_raw_json_columns = include_raw_json_columns

    def parse_all(self, match_data: dict[str, Any]) -> ParsedMatchDataset:
        """Parses the complete `matchCentreData` payload into structured DataFrames."""
        players_df = parse_players(match_data)
        return ParsedMatchDataset(
            match_id=match_data.get("matchId"),
            metadata_df=parse_match_metadata(match_data),
            players_df=players_df,
            events_df=self.parse_events(match_data, players_df=players_df),
            formations_df=parse_formations(match_data, players_df=players_df),
            player_stats_df=parse_player_stats(match_data),
            team_stats_df=parse_team_stats(match_data),
            shot_zones_df=parse_shot_zones(match_data),
            minute_mapping_df=parse_minute_mapping(match_data),
            lookups_df=parse_lookups(match_data),
            common_events_df=parse_common_events(match_data),
        )

    def parse_events(
        self,
        match_data: dict[str, Any],
        players_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        return parse_events(
            match_data,
            players_df=players_df,
            include_raw_json_columns=self.include_raw_json_columns,
        )

    parse_formations = staticmethod(parse_formations)
