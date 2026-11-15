"""
parser.py
=========
Orchestrates all sub-parsers in `src.parsers` to transform raw WhoScored
`matchCentreData` JSON into a `ParsedMatchDataset` of 10 structured Pandas DataFrames.
"""

from __future__ import annotations

from typing import Any

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

__all__ = ["ParsedMatchDataset", "parse_match"]


def parse_match(match_data: dict[str, Any]) -> ParsedMatchDataset:
    """Parses the complete `matchCentreData` payload into structured DataFrames."""
    players_df = parse_players(match_data)
    return ParsedMatchDataset(
        match_id=match_data.get("matchId"),
        metadata_df=parse_match_metadata(match_data),
        players_df=players_df,
        events_df=parse_events(match_data, players_df=players_df),
        formations_df=parse_formations(match_data, players_df=players_df),
        player_stats_df=parse_player_stats(match_data),
        team_stats_df=parse_team_stats(match_data),
        shot_zones_df=parse_shot_zones(match_data),
        minute_mapping_df=parse_minute_mapping(match_data),
        lookups_df=parse_lookups(match_data),
        common_events_df=parse_common_events(match_data),
    )
