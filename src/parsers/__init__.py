"""
parsers
=======
Domain-specific sub-parsers that transform sections of WhoScored `matchCentreData`
into flat Pandas DataFrames.
"""

from __future__ import annotations

from src.parsers.events import parse_events
from src.parsers.lineups import parse_formations, parse_players
from src.parsers.metadata import (
    parse_common_events,
    parse_lookups,
    parse_match_metadata,
    parse_minute_mapping,
)
from src.parsers.stats import parse_player_stats, parse_shot_zones, parse_team_stats

__all__ = [
    "parse_common_events",
    "parse_events",
    "parse_formations",
    "parse_lookups",
    "parse_match_metadata",
    "parse_minute_mapping",
    "parse_player_stats",
    "parse_players",
    "parse_shot_zones",
    "parse_team_stats",
]
