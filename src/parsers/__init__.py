"""
parsers
=======
Domain-specific sub-parsers that transform sections of WhoScored `matchCentreData`
into flat Pandas DataFrames.
"""

from __future__ import annotations

from src.parsers.events import (
    FLOAT_EVENT_COLUMNS,
    INT_EVENT_COLUMNS,
    enrich_with_pass_receivers,
    parse_events,
)
from src.parsers.helpers import (
    build_player_lookup,
    cast_int_columns,
    iter_teams,
    resolve_player_name,
)
from src.parsers.lineups import parse_formations, parse_players
from src.parsers.metadata import (
    METADATA_PERIODS,
    parse_common_events,
    parse_lookups,
    parse_match_metadata,
    parse_minute_mapping,
)
from src.parsers.qualifiers import (
    BODY_PART_QUALIFIERS,
    BOOLEAN_FLAG_QUALIFIERS,
    KEY_VALUE_QUALIFIERS,
    extract_qualifiers_info,
)
from src.parsers.stats import (
    SKIPPED_TEAM_STAT_KEYS,
    parse_player_stats,
    parse_shot_zones,
    parse_team_stats,
)

__all__ = [
    "BODY_PART_QUALIFIERS",
    "BOOLEAN_FLAG_QUALIFIERS",
    "FLOAT_EVENT_COLUMNS",
    "INT_EVENT_COLUMNS",
    "KEY_VALUE_QUALIFIERS",
    "METADATA_PERIODS",
    "SKIPPED_TEAM_STAT_KEYS",
    "build_player_lookup",
    "cast_int_columns",
    "enrich_with_pass_receivers",
    "extract_qualifiers_info",
    "iter_teams",
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
    "resolve_player_name",
]
