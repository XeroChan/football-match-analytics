"""
helpers.py
==========
Shared type-coercion, formatting, player lookup, and DataFrame sorting helpers
used across sub-parsers.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import pandas as pd

LONG_TABLE_INT_COLS = (
    "match_id",
    "team_id",
    "player_id",
    "minute",
    "period_value",
    "expanded_minute",
    "lookup_id",
    "count",
    "goal_count",
)


def clean_score(value: Any) -> str | None:
    """Removes the penalty-shootout marker '*' from WhoScored score strings."""
    return str(value).replace("*", "").strip() if value is not None else None


def display_name(value: Any) -> str | None:
    """Returns `displayName` from WhoScored `{displayName, value}` objects."""
    return value.get("displayName") if isinstance(value, dict) else None


def to_int(value: Any) -> int | None:
    try:
        return int(value) if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    try:
        return float(value) if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def iter_teams(
    match_data: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any], Any, Any]]:
    """Yields `(side, team_dict, team_id, team_name)` for `'home'` and `'away'`."""
    for side in ("home", "away"):
        raw_team = match_data.get(side)
        team = raw_team if isinstance(raw_team, dict) else {}
        yield side, team, team.get("teamId"), team.get("name")


def build_player_lookup(players_df: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    """Builds a `{player_id: row_dict}` lookup map from `players_df`."""
    if players_df is None or players_df.empty or "player_id" not in players_df.columns:
        return {}
    valid = players_df.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"]).copy()
    valid["player_id"] = valid["player_id"].astype(int)
    cols = [
        c
        for c in ("player_name", "shirt_no", "position", "is_first_eleven")
        if c in valid.columns
    ]
    raw_lookup = valid.set_index("player_id")[cols].to_dict("index")
    return {int(str(k)): {str(ck): cv for ck, cv in v.items()} for k, v in raw_lookup.items()}


def resolve_player_name(
    player_id: int | None,
    player_meta_map: dict[int, dict[str, Any]],
    fallback_names: dict[str, str] | None = None,
) -> str | None:
    """Resolves a player's display name from `player_meta_map` or `playerIdNameDictionary`."""
    if player_id is None:
        return None
    name = player_meta_map.get(player_id, {}).get("player_name")
    if name is not None and pd.notna(name):
        return str(name)
    if fallback_names is not None:
        return fallback_names.get(str(player_id))
    return None


def extract_final_rating(stats: Any) -> float | None:
    """Extracts the final recorded player rating from `player['stats']['ratings']`."""
    ratings = stats.get("ratings") if isinstance(stats, dict) else None
    if not isinstance(ratings, dict) or not ratings:
        return None
    try:
        last_key = max(ratings.keys(), key=int)
        return float(ratings[last_key])
    except (ValueError, TypeError):
        return None


def event_type_id_to_name(match_data: dict[str, Any]) -> dict[int, str]:
    """Inverts `_matchCentreEventTypeJson` ({name: id}) into {id: name}."""
    mapping: dict[int, str] = {}
    for name, type_id in (match_data.get("_matchCentreEventTypeJson") or {}).items():
        parsed_id = to_int(type_id)
        if parsed_id is not None:
            mapping.setdefault(parsed_id, str(name))
    return mapping


def cast_int_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Casts present columns in `df` to Pandas nullable `Int64` in place and returns `df`."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def sort_long_table(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    """Sorts a long table deterministically and casts integer ID/minute columns."""
    if df.empty:
        return df
    cast_int_columns(df, LONG_TABLE_INT_COLS)
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
