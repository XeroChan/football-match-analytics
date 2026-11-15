"""
stats.py
========
Parsers for minute-by-minute player stats (`player_stats_df`), team stats (`team_stats_df`),
and goal-mouth shot zones (`shot_zones_df`).
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Any

import pandas as pd

from src.parsers.helpers import as_dict, iter_teams, sort_long_table, to_float, to_int

# `home/away.stats.minutesWithStats` is a redundant list of minutes that can be derived
# directly from the minute column of `team_stats_df`.
SKIPPED_TEAM_STAT_KEYS = {"minutesWithStats"}


def _flatten_minute_stats(
    stats: Any,
    skip_keys: AbstractSet[str] = frozenset(),
) -> list[tuple[str, int | None, float | None]]:
    """Flattens `{stat_name: {minute: value}}` into `(stat_name, minute, value)` tuples."""
    if not isinstance(stats, dict):
        return []

    rows: list[tuple[str, int | None, float | None]] = []
    for stat_name, per_minute in stats.items():
        if stat_name in skip_keys or not isinstance(per_minute, dict):
            continue
        for minute, value in per_minute.items():
            rows.append((stat_name, to_int(minute), to_float(value)))
    return rows


def parse_player_stats(match_data: dict[str, Any]) -> pd.DataFrame:
    """
    Flattens `players[].stats` ({stat_name: {minute: value}}) into a long table with one
    row per (player, stat, minute). Includes the full `ratings` time series.
    """
    rows: list[dict[str, Any]] = []

    for team, base in iter_teams(match_data):
        for player in team.get("players") or []:
            player_id = player.get("playerId")
            player_name = player.get("name")
            for stat_name, minute, value in _flatten_minute_stats(player.get("stats")):
                rows.append(
                    {
                        **base,
                        "player_id": player_id,
                        "player_name": player_name,
                        "stat_name": stat_name,
                        "minute": minute,
                        "value": value,
                    },
                )

    return sort_long_table(
        pd.DataFrame(rows), ["team_side", "player_id", "stat_name", "minute"],
    )


def parse_team_stats(match_data: dict[str, Any]) -> pd.DataFrame:
    """
    Flattens `home/away.stats` ({stat_name: {minute: value}}) into a long table with one
    row per (team, stat, minute).
    """
    rows: list[dict[str, Any]] = []

    for team, base in iter_teams(match_data):
        for stat_name, minute, value in _flatten_minute_stats(
            team.get("stats"), skip_keys=SKIPPED_TEAM_STAT_KEYS,
        ):
            rows.append({**base, "stat_name": stat_name, "minute": minute, "value": value})

    return sort_long_table(pd.DataFrame(rows), ["team_side", "stat_name", "minute"])


def parse_shot_zones(match_data: dict[str, Any]) -> pd.DataFrame:
    """
    Flattens `home/away.shotZones` ({zone: {stats: {minute: {count, goalCount}}}}) into
    one row per (team, zone, minute).
    """
    rows: list[dict[str, Any]] = []

    for team, base in iter_teams(match_data):
        for zone_name, zone in as_dict(team.get("shotZones")).items():
            for minute, values in as_dict(as_dict(zone).get("stats")).items():
                vals = as_dict(values)
                rows.append(
                    {
                        **base,
                        "zone": zone_name,
                        "minute": to_int(minute),
                        "count": vals.get("count"),
                        "goal_count": vals.get("goalCount"),
                    },
                )

    return sort_long_table(pd.DataFrame(rows), ["team_side", "zone", "minute"])
