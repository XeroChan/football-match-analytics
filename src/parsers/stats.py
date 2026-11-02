"""
stats.py
========
Parsers for minute-by-minute player stats (`player_stats_df`), team stats (`team_stats_df`),
and goal-mouth shot zones (`shot_zones_df`).
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Any, Optional

import pandas as pd

from src.parsers.helpers import iter_teams, sort_long_table, to_float, to_int

# `home/away.stats.minutesWithStats` is a redundant list of minutes that can be derived
# directly from the minute column of `team_stats_df`.
SKIPPED_TEAM_STAT_KEYS = {"minutesWithStats"}


def _flatten_minute_stats(
    stats: Any,
    skip_keys: AbstractSet[str] = frozenset(),
) -> list[tuple[str, Optional[int], Optional[float]]]:
    """Flattens a `{stat_name: {minute: value}}` mapping into `(stat_name, minute, value)` tuples."""
    if not isinstance(stats, dict):
        return []

    rows: list[tuple[str, Optional[int], Optional[float]]] = []
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
    match_id = match_data.get("matchId")
    rows: list[dict[str, Any]] = []

    for side, team, team_id, team_name in iter_teams(match_data):
        for player in team.get("players") or []:
            player_id = player.get("playerId")
            player_name = player.get("name")
            for stat_name, minute, value in _flatten_minute_stats(player.get("stats")):
                rows.append(
                    {
                        "match_id": match_id,
                        "team_side": side,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player_id,
                        "player_name": player_name,
                        "stat_name": stat_name,
                        "minute": minute,
                        "value": value,
                    }
                )

    return sort_long_table(
        pd.DataFrame(rows), ["team_side", "player_id", "stat_name", "minute"]
    )


def parse_team_stats(match_data: dict[str, Any]) -> pd.DataFrame:
    """
    Flattens `home/away.stats` ({stat_name: {minute: value}}) into a long table with one
    row per (team, stat, minute).
    """
    match_id = match_data.get("matchId")
    rows: list[dict[str, Any]] = []

    for side, team, team_id, team_name in iter_teams(match_data):
        for stat_name, minute, value in _flatten_minute_stats(
            team.get("stats"), skip_keys=SKIPPED_TEAM_STAT_KEYS
        ):
            rows.append(
                {
                    "match_id": match_id,
                    "team_side": side,
                    "team_id": team_id,
                    "team_name": team_name,
                    "stat_name": stat_name,
                    "minute": minute,
                    "value": value,
                }
            )

    return sort_long_table(pd.DataFrame(rows), ["team_side", "stat_name", "minute"])


def parse_shot_zones(match_data: dict[str, Any]) -> pd.DataFrame:
    """
    Flattens `home/away.shotZones` ({zone: {stats: {minute: {count, goalCount}}}}) into
    one row per (team, zone, minute).
    """
    match_id = match_data.get("matchId")
    rows: list[dict[str, Any]] = []

    for side, team, team_id, team_name in iter_teams(match_data):
        zones = team.get("shotZones")
        if not isinstance(zones, dict):
            continue
        for zone_name, zone in zones.items():
            per_minute = zone.get("stats") if isinstance(zone, dict) else None
            if not isinstance(per_minute, dict):
                continue
            for minute, values in per_minute.items():
                values = values if isinstance(values, dict) else {}
                rows.append(
                    {
                        "match_id": match_id,
                        "team_side": side,
                        "team_id": team_id,
                        "team_name": team_name,
                        "zone": zone_name,
                        "minute": to_int(minute),
                        "count": values.get("count"),
                        "goal_count": values.get("goalCount"),
                    }
                )

    return sort_long_table(pd.DataFrame(rows), ["team_side", "zone", "minute"])
