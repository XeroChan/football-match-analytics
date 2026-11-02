"""
events.py
=========
Parses chronological match events (`events` + `home/away.incidentEvents`) and infers
pass receivers.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.parsers.helpers import (
    build_player_lookup,
    cast_int_columns,
    clean_score,
    event_type_id_to_name,
    iter_teams,
    resolve_player_name,
    to_int,
)
from src.parsers.lineups import parse_players
from src.parsers.qualifiers import extract_qualifiers_info

INT_EVENT_COLUMNS: tuple[str, ...] = (
    "match_id",
    "id",
    "event_id",
    "minute",
    "second",
    "expanded_minute",
    "period_value",
    "team_id",
    "player_id",
    "shirt_no",
    "event_type_value",
    "outcome_type_value",
    "card_type_value",
    "related_player_id",
    "event_field",
    "minute_info_minute",
    "minute_info_expanded_minute",
    "minute_info_period",
    "minute_info_extra_time",
    "next_team_id",
    "next_player_id",
    "receiver_id",
)

FLOAT_EVENT_COLUMNS: tuple[str, ...] = ("x", "y", "end_x", "end_y", "pass_length", "pass_angle")

EventLike = Union[dict[str, Any], "_MergedEventView"]


class _MergedEventView:
    """
    Read-through view over a primary `events` dict and its matching `incidentEvents` dict(s).
    Calling `.get(key)` reads `key` from every underlying source dict (so coverage tests only
    record fields that `parse_events` actually queries) and returns the first non-null value.
    """

    def __init__(self, sources: list[dict[str, Any]]) -> None:
        self._sources = [s for s in sources if isinstance(s, dict)]

    def get(self, key: str, default: Any = None) -> Any:
        vals = [s.get(key) for s in self._sources]
        dict_vals = [v for v in vals if isinstance(v, dict)]
        if dict_vals:
            return _MergedEventView(dict_vals)
        list_vals = [v for v in vals if isinstance(v, list)]
        if list_vals:
            return _merge_event_lists(list_vals)
        for v in vals:
            if v is not None:
                return v
        return default


def _merge_event_lists(lists: list[list[Any]]) -> list[Any]:
    """Merges parallel lists (such as `qualifiers`, `satisfiers`, or `satisfiedEventsTypes`) across event sources."""
    non_empty = [lst for lst in lists if lst]
    if not non_empty:
        return []
    if all(isinstance(x, dict) for lst in non_empty for x in lst):
        return [item for lst in non_empty for item in lst]
    return list(dict.fromkeys(item for lst in non_empty for item in lst))


def _as_mapping(value: Any) -> EventLike:
    return value if isinstance(value, (dict, _MergedEventView)) else {}


def enrich_with_pass_receivers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds pass receiver columns (`receiver_id`, `receiver_name`, `receiver_is_first_eleven`)
    for successful pass events by looking at the next chronological player event in the same period.
    """
    if df.empty:
        return df

    df = df.copy()
    on_ball = df.loc[df["player_id"].notna()]
    next_cols = ["team_id", "player_id", "player_name", "is_first_eleven"]
    shifted = on_ball.groupby("period", sort=False, dropna=False)[next_cols].shift(-1)

    df["next_team_id"] = shifted["team_id"]
    df["next_player_id"] = shifted["player_id"]
    df["next_player_name"] = shifted["player_name"]
    df["next_player_is_first_eleven"] = shifted["is_first_eleven"]

    valid_pass_receiver = (
        (df["event_type"] == "Pass")
        & (df["outcome_type"] == "Successful")
        & (df["team_id"] == df["next_team_id"])
        & df["player_id"].notna()
        & df["next_player_id"].notna()
        & (df["player_id"] != df["next_player_id"])
    )

    df["receiver_id"] = np.where(valid_pass_receiver, df["next_player_id"], None)
    df["receiver_name"] = np.where(valid_pass_receiver, df["next_player_name"], None)
    df["receiver_is_first_eleven"] = np.where(
        valid_pass_receiver, df["next_player_is_first_eleven"], None
    )

    cast_int_columns(df, INT_EVENT_COLUMNS)
    for float_col in FLOAT_EVENT_COLUMNS:
        if float_col in df.columns:
            df[float_col] = pd.to_numeric(df[float_col], errors="coerce")

    return df


def parse_events(
    match_data: dict[str, Any],
    players_df: Optional[pd.DataFrame] = None,
    include_raw_json_columns: bool = True,
) -> pd.DataFrame:
    """
    Parses ALL chronological match events from `matchCentreData["events"]` for both teams,
    enriching every row with team/player metadata, flattened qualifiers, and inferred pass
    receiver columns.
    """
    events = match_data.get("events") or []
    if not events:
        return pd.DataFrame()

    if players_df is None:
        players_df = parse_players(match_data)

    match_id = match_data.get("matchId")
    player_name_dict = {
        str(k): str(v)
        for k, v in (match_data.get("playerIdNameDictionary") or {}).items()
    }
    event_type_names = event_type_id_to_name(match_data)

    team_map: dict[Any, dict[str, str]] = {}
    incidents_by_id: dict[Any, list[dict[str, Any]]] = {}
    orphan_incidents: list[dict[str, Any]] = []
    event_ids = {
        ev.get("id") for ev in events if isinstance(ev, dict) and ev.get("id") is not None
    }

    for side, info, team_id_val, team_name_val in iter_teams(match_data):
        if team_id_val is not None:
            team_map[team_id_val] = {"team_name": team_name_val, "team_side": side}
        for inc in info.get("incidentEvents") or []:
            if not isinstance(inc, dict):
                continue
            inc_id = inc.get("id")
            if inc_id in event_ids:
                incidents_by_id.setdefault(inc_id, []).append(inc)
            else:
                orphan_incidents.append(inc)

    all_events: list[tuple[EventLike, bool]] = [
        (
            _MergedEventView([ev, *incidents_by_id[ev.get("id")]])
            if isinstance(ev, dict) and ev.get("id") in incidents_by_id
            else ev,
            isinstance(ev, dict) and ev.get("id") in incidents_by_id,
        )
        for ev in events
    ] + [(inc, True) for inc in orphan_incidents]

    player_meta_map = build_player_lookup(players_df)

    parsed_rows: list[dict[str, Any]] = []
    for idx, (ev, is_incident) in enumerate(all_events):
        ev_type = _as_mapping(ev.get("type"))
        outcome = _as_mapping(ev.get("outcomeType"))
        period = _as_mapping(ev.get("period"))
        card = _as_mapping(ev.get("cardType"))
        minute_info = _as_mapping(ev.get("minuteInfo"))

        team_id = ev.get("teamId")
        team_meta = team_map.get(team_id, {})

        player_id = to_int(ev.get("playerId"))
        p_meta = player_meta_map.get(player_id, {}) if player_id is not None else {}
        ep_name = ev.get("playerName")
        player_name = resolve_player_name(player_id, player_meta_map, player_name_dict) or ep_name

        related_player_id = to_int(ev.get("relatedPlayerId"))

        satisfied_ids = ev.get("satisfiedEventsTypes") or []
        satisfied_names = [
            event_type_names.get(to_int(t), str(t)) for t in satisfied_ids
        ]
        satisfiers = ev.get("satisfiers") or []
        satisfier_names: list[str] = []
        satisfier_ids: list[str] = []
        for s in satisfiers:
            if not isinstance(s, dict):
                continue
            s_name = s.get("name")
            s_id = to_int(s.get("id"))
            if s_name and str(s_name) not in satisfier_names:
                satisfier_names.append(str(s_name))
            if s_id is not None and str(s_id) not in satisfier_ids:
                satisfier_ids.append(str(s_id))

        row: dict[str, Any] = {
            "match_id": match_id,
            "event_index": idx,
            "id": ev.get("id"),
            "event_id": ev.get("eventId"),
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "expanded_minute": ev.get("expandedMinute"),
            "period": period.get("displayName"),
            "period_value": period.get("value"),
            "team_id": team_id,
            "team_name": team_meta.get("team_name"),
            "team_side": team_meta.get("team_side"),
            "player_id": player_id,
            "player_name": player_name,
            "shirt_no": p_meta.get("shirt_no"),
            "position": p_meta.get("position"),
            "is_first_eleven": p_meta.get("is_first_eleven"),
            "event_type": ev_type.get("displayName"),
            "event_type_value": ev_type.get("value"),
            "outcome_type": outcome.get("displayName"),
            "outcome_type_value": outcome.get("value"),
            "is_touch": bool(ev.get("isTouch", False)),
            "is_shot": bool(ev.get("isShot", False)),
            "is_goal": bool(ev.get("isGoal", False)),
            "is_incident": is_incident,
            "score": clean_score(ev.get("score")),
            "x": ev.get("x"),
            "y": ev.get("y"),
            "end_x": ev.get("endX"),
            "end_y": ev.get("endY"),
            "goal_mouth_y": ev.get("goalMouthY"),
            "goal_mouth_z": ev.get("goalMouthZ"),
            "blocked_x": ev.get("blockedX"),
            "blocked_y": ev.get("blockedY"),
            "card_type": card.get("displayName"),
            "card_type_value": card.get("value"),
            "related_player_id": related_player_id,
            "related_player_name": resolve_player_name(
                related_player_id, player_meta_map, player_name_dict
            ),
            "event_field": ev.get("field"),
            "event_text": ev.get("text"),
            "minute_info_minute": to_int(minute_info.get("minute")),
            "minute_info_minute_string": minute_info.get("minuteString"),
            "minute_info_expanded_minute": to_int(minute_info.get("expandedMinute")),
            "minute_info_period": to_int(minute_info.get("period")),
            "minute_info_extra_time": minute_info.get("extraTime"),
            "minute_info_is_end_of_period": minute_info.get("isEndOfPeriod"),
            "minute_info_is_penalty_shootout": minute_info.get("isPenaltyShootOut"),
            "satisfiers": "|".join(satisfier_names),
            "satisfier_ids": "|".join(satisfier_ids),
            "satisfied_event_type_names": "|".join(satisfied_names),
        }
        if include_raw_json_columns:
            row["satisfied_events_types"] = json.dumps(satisfied_ids)

        row.update(
            extract_qualifiers_info(
                ev.get("qualifiers"),
                include_raw_json_columns=include_raw_json_columns,
            )
        )

        # Top-level `relatedEventId` (incident events) fills the gap when the
        # `RelatedEventId` qualifier is absent.
        top_related_event_id = ev.get("relatedEventId")
        if row.get("related_event_id") is None and top_related_event_id is not None:
            row["related_event_id"] = top_related_event_id

        parsed_rows.append(row)

    df = pd.DataFrame(parsed_rows)
    return enrich_with_pass_receivers(df)
