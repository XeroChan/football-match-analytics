"""
metadata.py
===========
Parsers for match-level metadata (`metadata_df`), period/minute mappings
(`minute_mapping_df`), lookup dictionaries (`lookups_df`), and `commonEvents`
(`common_events_df`).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.parsers.helpers import clean_score, sort_long_table, to_int

# Periods for which fixed metadata columns are always emitted (1H, 2H, ET1, ET2, PSO).
METADATA_PERIODS = (1, 2, 3, 4, 5)


def parse_match_metadata(match_data: dict[str, Any]) -> pd.DataFrame:
    """Extracts match-level summary information (teams, score, date, venue, referee)."""
    home = match_data.get("home") or {}
    away = match_data.get("away") or {}
    raw_referee = match_data.get("referee")
    referee = raw_referee if isinstance(raw_referee, dict) else {}

    home_formations = home.get("formations") or []
    away_formations = away.get("formations") or []

    record: dict[str, Any] = {
        "match_id": match_data.get("matchId"),
        "source_url": match_data.get("_sourceUrl"),
        "start_time": match_data.get("startTime"),
        "start_date": match_data.get("startDate"),
        "time_stamp": match_data.get("timeStamp"),
        "status_code": match_data.get("statusCode"),
        "elapsed": match_data.get("elapsed"),
        "venue_name": match_data.get("venueName"),
        "attendance": match_data.get("attendance"),
        "weather_code": match_data.get("weatherCode"),
        "referee_id": referee.get("officialId"),
        "referee_name": referee.get("name"),
        "referee_first_name": referee.get("firstName"),
        "referee_last_name": referee.get("lastName"),
        "referee_has_participated_matches": referee.get("hasParticipatedMatches"),
        "score": clean_score(match_data.get("score")),
        "ht_score": clean_score(match_data.get("htScore")),
        "ft_score": clean_score(match_data.get("ftScore")),
        "et_score": clean_score(match_data.get("etScore")),
        "pk_score": match_data.get("pkScore"),
        "decided_on_penalties": str(match_data.get("score") or "").strip().startswith("*"),
        "home_team_id": home.get("teamId"),
        "home_team_name": home.get("name"),
        "home_field": home.get("field"),
        "home_country_name": home.get("countryName"),
        "home_manager_name": home.get("managerName"),
        "home_average_age": home.get("averageAge"),
        "home_starting_formation": (
            home_formations[0].get("formationName") if home_formations else None
        ),
        "away_team_id": away.get("teamId"),
        "away_team_name": away.get("name"),
        "away_field": away.get("field"),
        "away_country_name": away.get("countryName"),
        "away_manager_name": away.get("managerName"),
        "away_average_age": away.get("averageAge"),
        "away_starting_formation": (
            away_formations[0].get("formationName") if away_formations else None
        ),
        "max_minute": match_data.get("maxMinute"),
        "expanded_max_minute": match_data.get("expandedMaxMinute"),
        "minute_expanded": match_data.get("minuteExpanded"),
        "max_period": match_data.get("maxPeriod"),
        "period_code": match_data.get("periodCode"),
        "timeout_in_seconds": match_data.get("timeoutInSeconds"),
        "total_events_count": len(match_data.get("events") or []),
    }

    # `home/away.scores` -> {fulltime, halftime, running, ...}
    for side, team in (("home", home), ("away", away)):
        raw_scores = team.get("scores")
        scores = raw_scores if isinstance(raw_scores, dict) else {}
        for key in ("halftime", "fulltime", "running"):
            record[f"{side}_score_{key}"] = scores.get(key)
        for key in sorted(set(scores.keys()) - {"halftime", "fulltime", "running"}):
            record[f"{side}_score_{key}"] = scores.get(key)

    # `periodMinuteLimits` / `periodEndMinutes` -> fixed per-period columns
    limits = match_data.get("periodMinuteLimits") or {}
    end_minutes = match_data.get("periodEndMinutes") or {}
    periods = sorted(
        set(METADATA_PERIODS)
        | {p for p in (to_int(k) for k in (*limits.keys(), *end_minutes.keys())) if p is not None}
    )
    for p in periods:
        record[f"period_{p}_minute_limit"] = limits.get(str(p), limits.get(p))
        record[f"period_{p}_end_minute"] = end_minutes.get(str(p), end_minutes.get(p))

    return pd.DataFrame([record])


def parse_minute_mapping(match_data: dict[str, Any]) -> pd.DataFrame:
    """Flattens `expandedMinutes` ({period: {minute: expanded_minute}})."""
    match_id = match_data.get("matchId")
    rows: list[dict[str, Any]] = []

    for period, minutes in (match_data.get("expandedMinutes") or {}).items():
        if not isinstance(minutes, dict):
            continue
        for minute, expanded in minutes.items():
            rows.append(
                {
                    "match_id": match_id,
                    "period_value": to_int(period),
                    "minute": to_int(minute),
                    "expanded_minute": to_int(expanded),
                }
            )

    return sort_long_table(pd.DataFrame(rows), ["period_value", "minute"])


def parse_lookups(match_data: dict[str, Any]) -> pd.DataFrame:
    """
    Exports WhoScored's ID -> name dictionaries as one long lookup table:
    `event_type` (`_matchCentreEventTypeJson`, used by `satisfied_events_types`) and
    `formation` (`_formationIdNameMappings`).
    """
    rows: list[dict[str, Any]] = []

    for name, type_id in (match_data.get("_matchCentreEventTypeJson") or {}).items():
        rows.append({"lookup_type": "event_type", "lookup_id": to_int(type_id), "name": name})

    for formation_id, name in (match_data.get("_formationIdNameMappings") or {}).items():
        rows.append(
            {"lookup_type": "formation", "lookup_id": to_int(formation_id), "name": name}
        )

    return sort_long_table(pd.DataFrame(rows), ["lookup_type", "lookup_id"])


def parse_common_events(match_data: dict[str, Any]) -> pd.DataFrame:
    """
    Flattens `commonEvents`. The list is empty for most matches and its item schema
    is not fixed, so dict items are normalized with `pd.json_normalize` (nested keys
    joined with `_`, lists kept as JSON strings) and every item is also preserved
    verbatim in `raw_json` so no information is lost.
    """
    items = match_data.get("commonEvents") or []
    if not isinstance(items, list) or not items:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        row: dict[str, Any] = {"match_id": match_data.get("matchId"), "common_event_index": idx}
        if isinstance(item, dict):
            flat = pd.json_normalize(item, sep="_").iloc[0].to_dict() if item else {}
            for key, value in flat.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                row[key if key not in row else f"item_{key}"] = value
        else:
            row["value"] = item
        row["raw_json"] = json.dumps(item, ensure_ascii=False)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[[c for c in df.columns if c != "raw_json"] + ["raw_json"]]

