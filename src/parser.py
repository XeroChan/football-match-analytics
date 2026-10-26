"""
parser.py
=========
Transforms raw WhoScored `matchCentreData` JSON into clean, flat Pandas DataFrames
tailored for exploratory data analysis (EDA), tactical analytics, and machine learning.

Outputs produced by `MatchDataParser`:
1. `events_df`: Every chronological match event for both teams, enriched with player
   details, inferred pass receivers, and flattened event qualifiers.
2. `players_df`: Full squad/lineup metadata and match summary stats for both Home and
   Away players (Starting XI flag, shirt number, position, age, height, weight, rating).
3. `formations_df`: Team tactical formations and active player pitch slots across match
   minute intervals.
4. `metadata_df`: Single-row DataFrame summarizing match-level info (teams, score,
   venue, referee, attendance, start time, weather, period boundaries).
5. `player_stats_df`: Long-format minute-by-minute player stats (`players[].stats`),
   including the full rating time series.
6. `team_stats_df`: Long-format minute-by-minute team stats (`home/away.stats`).
7. `shot_zones_df`: Shot counts and goals per goal-mouth zone and minute
   (`home/away.shotZones`).
8. `minute_mapping_df`: Mapping of (period, minute) to expanded minute (`expandedMinutes`).
9. `lookups_df`: WhoScored ID -> name dictionaries (`_matchCentreEventTypeJson`,
   `_formationIdNameMappings`).
10. `common_events_df`: Generic flattening of `commonEvents` (empty for most matches;
    the schema is not fixed, so every item is normalized and also kept as raw JSON).

Intentionally skipped: `home/away.incidentEvents` (exact duplicates of `events`),
`stats.minutesWithStats` (index derivable from the stats tables) and the internal
`$idx` / `$len` event fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ParsedMatchDataset:
    """Container holding all structured tables extracted from a single match."""

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


class MatchDataParser:
    """
    Stateless parser that converts WhoScored's nested `matchCentreData` dictionary
    into tabular Pandas DataFrames.
    """

    KEY_VALUE_QUALIFIERS = {
        "Zone": "qualifier_zone",
        "Length": "pass_length",
        "Angle": "pass_angle",
        "PassEndX": "qualifier_pass_end_x",
        "PassEndY": "qualifier_pass_end_y",
        "RelatedEventId": "related_event_id",
        "OppositeRelatedEvent": "opposite_related_event_id",
        "JerseyNumber": "qualifier_jersey_number",
        "FormationSlot": "formation_slot",
    }

    BOOLEAN_FLAG_QUALIFIERS = {
        "KeyPass": "is_key_pass",
        "IntentionalGoalAssist": "is_assist",
        "IntentionalAssist": "is_intentional_assist",
        "Cross": "is_cross",
        "Longball": "is_long_ball",
        "Throughball": "is_through_ball",
        "Chipped": "is_chipped_pass",
        "HeadPass": "is_head_pass",
        "ThrowIn": "is_throw_in",
        "CornerTaken": "is_corner",
        "FreekickTaken": "is_free_kick",  # actual WhoScored spelling
        "FreeKickTaken": "is_free_kick",  # defensive alias
        "GoalKick": "is_goal_kick",
        "BigChance": "is_big_chance",
        "BigChanceCreated": "is_big_chance_created",
        "FastBreak": "is_fast_break",
        "Penalty": "is_penalty",
        "OwnGoal": "is_own_goal",
        "Red": "is_red_card",
        "SecondYellow": "is_second_yellow_card",
        "Yellow": "is_yellow_card",
        "Offensive": "is_offensive",
        "Defensive": "is_defensive",
        "FirstTouch": "is_first_touch",
        "Blocked": "is_blocked",
    }

    BODY_PART_QUALIFIERS = {"RightFoot", "LeftFoot", "Head", "OtherBodyPart"}

    # Periods for which fixed metadata columns are always emitted (1H, 2H, ET1, ET2, PSO).
    METADATA_PERIODS = (1, 2, 3, 4, 5)

    INT_EVENT_COLUMNS = (
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
        "minute_info_extra_time",
        "next_team_id",
        "next_player_id",
        "receiver_id",
    )

    FLOAT_EVENT_COLUMNS = ("x", "y", "end_x", "end_y", "pass_length", "pass_angle")

    def __init__(self, include_raw_json_columns: bool = True) -> None:
        self.include_raw_json_columns = include_raw_json_columns

    def parse_all(self, match_data: dict[str, Any]) -> ParsedMatchDataset:
        """Parses the complete `matchCentreData` payload into structured DataFrames."""
        match_id = match_data.get("matchId")
        metadata_df = self.parse_match_metadata(match_data)
        players_df = self.parse_players(match_data)
        events_df = self.parse_events(match_data, players_df=players_df)
        formations_df = self.parse_formations(match_data, players_df=players_df)

        return ParsedMatchDataset(
            match_id=match_id,
            metadata_df=metadata_df,
            players_df=players_df,
            events_df=events_df,
            formations_df=formations_df,
            player_stats_df=self.parse_player_stats(match_data),
            team_stats_df=self.parse_team_stats(match_data),
            shot_zones_df=self.parse_shot_zones(match_data),
            minute_mapping_df=self.parse_minute_mapping(match_data),
            lookups_df=self.parse_lookups(match_data),
            common_events_df=self.parse_common_events(match_data),
        )

    @staticmethod
    def _clean_score(value: Any) -> Optional[str]:
        """Removes the penalty-shootout marker '*' from WhoScored score strings."""
        return str(value).replace("*", "").strip() if value is not None else None

    @staticmethod
    def _extract_final_rating(stats: Any) -> Optional[float]:
        """Extracts the final recorded player rating from `player['stats']['ratings']`."""
        ratings = stats.get("ratings") if isinstance(stats, dict) else None
        if not isinstance(ratings, dict) or not ratings:
            return None
        try:
            last_key = max(ratings.keys(), key=lambda k: int(k))
            return float(ratings[last_key])
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _display_name(value: Any) -> Optional[str]:
        """Returns `displayName` from WhoScored `{displayName, value}` objects."""
        return value.get("displayName") if isinstance(value, dict) else None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _event_type_id_to_name(match_data: dict[str, Any]) -> dict[int, str]:
        """Inverts `_matchCentreEventTypeJson` ({name: id}) into {id: name}."""
        mapping: dict[int, str] = {}
        for name, type_id in (match_data.get("_matchCentreEventTypeJson") or {}).items():
            try:
                mapping.setdefault(int(type_id), str(name))
            except (TypeError, ValueError):
                continue
        return mapping

    def parse_match_metadata(self, match_data: dict[str, Any]) -> pd.DataFrame:
        """Extracts match-level summary information (teams, score, date, venue, referee)."""
        home = match_data.get("home") or {}
        away = match_data.get("away") or {}
        referee = match_data.get("referee") if isinstance(match_data.get("referee"), dict) else {}

        home_formations = home.get("formations") or []
        away_formations = away.get("formations") or []

        record = {
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
            "score": self._clean_score(match_data.get("score")),
            "ht_score": self._clean_score(match_data.get("htScore")),
            "ft_score": self._clean_score(match_data.get("ftScore")),
            "et_score": self._clean_score(match_data.get("etScore")),
            "pk_score": match_data.get("pkScore"),
            "decided_on_penalties": str(match_data.get("score") or "").strip().startswith("*"),
            "home_team_id": home.get("teamId"),
            "home_team_name": home.get("name"),
            "home_country_name": home.get("countryName"),
            "home_manager_name": home.get("managerName"),
            "home_average_age": home.get("averageAge"),
            "home_starting_formation": (
                home_formations[0].get("formationName") if home_formations else None
            ),
            "away_team_id": away.get("teamId"),
            "away_team_name": away.get("name"),
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

        # `home/away.scores` -> {fulltime, halftime, running}
        for side, team in (("home", home), ("away", away)):
            scores = team.get("scores") if isinstance(team.get("scores"), dict) else {}
            for key in ("halftime", "fulltime", "running"):
                record[f"{side}_score_{key}"] = scores.get(key)
            for key in sorted(set(scores) - {"halftime", "fulltime", "running"}):
                record[f"{side}_score_{key}"] = scores.get(key)

        # `periodMinuteLimits` / `periodEndMinutes` -> fixed per-period columns
        limits = match_data.get("periodMinuteLimits") or {}
        end_minutes = match_data.get("periodEndMinutes") or {}
        periods = sorted(
            set(self.METADATA_PERIODS)
            | {p for p in (self._to_int(k) for k in (*limits, *end_minutes)) if p is not None}
        )
        for p in periods:
            record[f"period_{p}_minute_limit"] = limits.get(str(p), limits.get(p))
            record[f"period_{p}_end_minute"] = end_minutes.get(str(p), end_minutes.get(p))

        return pd.DataFrame([record])

    def parse_players(self, match_data: dict[str, Any]) -> pd.DataFrame:
        """Extracts player roster, lineup status, physical attributes, and match ratings."""
        match_id = match_data.get("matchId")
        rows: list[dict[str, Any]] = []

        for side in ("home", "away"):
            team = match_data.get(side) or {}
            team_id = team.get("teamId")
            team_name = team.get("name")

            for player in team.get("players") or []:
                subbed_in_period = player.get("subbedInPeriod")
                subbed_out_period = player.get("subbedOutPeriod")
                rows.append(
                    {
                        "match_id": match_id,
                        "team_side": side,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player.get("playerId"),
                        "player_name": player.get("name"),
                        "shirt_no": player.get("shirtNo"),
                        "position": player.get("position"),
                        "field": player.get("field"),
                        "is_first_eleven": bool(player.get("isFirstEleven", False)),
                        "is_man_of_the_match": bool(player.get("isManOfTheMatch", False)),
                        "subbed_in_player_id": player.get("subbedInPlayerId"),
                        "subbed_out_player_id": player.get("subbedOutPlayerId"),
                        "subbed_in_expanded_minute": player.get("subbedInExpandedMinute"),
                        "subbed_out_expanded_minute": player.get("subbedOutExpandedMinute"),
                        "subbed_in_period": self._display_name(subbed_in_period),
                        "subbed_in_period_value": (
                            subbed_in_period.get("value")
                            if isinstance(subbed_in_period, dict)
                            else None
                        ),
                        "subbed_out_period": self._display_name(subbed_out_period),
                        "subbed_out_period_value": (
                            subbed_out_period.get("value")
                            if isinstance(subbed_out_period, dict)
                            else None
                        ),
                        "age": player.get("age"),
                        "height_cm": player.get("height"),
                        "weight_kg": player.get("weight"),
                        "final_rating": self._extract_final_rating(player.get("stats")),
                    }
                )

        df = pd.DataFrame(rows)
        for col in ("subbed_in_period_value", "subbed_out_period_value"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df

    def parse_player_stats(self, match_data: dict[str, Any]) -> pd.DataFrame:
        """
        Flattens `players[].stats` ({stat_name: {minute: value}}) into a long table with one
        row per (player, stat, minute). Includes the full `ratings` time series.
        """
        match_id = match_data.get("matchId")
        rows: list[dict[str, Any]] = []

        for side in ("home", "away"):
            team = match_data.get(side) or {}
            for player in team.get("players") or []:
                stats = player.get("stats")
                if not isinstance(stats, dict):
                    continue
                for stat_name, per_minute in stats.items():
                    if not isinstance(per_minute, dict):
                        continue
                    for minute, value in per_minute.items():
                        rows.append(
                            {
                                "match_id": match_id,
                                "team_side": side,
                                "team_id": team.get("teamId"),
                                "team_name": team.get("name"),
                                "player_id": player.get("playerId"),
                                "player_name": player.get("name"),
                                "stat_name": stat_name,
                                "minute": self._to_int(minute),
                                "value": self._to_float(value),
                            }
                        )

        return self._sort_long_table(
            pd.DataFrame(rows), ["team_side", "player_id", "stat_name", "minute"]
        )

    def parse_team_stats(self, match_data: dict[str, Any]) -> pd.DataFrame:
        """
        Flattens `home/away.stats` ({stat_name: {minute: value}}) into a long table with one
        row per (team, stat, minute).
        """
        match_id = match_data.get("matchId")
        rows: list[dict[str, Any]] = []

        for side in ("home", "away"):
            team = match_data.get(side) or {}
            stats = team.get("stats")
            if not isinstance(stats, dict):
                continue
            for stat_name, per_minute in stats.items():
                # `minutesWithStats` is a plain list of minutes (index only) -> skipped.
                if not isinstance(per_minute, dict):
                    continue
                for minute, value in per_minute.items():
                    rows.append(
                        {
                            "match_id": match_id,
                            "team_side": side,
                            "team_id": team.get("teamId"),
                            "team_name": team.get("name"),
                            "stat_name": stat_name,
                            "minute": self._to_int(minute),
                            "value": self._to_float(value),
                        }
                    )

        return self._sort_long_table(pd.DataFrame(rows), ["team_side", "stat_name", "minute"])

    def parse_shot_zones(self, match_data: dict[str, Any]) -> pd.DataFrame:
        """
        Flattens `home/away.shotZones` ({zone: {stats: {minute: {count, goalCount}}}}) into
        one row per (team, zone, minute).
        """
        match_id = match_data.get("matchId")
        rows: list[dict[str, Any]] = []

        for side in ("home", "away"):
            team = match_data.get(side) or {}
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
                            "team_id": team.get("teamId"),
                            "team_name": team.get("name"),
                            "zone": zone_name,
                            "minute": self._to_int(minute),
                            "count": values.get("count"),
                            "goal_count": values.get("goalCount"),
                        }
                    )

        return self._sort_long_table(pd.DataFrame(rows), ["team_side", "zone", "minute"])

    def parse_minute_mapping(self, match_data: dict[str, Any]) -> pd.DataFrame:
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
                        "period_value": self._to_int(period),
                        "minute": self._to_int(minute),
                        "expanded_minute": self._to_int(expanded),
                    }
                )

        return self._sort_long_table(pd.DataFrame(rows), ["period_value", "minute"])

    def parse_lookups(self, match_data: dict[str, Any]) -> pd.DataFrame:
        """
        Exports WhoScored's ID -> name dictionaries as one long lookup table:
        `event_type` (`_matchCentreEventTypeJson`, used by `satisfied_events_types`) and
        `formation` (`_formationIdNameMappings`).
        """
        rows: list[dict[str, Any]] = []

        for name, type_id in (match_data.get("_matchCentreEventTypeJson") or {}).items():
            rows.append({"lookup_type": "event_type", "lookup_id": self._to_int(type_id), "name": name})

        for formation_id, name in (match_data.get("_formationIdNameMappings") or {}).items():
            rows.append(
                {"lookup_type": "formation", "lookup_id": self._to_int(formation_id), "name": name}
            )

        return self._sort_long_table(pd.DataFrame(rows), ["lookup_type", "lookup_id"])

    def parse_common_events(self, match_data: dict[str, Any]) -> pd.DataFrame:
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

    @staticmethod
    def _sort_long_table(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
        """Sorts a long table deterministically and casts integer ID/minute columns."""
        if df.empty:
            return df
        for col in ("match_id", "team_id", "player_id", "minute", "period_value",
                    "expanded_minute", "lookup_id", "count", "goal_count"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    def parse_formations(
        self,
        match_data: dict[str, Any],
        players_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extracts team tactical formations and active pitch coordinates per player slot
        across match minute intervals (`home.formations` and `away.formations`).
        """
        match_id = match_data.get("matchId")
        if players_df is None:
            players_df = self.parse_players(match_data)

        player_names: dict[int, str] = {}
        if not players_df.empty and "player_id" in players_df.columns:
            valid = players_df.dropna(subset=["player_id"])
            player_names = dict(
                zip(valid["player_id"].astype(int), valid["player_name"].astype(str))
            )

        rows: list[dict[str, Any]] = []
        for side in ("home", "away"):
            team = match_data.get(side) or {}
            team_id = team.get("teamId")
            team_name = team.get("name")

            for f_idx, formation in enumerate(team.get("formations") or []):
                positions = formation.get("formationPositions") or []
                player_ids = formation.get("playerIds") or []
                jersey_numbers = formation.get("jerseyNumbers") or []
                slots = formation.get("formationSlots") or []

                for slot_idx, pos in enumerate(positions):
                    pid = player_ids[slot_idx] if slot_idx < len(player_ids) else None
                    rows.append(
                        {
                            "match_id": match_id,
                            "team_side": side,
                            "team_id": team_id,
                            "team_name": team_name,
                            "formation_index": f_idx,
                            "formation_id": formation.get("formationId"),
                            "formation_name": formation.get("formationName"),
                            "period": formation.get("period"),
                            "start_minute_expanded": formation.get("startMinuteExpanded"),
                            "end_minute_expanded": formation.get("endMinuteExpanded"),
                            "captain_player_id": formation.get("captainPlayerId"),
                            "slot_number": (
                                slots[slot_idx] if slot_idx < len(slots) else slot_idx + 1
                            ),
                            "player_id": pid,
                            "player_name": player_names.get(int(pid)) if pid is not None else None,
                            "shirt_no": (
                                jersey_numbers[slot_idx]
                                if slot_idx < len(jersey_numbers)
                                else None
                            ),
                            "horizontal": pos.get("horizontal"),
                            "vertical": pos.get("vertical"),
                        }
                    )

        return pd.DataFrame(rows)

    def _extract_qualifiers_info(self, qualifiers: Any) -> dict[str, Any]:
        """Flattens an event's `qualifiers` list into top-level columns."""
        result: dict[str, Any] = {
            "qualifier_names": "",
            "body_part": None,
            **{col: None for col in self.KEY_VALUE_QUALIFIERS.values()},
            **{col: False for col in self.BOOLEAN_FLAG_QUALIFIERS.values()},
        }
        if self.include_raw_json_columns:
            result["qualifiers_json"] = "{}"

        if not isinstance(qualifiers, list) or not qualifiers:
            return result

        names: list[str] = []
        kv_map: dict[str, Any] = {}

        for q in qualifiers:
            if not isinstance(q, dict):
                continue
            q_type = q.get("type")
            raw_name = q_type.get("displayName") if isinstance(q_type, dict) else None
            if not raw_name:
                continue

            display_name = str(raw_name).strip()
            val = q.get("value", True)
            names.append(display_name)
            kv_map[display_name] = val

            if display_name in self.KEY_VALUE_QUALIFIERS:
                result[self.KEY_VALUE_QUALIFIERS[display_name]] = val
            if display_name in self.BOOLEAN_FLAG_QUALIFIERS:
                result[self.BOOLEAN_FLAG_QUALIFIERS[display_name]] = True
            if display_name in self.BODY_PART_QUALIFIERS:
                result["body_part"] = display_name

        result["qualifier_names"] = "|".join(sorted(set(names)))
        if self.include_raw_json_columns:
            result["qualifiers_json"] = json.dumps(kv_map, ensure_ascii=False)
        return result

    def parse_events(
        self,
        match_data: dict[str, Any],
        players_df: Optional[pd.DataFrame] = None,
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
            players_df = self.parse_players(match_data)

        match_id = match_data.get("matchId")
        player_name_dict = {
            str(k): str(v)
            for k, v in (match_data.get("playerIdNameDictionary") or {}).items()
        }
        event_type_names = self._event_type_id_to_name(match_data)

        team_map: dict[Any, dict[str, str]] = {}
        for side in ("home", "away"):
            info = match_data.get(side) or {}
            if info.get("teamId") is not None:
                team_map[info["teamId"]] = {"team_name": info.get("name"), "team_side": side}

        player_meta_map: dict[int, dict[str, Any]] = {}
        if not players_df.empty and "player_id" in players_df.columns:
            valid_players = players_df.dropna(subset=["player_id"]).copy()
            valid_players["player_id"] = valid_players["player_id"].astype(int)
            player_meta_map = (
                valid_players.drop_duplicates(subset=["player_id"])
                .set_index("player_id")[["player_name", "shirt_no", "position", "is_first_eleven"]]
                .to_dict("index")
            )

        def resolve_player_name(pid: Optional[int]) -> Optional[str]:
            if pid is None:
                return None
            return player_meta_map.get(pid, {}).get("player_name") or player_name_dict.get(str(pid))

        parsed_rows: list[dict[str, Any]] = []
        for idx, ev in enumerate(events):
            ev_type = ev.get("type") if isinstance(ev.get("type"), dict) else {}
            outcome = ev.get("outcomeType") if isinstance(ev.get("outcomeType"), dict) else {}
            period = ev.get("period") if isinstance(ev.get("period"), dict) else {}
            card = ev.get("cardType") if isinstance(ev.get("cardType"), dict) else {}
            minute_info = ev.get("minuteInfo") if isinstance(ev.get("minuteInfo"), dict) else {}

            team_id = ev.get("teamId")
            team_meta = team_map.get(team_id, {})

            raw_pid = ev.get("playerId")
            player_id = int(raw_pid) if raw_pid is not None and pd.notna(raw_pid) else None
            p_meta = player_meta_map.get(player_id, {}) if player_id is not None else {}
            player_name = resolve_player_name(player_id)

            related_player_id = self._to_int(ev.get("relatedPlayerId"))

            satisfied_ids = ev.get("satisfiedEventsTypes") or []
            satisfied_names = [
                event_type_names.get(self._to_int(t), str(t)) for t in satisfied_ids
            ]
            satisfiers = ev.get("satisfiers") or []
            satisfier_names = [
                str(s.get("name")) for s in satisfiers if isinstance(s, dict) and s.get("name")
            ]

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
                "related_player_name": resolve_player_name(related_player_id),
                "event_field": ev.get("field"),
                "event_text": ev.get("text"),
                "minute_info_minute": minute_info.get("minute"),
                "minute_info_minute_string": minute_info.get("minuteString"),
                "minute_info_extra_time": minute_info.get("extraTime"),
                "minute_info_is_end_of_period": minute_info.get("isEndOfPeriod"),
                "satisfiers": "|".join(satisfier_names),
                "satisfied_event_type_names": "|".join(satisfied_names),
            }
            if self.include_raw_json_columns:
                row["satisfied_events_types"] = json.dumps(satisfied_ids)

            row.update(self._extract_qualifiers_info(ev.get("qualifiers")))

            # Top-level `relatedEventId` (incident events) fills the gap when the
            # `RelatedEventId` qualifier is absent.
            if row.get("related_event_id") is None and ev.get("relatedEventId") is not None:
                row["related_event_id"] = ev.get("relatedEventId")

            parsed_rows.append(row)

        df = pd.DataFrame(parsed_rows)
        return self._enrich_with_pass_receivers(df)

    @classmethod
    def _enrich_with_pass_receivers(cls, df: pd.DataFrame) -> pd.DataFrame:
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

        for int_col in cls.INT_EVENT_COLUMNS:
            if int_col in df.columns:
                df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")

        for float_col in cls.FLOAT_EVENT_COLUMNS:
            if float_col in df.columns:
                df[float_col] = pd.to_numeric(df[float_col], errors="coerce")

        return df
