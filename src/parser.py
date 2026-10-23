"""
parser.py
=========
Transforms raw WhoScored `matchCentreData` JSON into clean, flat Pandas DataFrames
tailored for exploratory data analysis (EDA), tactical analytics, and machine learning.

Outputs produced by `MatchDataParser`:
1. `events_df`: Every chronological match event (passes, shots, tackles, fouls, aerials,
   cards, substitutions, formation changes, etc.) for BOTH teams, enriched with player
   details, inferred pass receivers, and flattened event qualifiers.
2. `players_df`: Full squad/lineup metadata and match summary stats for both Home and
   Away players (Starting XI flag, shirt number, position, age, height, weight, rating, etc.).
3. `metadata_df`: Single-row DataFrame summarizing match-level info (teams, score,
   venue, referee, attendance, start time, weather).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


class MatchDataParser:
    """
    Stateless parser that converts WhoScored's nested `matchCentreData` dictionary
    into tabular Pandas DataFrames.
    """

    # Common Opta/WhoScored qualifier displayNames that are useful as top-level columns in EDA
    KEY_VALUE_QUALIFIERS = {
        "Zone": "qualifier_zone",
        "Length": "pass_length",
        "Angle": "pass_angle",
        "PassEndX": "qualifier_pass_end_x",
        "PassEndY": "qualifier_pass_end_y",
        "RelatedEventId": "related_event_id",
        "OppositeRelatedEvent": "opposite_related_event_id",
        " JerseyNumber": "qualifier_jersey_number",
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
        "FreeKickTaken": "is_free_kick",
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

    def parse_all(self, match_data: dict[str, Any]) -> ParsedMatchDataset:
        """
        Parses the complete `matchCentreData` payload into metadata, players, and events DataFrames.

        Args:
            match_data: Dictionary extracted by `WhoScoredScraper`.

        Returns:
            `ParsedMatchDataset` containing `metadata_df`, `players_df`, and `events_df`.
        """
        match_id = match_data.get("matchId")
        metadata_df = self.parse_match_metadata(match_data)
        players_df = self.parse_players(match_data)
        events_df = self.parse_events(match_data, players_df=players_df)

        return ParsedMatchDataset(
            match_id=match_id,
            metadata_df=metadata_df,
            players_df=players_df,
            events_df=events_df,
        )

    def parse_match_metadata(self, match_data: dict[str, Any]) -> pd.DataFrame:
        """
        Extracts match-level summary information (teams, score, date, venue, referee, etc.).
        """
        home = match_data.get("home") or {}
        away = match_data.get("away") or {}
        referee = match_data.get("referee") or {}

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
            "referee_id": referee.get("officialId") if isinstance(referee, dict) else None,
            "referee_name": referee.get("name") if isinstance(referee, dict) else None,
            "score": match_data.get("score"),
            "ht_score": match_data.get("htScore"),
            "ft_score": match_data.get("ftScore"),
            "et_score": match_data.get("etScore"),
            "pk_score": match_data.get("pkScore"),
            "home_team_id": home.get("teamId"),
            "home_team_name": home.get("name"),
            "home_country_name": home.get("countryName"),
            "home_manager_name": home.get("managerName"),
            "home_starting_formation": (
                home_formations[0].get("formationName") if home_formations else None
            ),
            "away_team_id": away.get("teamId"),
            "away_team_name": away.get("name"),
            "away_country_name": away.get("countryName"),
            "away_manager_name": away.get("managerName"),
            "away_starting_formation": (
                away_formations[0].get("formationName") if away_formations else None
            ),
            "max_minute": match_data.get("maxMinute"),
            "total_events_count": len(match_data.get("events") or []),
        }
        return pd.DataFrame([record])

    def parse_players(self, match_data: dict[str, Any]) -> pd.DataFrame:
        """
        Extracts player roster, lineup status, physical attributes, and match ratings
        for both the `home` and `away` teams.
        """
        match_id = match_data.get("matchId")
        rows: list[dict[str, Any]] = []

        for side in ("home", "away"):
            team_info = match_data.get(side) or {}
            team_id = team_info.get("teamId")
            team_name = team_info.get("name")
            players = team_info.get("players") or []

            for player in players:
                player_id = player.get("playerId")
                stats = player.get("stats") or {}

                # Extract final match rating if available in player['stats']['ratings']
                ratings_dict = stats.get("ratings") if isinstance(stats, dict) else None
                final_rating: Optional[float] = None
                if isinstance(ratings_dict, dict) and ratings_dict:
                    # The last key in ratings_dict represents the final recorded match rating
                    try:
                        last_key = max(ratings_dict.keys(), key=lambda k: int(k))
                        final_rating = float(ratings_dict[last_key])
                    except (ValueError, TypeError):
                        final_rating = None

                rows.append(
                    {
                        "match_id": match_id,
                        "team_side": side,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player_id,
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
                        "age": player.get("age"),
                        "height_cm": player.get("height"),
                        "weight_kg": player.get("weight"),
                        "final_rating": final_rating,
                    }
                )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    @classmethod
    def _extract_qualifiers_info(cls, qualifiers: Any) -> dict[str, Any]:
        """
        Flattens an event's `qualifiers` list into structured columns for easy filtering
        in CSV/Pandas while preserving the complete qualifier dictionary as a JSON string.
        """
        result: dict[str, Any] = {
            "qualifier_names": "",
            "qualifiers_json": "{}",
            "body_part": None,
        }
        for col in cls.KEY_VALUE_QUALIFIERS.values():
            result[col] = None
        for col in cls.BOOLEAN_FLAG_QUALIFIERS.values():
            result[col] = False

        if not isinstance(qualifiers, list) or not qualifiers:
            return result

        names: list[str] = []
        kv_map: dict[str, Any] = {}

        for q in qualifiers:
            if not isinstance(q, dict):
                continue
            q_type = q.get("type") or {}
            display_name = q_type.get("displayName") if isinstance(q_type, dict) else None
            if not display_name:
                continue

            val = q.get("value", True)
            names.append(display_name)
            kv_map[display_name] = val

            if display_name in cls.KEY_VALUE_QUALIFIERS:
                result[cls.KEY_VALUE_QUALIFIERS[display_name]] = val

            if display_name in cls.BOOLEAN_FLAG_QUALIFIERS:
                result[cls.BOOLEAN_FLAG_QUALIFIERS[display_name]] = True

            if display_name in cls.BODY_PART_QUALIFIERS:
                result["body_part"] = display_name

        result["qualifier_names"] = "|".join(sorted(set(names)))
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
        receiver columns (without dropping non-pass events).

        Args:
            match_data: Raw `matchCentreData` dictionary.
            players_df: Optional pre-parsed players DataFrame to avoid recomputing lookups.

        Returns:
            Comprehensive Pandas DataFrame of all match events.
        """
        events = match_data.get("events") or []
        if not events:
            return pd.DataFrame()

        if players_df is None:
            players_df = self.parse_players(match_data)

        match_id = match_data.get("matchId")
        player_name_dict: dict[str, str] = {
            str(k): str(v)
            for k, v in (match_data.get("playerIdNameDictionary") or {}).items()
        }

        # Build lookup maps for team & player attributes
        home_info = match_data.get("home") or {}
        away_info = match_data.get("away") or {}
        team_map: dict[Any, dict[str, Any]] = {}
        if home_info.get("teamId") is not None:
            team_map[home_info["teamId"]] = {
                "team_name": home_info.get("name"),
                "team_side": "home",
            }
        if away_info.get("teamId") is not None:
            team_map[away_info["teamId"]] = {
                "team_name": away_info.get("name"),
                "team_side": "away",
            }

        player_meta_map: dict[int, dict[str, Any]] = {}
        if not players_df.empty:
            for _, prow in players_df.iterrows():
                pid = prow.get("player_id")
                if pd.notna(pid):
                    player_meta_map[int(pid)] = {
                        "player_name": prow.get("player_name"),
                        "shirt_no": prow.get("shirt_no"),
                        "position": prow.get("position"),
                        "is_first_eleven": bool(prow.get("is_first_eleven", False)),
                    }

        parsed_rows: list[dict[str, Any]] = []
        for idx, ev in enumerate(events):
            ev_type_obj = ev.get("type") if isinstance(ev.get("type"), dict) else {}
            outcome_obj = (
                ev.get("outcomeType") if isinstance(ev.get("outcomeType"), dict) else {}
            )
            period_obj = ev.get("period") if isinstance(ev.get("period"), dict) else {}

            team_id = ev.get("teamId")
            team_meta = team_map.get(team_id, {})

            raw_player_id = ev.get("playerId")
            player_id: Optional[int] = (
                int(raw_player_id) if raw_player_id is not None and pd.notna(raw_player_id) else None
            )
            p_meta = player_meta_map.get(player_id, {}) if player_id is not None else {}
            player_name = (
                p_meta.get("player_name")
                or (player_name_dict.get(str(player_id)) if player_id is not None else None)
            )

            qualifiers_info = self._extract_qualifiers_info(ev.get("qualifiers"))

            row: dict[str, Any] = {
                "match_id": match_id,
                "event_index": idx,
                "id": ev.get("id"),
                "event_id": ev.get("eventId"),
                "minute": ev.get("minute"),
                "second": ev.get("second"),
                "expanded_minute": ev.get("expandedMinute"),
                "period": period_obj.get("displayName"),
                "period_value": period_obj.get("value"),
                "team_id": team_id,
                "team_name": team_meta.get("team_name"),
                "team_side": team_meta.get("team_side"),
                "player_id": player_id,
                "player_name": player_name,
                "shirt_no": p_meta.get("shirt_no"),
                "position": p_meta.get("position"),
                "is_first_eleven": p_meta.get("is_first_eleven"),
                "event_type": ev_type_obj.get("displayName"),
                "event_type_value": ev_type_obj.get("value"),
                "outcome_type": outcome_obj.get("displayName"),
                "outcome_type_value": outcome_obj.get("value"),
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
                "card_type": (
                    ev.get("cardType", {}).get("displayName")
                    if isinstance(ev.get("cardType"), dict)
                    else None
                ),
                "satisfied_events_types": json.dumps(ev.get("satisfiedEventsTypes") or []),
            }
            row.update(qualifiers_info)
            parsed_rows.append(row)

        df = pd.DataFrame(parsed_rows)

        # Infer pass receiver from the next chronological player action without dropping non-player events
        df = self._enrich_with_pass_receivers(df)
        return df

    @staticmethod
    def _enrich_with_pass_receivers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds pass receiver columns (`receiver_id`, `receiver_name`, `receiver_is_first_eleven`)
        for successful pass events by looking at the next chronological player event in the same period,
        while preserving ALL events in the DataFrame.
        """
        if df.empty:
            return df

        df = df.copy()

        # Identify rows that have a valid player_id (on-ball/player actions)
        player_events_mask = df["player_id"].notna()
        player_events = df.loc[
            player_events_mask,
            ["team_id", "period", "player_id", "player_name", "is_first_eleven"],
        ].copy()

        # Shift(-1) across player events to find the next player action
        player_events["next_team_id"] = player_events["team_id"].shift(-1)
        player_events["next_period"] = player_events["period"].shift(-1)
        player_events["next_player_id"] = player_events["player_id"].shift(-1)
        player_events["next_player_name"] = player_events["player_name"].shift(-1)
        player_events["next_player_is_first_eleven"] = player_events["is_first_eleven"].shift(-1)

        # Merge next-player context back onto the full events table
        for col in (
            "next_team_id",
            "next_period",
            "next_player_id",
            "next_player_name",
            "next_player_is_first_eleven",
        ):
            df[col] = np.nan
            df[col] = df[col].astype(object)
            df.loc[player_events.index, col] = player_events[col]

        # A pass has a valid teammate receiver when:
        # 1. event_type == "Pass" and outcome_type == "Successful"
        # 2. The next player event is by the same team in the same period
        # 3. The receiver is a different player from the passer
        valid_pass_receiver_mask = (
            (df["event_type"] == "Pass")
            & (df["outcome_type"] == "Successful")
            & (df["team_id"] == df["next_team_id"])
            & (df["period"] == df["next_period"])
            & (df["player_id"].notna())
            & (df["next_player_id"].notna())
            & (df["player_id"] != df["next_player_id"])
        )

        df["receiver_id"] = np.where(valid_pass_receiver_mask, df["next_player_id"], None)
        df["receiver_name"] = np.where(valid_pass_receiver_mask, df["next_player_name"], None)
        df["receiver_is_first_eleven"] = np.where(
            valid_pass_receiver_mask, df["next_player_is_first_eleven"], None
        )

        # Drop internal helper column `next_period`
        df = df.drop(columns=["next_period"])

        # Convert numeric ID columns to nullable Int64 so CSV exports integers instead of '12345.0'
        for int_col in (
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
            "next_team_id",
            "next_player_id",
            "receiver_id",
        ):
            if int_col in df.columns:
                df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")

        for float_col in ("x", "y", "end_x", "end_y", "pass_length", "pass_angle"):
            if float_col in df.columns:
                df[float_col] = pd.to_numeric(df[float_col], errors="coerce")

        return df

