"""
lineups.py
==========
Parsers for player rosters/lineups (`players_df`) and tactical formations (`formations_df`).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.parsers.helpers import (
    as_dict,
    build_player_lookup,
    cast_int_columns,
    extract_final_rating,
    iter_teams,
    resolve_player_name,
    to_int,
)


def _at(seq: list[Any], idx: int, default: Any = None) -> Any:
    return seq[idx] if idx < len(seq) else default


def parse_players(match_data: dict[str, Any]) -> pd.DataFrame:
    """Extracts player roster, lineup status, physical attributes, and match ratings."""
    rows: list[dict[str, Any]] = []

    for team, base in iter_teams(match_data):
        for player in team.get("players") or []:
            subbed_in = as_dict(player.get("subbedInPeriod"))
            subbed_out = as_dict(player.get("subbedOutPeriod"))
            rows.append(
                {
                    **base,
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
                    "subbed_in_period": subbed_in.get("displayName"),
                    "subbed_in_period_value": subbed_in.get("value"),
                    "subbed_out_period": subbed_out.get("displayName"),
                    "subbed_out_period_value": subbed_out.get("value"),
                    "age": player.get("age"),
                    "height_cm": player.get("height"),
                    "weight_kg": player.get("weight"),
                    "final_rating": extract_final_rating(player.get("stats")),
                },
            )

    return cast_int_columns(
        pd.DataFrame(rows), ("subbed_in_period_value", "subbed_out_period_value"),
    )


def parse_formations(
    match_data: dict[str, Any],
    players_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Extracts team tactical formations and active pitch coordinates per player slot
    across match minute intervals (`home.formations` and `away.formations`), including
    both on-pitch slots (`slot_number` 1..11) and bench slots (`slot_number` 0).
    """
    player_meta = build_player_lookup(players_df)
    rows: list[dict[str, Any]] = []

    for team, base in iter_teams(match_data):
        for f_idx, formation in enumerate(team.get("formations") or []):
            positions = formation.get("formationPositions") or []
            player_ids = formation.get("playerIds") or []
            jersey_numbers = formation.get("jerseyNumbers") or []
            slots = formation.get("formationSlots") or []
            sub_on_pid = to_int(formation.get("subOnPlayerId"))
            sub_off_pid = to_int(formation.get("subOffPlayerId"))

            total_entries = max(
                len(positions), len(player_ids), len(jersey_numbers), len(slots),
            )
            for slot_idx in range(total_entries):
                pos_dict = as_dict(_at(positions, slot_idx))
                pid = to_int(_at(player_ids, slot_idx))
                default_slot = slot_idx + 1 if slot_idx < len(positions) else 0
                rows.append(
                    {
                        **base,
                        "formation_index": f_idx,
                        "formation_id": formation.get("formationId"),
                        "formation_name": formation.get("formationName"),
                        "period": formation.get("period"),
                        "start_minute_expanded": formation.get("startMinuteExpanded"),
                        "end_minute_expanded": formation.get("endMinuteExpanded"),
                        "captain_player_id": formation.get("captainPlayerId"),
                        "sub_on_player_id": sub_on_pid,
                        "sub_on_player_name": resolve_player_name(sub_on_pid, player_meta),
                        "sub_off_player_id": sub_off_pid,
                        "sub_off_player_name": resolve_player_name(sub_off_pid, player_meta),
                        "slot_number": _at(slots, slot_idx, default_slot),
                        "player_id": pid,
                        "player_name": resolve_player_name(pid, player_meta),
                        "shirt_no": _at(jersey_numbers, slot_idx),
                        "horizontal": pos_dict.get("horizontal"),
                        "vertical": pos_dict.get("vertical"),
                    },
                )

    return pd.DataFrame(rows)
