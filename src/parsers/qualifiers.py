"""
qualifiers.py
=============
Qualifier mapping tables and flattening logic for WhoScored event `qualifiers` lists.
"""

from __future__ import annotations

import json
from typing import Any

from src.parsers.helpers import to_int

KEY_VALUE_QUALIFIERS: dict[str, str] = {
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

BOOLEAN_FLAG_QUALIFIERS: dict[str, str] = {
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

BODY_PART_QUALIFIERS: set[str] = {"RightFoot", "LeftFoot", "Head", "OtherBodyPart"}


def extract_qualifiers_info(
    qualifiers: Any,
    include_raw_json_columns: bool = True,
) -> dict[str, Any]:
    """Flattens an event's `qualifiers` list into top-level columns."""
    result: dict[str, Any] = {
        "qualifier_names": "",
        "qualifier_type_ids": "",
        "body_part": None,
        **{col: None for col in KEY_VALUE_QUALIFIERS.values()},
        **{col: False for col in BOOLEAN_FLAG_QUALIFIERS.values()},
    }
    if include_raw_json_columns:
        result["qualifiers_json"] = "{}"

    if not isinstance(qualifiers, list) or not qualifiers:
        return result

    names: list[str] = []
    type_ids: list[int] = []
    kv_map: dict[str, Any] = {}

    for q in qualifiers:
        if not isinstance(q, dict):
            continue
        raw_type = q.get("type")
        q_type = raw_type if isinstance(raw_type, dict) else {}
        raw_name = q_type.get("displayName")
        raw_type_id = to_int(q_type.get("value"))
        if raw_type_id is not None:
            type_ids.append(raw_type_id)
        if not raw_name:
            continue

        display_name_str = str(raw_name).strip()
        val = q.get("value", True)
        names.append(display_name_str)
        if display_name_str not in kv_map or kv_map[display_name_str] is True:
            kv_map[display_name_str] = val

        if display_name_str in KEY_VALUE_QUALIFIERS:
            col = KEY_VALUE_QUALIFIERS[display_name_str]
            if result[col] is None and val is not None:
                result[col] = val
        if display_name_str in BOOLEAN_FLAG_QUALIFIERS:
            result[BOOLEAN_FLAG_QUALIFIERS[display_name_str]] = True
        if display_name_str in BODY_PART_QUALIFIERS and result["body_part"] is None:
            result["body_part"] = display_name_str

    result["qualifier_names"] = "|".join(sorted(set(names)))
    result["qualifier_type_ids"] = "|".join(str(tid) for tid in sorted(set(type_ids)))
    if include_raw_json_columns:
        result["qualifiers_json"] = json.dumps(kv_map, ensure_ascii=False)
    return result

