"""
test_parsers.py
===============
Unit tests for domain parsers in `src.parsers` (event & incident merging, tactical
formations, pass-receiver inference, and `commonEvents` normalization).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.parser import MatchDataParser
from src.parsers import parse_common_events, parse_events, parse_formations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILES = sorted(RAW_DIR.glob("*_raw.json"))


def test_incident_event_list_merge_preserves_all_qualifiers_and_satisfiers() -> None:
    """
    Verifies that merging a primary `events` dict with an `incidentEvents` dict preserves
    qualifiers, satisfiers, and `satisfiedEventsTypes` from both sources without dropping
    elements that share the same list index.
    """
    payload = {
        "matchId": 555,
        "_matchCentreEventTypeJson": {"shotOnTarget": 9, "goal": 16},
        "home": {
            "teamId": 1,
            "name": "Home FC",
            "incidentEvents": [
                {
                    "id": 1001,
                    "eventId": 10,
                    "minute": 12,
                    "teamId": 1,
                    "playerId": 7,
                    "type": {"displayName": "Goal", "value": 16},
                    "outcomeType": {"displayName": "Successful", "value": 1},
                    "period": {"displayName": "FirstHalf", "value": 1},
                    "satisfiedEventsTypes": [9, 16],
                    "satisfiers": [{"id": 99, "name": "goalOpenPlay"}],
                    "qualifiers": [
                        {"type": {"displayName": "BigChance", "value": 214}},
                        {"type": {"displayName": "RelatedEventId", "value": 55}, "value": "9"},
                    ],
                }
            ],
        },
        "away": {"teamId": 2, "name": "Away FC", "incidentEvents": []},
        "events": [
            {
                "id": 1001,
                "eventId": 10,
                "minute": 12,
                "teamId": 1,
                "playerId": 7,
                "type": {"displayName": "Goal", "value": 16},
                "outcomeType": {"displayName": "Successful", "value": 1},
                "period": {"displayName": "FirstHalf", "value": 1},
                "satisfiedEventsTypes": [9],
                "qualifiers": [
                    {"type": {"displayName": "Zone", "value": 56}, "value": "Center"},
                    {"type": {"displayName": "RightFoot", "value": 20}},
                ],
            }
        ],
    }

    events_df = MatchDataParser().parse_events(payload)
    assert len(events_df) == 1
    row = events_df.iloc[0]
    assert bool(row["is_incident"]) is True
    assert row["qualifier_zone"] == "Center"
    assert bool(row["is_big_chance"]) is True
    assert int(row["related_event_id"]) == 9
    assert row["body_part"] == "RightFoot"
    assert set(row["qualifier_names"].split("|")) == {
        "BigChance",
        "RelatedEventId",
        "RightFoot",
        "Zone",
    }
    assert row["satisfiers"] == "goalOpenPlay"
    assert row["satisfier_ids"] == "99"
    assert row["satisfied_event_type_names"] == "shotOnTarget|goal"


@pytest.mark.skipif(not RAW_FILES, reason=f"No raw JSON files in {RAW_DIR}")
def test_formations_preserves_all_on_pitch_and_bench_slots() -> None:
    """
    Checks that `parse_formations` captures every player entry in `formations[].playerIds`
    (both the 11 on-pitch slots with coordinates and the bench slots with `slot_number == 0`).
    """
    raw = json.loads(RAW_FILES[0].read_text(encoding="utf-8"))
    formations_df = parse_formations(raw)

    expected_rows = sum(
        max(
            len(f.get("formationPositions") or []),
            len(f.get("playerIds") or []),
            len(f.get("jerseyNumbers") or []),
            len(f.get("formationSlots") or []),
        )
        for side in ("home", "away")
        for f in (raw.get(side) or {}).get("formations") or []
    )
    assert len(formations_df) == expected_rows

    on_pitch = formations_df[formations_df["slot_number"] > 0]
    bench = formations_df[formations_df["slot_number"] == 0]
    assert not on_pitch.empty
    assert not bench.empty
    assert on_pitch["horizontal"].notna().all()
    assert on_pitch["vertical"].notna().all()
    assert bench["horizontal"].isna().all()
    assert bench["vertical"].isna().all()


def test_pass_receiver_inference() -> None:
    """
    Verifies that `parse_events` links a successful pass to the next chronological
    teammate event in the same period, and does not assign a receiver across teams.
    """
    payload = {
        "matchId": 777,
        "home": {
            "teamId": 10,
            "name": "Home",
            "players": [
                {"playerId": 1, "name": "Alice", "shirtNo": 8, "position": "MC", "isFirstEleven": True},
                {"playerId": 2, "name": "Bob", "shirtNo": 9, "position": "FW", "isFirstEleven": True},
            ],
        },
        "away": {
            "teamId": 20,
            "name": "Away",
            "players": [
                {"playerId": 3, "name": "Charlie", "shirtNo": 4, "position": "DC", "isFirstEleven": True},
            ],
        },
        "events": [
            {
                "id": 1,
                "eventId": 1,
                "minute": 1,
                "teamId": 10,
                "playerId": 1,
                "type": {"displayName": "Pass", "value": 1},
                "outcomeType": {"displayName": "Successful", "value": 1},
                "period": {"displayName": "FirstHalf", "value": 1},
            },
            {
                "id": 2,
                "eventId": 2,
                "minute": 1,
                "teamId": 10,
                "playerId": 2,
                "type": {"displayName": "Pass", "value": 1},
                "outcomeType": {"displayName": "Unsuccessful", "value": 0},
                "period": {"displayName": "FirstHalf", "value": 1},
            },
            {
                "id": 3,
                "eventId": 3,
                "minute": 1,
                "teamId": 20,
                "playerId": 3,
                "type": {"displayName": "Interception", "value": 8},
                "outcomeType": {"displayName": "Successful", "value": 1},
                "period": {"displayName": "FirstHalf", "value": 1},
            },
        ],
    }
    df = parse_events(payload)
    assert int(df.loc[0, "receiver_id"]) == 2
    assert df.loc[0, "receiver_name"] == "Bob"
    assert bool(df.loc[0, "receiver_is_first_eleven"]) is True
    assert pd.isna(df.loc[1, "receiver_id"])
    assert pd.isna(df.loc[1, "receiver_name"])


def test_parse_common_events_flattens_and_preserves_raw_json() -> None:
    """Checks that `parse_common_events` normalizes nested dicts and keeps `raw_json` last."""
    payload = {
        "matchId": 888,
        "commonEvents": [
            {"id": 10, "meta": {"kind": "var", "tags": ["check", "confirmed"]}},
        ],
    }
    df = parse_common_events(payload)
    assert len(df) == 1
    assert df.columns[-1] == "raw_json"
    assert df.loc[0, "meta_kind"] == "var"
    assert json.loads(df.loc[0, "meta_tags"]) == ["check", "confirmed"]
    assert json.loads(df.loc[0, "raw_json"]) == payload["commonEvents"][0]

