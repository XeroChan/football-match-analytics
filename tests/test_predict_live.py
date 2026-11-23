"""
test_predict_live.py
====================
Unit tests for ``src.predict_live`` verifying URL parsing, live window extraction,
incomplete-minute dropping, halftone isolation, scaling + Platt calibration,
and integration on real raw JSON files.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from src.lstm_data import FEATURE_NAMES
from src.predict_live import (
    calibrate_logit,
    extract_latest_windows,
    extract_match_id_from_url,
    load_lstm_checkpoint,
    predict_match_history,
    predict_match_snapshot,
    resolve_default_model_path,
)
from tests.conftest import RAW_DIR, RAW_FILES


def _make_raw_match_payload(
    home_minutes: list[tuple[int, int]],
    away_minutes: list[tuple[int, int]],
    include_stats: bool = True,
) -> dict[str, Any]:
    """
    Builds a minimal WhoScored ``matchCentreData`` dict for testing live prediction.

    Args:
        home_minutes: List of ``(period_value, expanded_minute)`` pairs for Home events.
        away_minutes: List of ``(period_value, expanded_minute)`` pairs for Away events.
        include_stats: When ``False``, omits team ``stats`` to test early-match empty stats.
    """
    events: list[dict[str, Any]] = []
    event_id = 1
    for team_id, pairs in ((100, home_minutes), (200, away_minutes)):
        for period_val, minute in pairs:
            events.append({
                "id": event_id,
                "eventId": event_id,
                "minute": minute,
                "second": 15,
                "expandedMinute": minute,
                "period": {
                    "displayName": "FirstHalf" if period_val == 1 else "SecondHalf",
                    "value": period_val,
                },
                "teamId": team_id,
                "playerId": team_id + 1,
                "type": {"displayName": "Pass", "value": 1},
                "outcomeType": {"displayName": "Successful", "value": 1},
                "isTouch": True,
                "isShot": False,
                "isGoal": False,
                "x": 85.0 if team_id == 100 else 45.0,
                "y": 50.0,
                "endX": 88.0 if team_id == 100 else 48.0,
                "endY": 50.0,
                "qualifiers": [],
                "satisfiedEventsTypes": [],
            })
            event_id += 1

    home_stats = {"possession": {"0": 6, "5": 8}, "clearances": {"1": 1}} if include_stats else {}
    away_stats = {"possession": {"0": 4, "5": 2}, "clearances": {"1": 3}} if include_stats else {}

    return {
        "matchId": 999001,
        "home": {
            "teamId": 100,
            "name": "Home FC",
            "field": "home",
            "stats": home_stats,
            "players": [
                {
                    "playerId": 101,
                    "name": "H Player",
                    "shirtNo": 9,
                    "position": "FW",
                    "isFirstEleven": True,
                },
            ],
            "incidentEvents": [],
            "formations": [],
        },
        "away": {
            "teamId": 200,
            "name": "Away FC",
            "field": "away",
            "stats": away_stats,
            "players": [
                {
                    "playerId": 201,
                    "name": "A Player",
                    "shirtNo": 10,
                    "position": "FW",
                    "isFirstEleven": True,
                },
            ],
            "incidentEvents": [],
            "formations": [],
        },
        "events": events,
        "playerIdNameDictionary": {"101": "H Player", "201": "A Player"},
        "_matchCentreEventTypeJson": {},
    }


def _dummy_ckpt(seq_len: int = 10, horizon: int = 10) -> dict[str, Any]:
    """Returns a deterministic checkpoint dictionary for unit tests."""
    n_feat = len(FEATURE_NAMES)
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": np.zeros(n_feat, dtype=np.float32),
        "feature_std": np.ones(n_feat, dtype=np.float32),
        "sequence_length": seq_len,
        "horizon_minutes": horizon,
        "platt_a": 1.5,
        "platt_b": -1.8,
    }


def test_extract_match_id_from_url_variants() -> None:
    """Verifies match ID extraction from full WhoScored URLs, plain IDs, and invalid strings."""
    assert extract_match_id_from_url(
        "https://www.whoscored.com/matches/1821123/live/england-premier-league-southampton-aston-villa",
    ) == 1821123
    assert extract_match_id_from_url("https://www.whoscored.com/Matches/1729376/Live") == 1729376
    assert extract_match_id_from_url("  1821123  ") == 1821123

    with pytest.raises(ValueError, match="match ID"):
        extract_match_id_from_url("https://www.whoscored.com/teams/13")


def test_extract_latest_windows_drops_incomplete_minute_and_respects_half() -> None:
    """
    Verifies that ``drop_incomplete_minute=True`` excludes minute ``t_now``,
    that Period 2 windows stay inside Period 2, and that < L minutes raises ``ValueError``.
    """
    p1 = [(1, 0), (1, 45)]
    p2 = [(2, 46), (2, 60)]
    raw = _make_raw_match_payload(home_minutes=[*p1, *p2], away_minutes=[*p1, *p2])

    win_drop = extract_latest_windows(
        raw, FEATURE_NAMES, sequence_length=10, horizon_minutes=10, drop_incomplete_minute=True,
    )
    assert win_drop["home"].period_value == 2
    assert win_drop["home"].window_start_minute == 50
    assert win_drop["home"].window_end_minute == 59

    win_keep = extract_latest_windows(
        raw, FEATURE_NAMES, sequence_length=10, horizon_minutes=10, drop_incomplete_minute=False,
    )
    assert win_keep["home"].window_start_minute == 51
    assert win_keep["home"].window_end_minute == 60

    # Early Period 2 (only minutes 46..52 -> 7 rows): dropping current minute leaves 6 < 10.
    short_p2 = [(1, 0), (1, 45), (2, 46), (2, 52)]
    raw_short = _make_raw_match_payload(home_minutes=short_p2, away_minutes=short_p2)
    with pytest.raises(ValueError, match="Not enough completed minutes"):
        extract_latest_windows(
            raw_short, FEATURE_NAMES, sequence_length=10, drop_incomplete_minute=True,
        )


def test_extract_latest_windows_handles_empty_team_stats() -> None:
    """Verifies extraction still works when ``team_stats`` is completely empty early on."""
    raw = _make_raw_match_payload(
        home_minutes=[(1, 0), (1, 15)],
        away_minutes=[(1, 0), (1, 15)],
        include_stats=False,
    )
    windows = extract_latest_windows(
        raw, FEATURE_NAMES, sequence_length=10, drop_incomplete_minute=False,
    )
    assert windows["home"].raw_features.shape == (10, len(FEATURE_NAMES))
    # Default possession share is 0.5 when no possession counts exist.
    poss_idx = FEATURE_NAMES.index("poss_share_cum")
    assert float(windows["home"].raw_features[-1, poss_idx]) == pytest.approx(0.5)


def test_predict_match_snapshot_scaling_and_platt_calibration() -> None:
    """
    Verifies z-score scaling, [-5, 5] clipping, Platt calibration math,
    and combined any-goal probability.
    """
    raw = _make_raw_match_payload(
        home_minutes=[(1, 0), (1, 14)],
        away_minutes=[(1, 0), (1, 14)],
    )
    ckpt = _dummy_ckpt(seq_len=10, horizon=10)
    # Force large feature mean offset so clipping at +/-5.0 is exercised.
    ckpt["feature_mean"] = np.full(len(FEATURE_NAMES), 1000.0, dtype=np.float32)

    captured_batches: list[np.ndarray] = []

    def dummy_predict(scaled_x: np.ndarray) -> np.ndarray:
        captured_batches.append(scaled_x)
        return np.array([1.2, -0.4], dtype=np.float64)

    pred = predict_match_snapshot(raw, dummy_predict, ckpt, drop_incomplete_minute=True)
    assert len(captured_batches) == 1
    assert captured_batches[0].shape == (2, 10, len(FEATURE_NAMES))
    assert np.all(captured_batches[0] >= -5.0) and np.all(captured_batches[0] <= 5.0)

    expected_home = float(expit(1.5 * 1.2 - 1.8))
    expected_away = float(expit(1.5 * (-0.4) - 1.8))
    expected_any = 1.0 - (1.0 - expected_home) * (1.0 - expected_away)

    assert pred.home.goal_probability == pytest.approx(expected_home, rel=1e-6)
    assert pred.away.goal_probability == pytest.approx(expected_away, rel=1e-6)
    assert pred.any_goal_probability == pytest.approx(expected_any, rel=1e-6)
    assert calibrate_logit(0.0, 1.5, -1.8) == pytest.approx(float(expit(-1.8)))


def test_predict_match_history_no_halftime_crossing() -> None:
    """Verifies history mode slices every valid window per half without crossing halftime."""
    raw = _make_raw_match_payload(
        home_minutes=[(1, 0), (1, 15), (2, 46), (2, 57)],
        away_minutes=[(1, 0), (1, 15), (2, 46), (2, 57)],
    )
    ckpt = _dummy_ckpt(seq_len=10, horizon=10)

    def dummy_predict(scaled_x: np.ndarray) -> np.ndarray:
        return np.zeros(len(scaled_x), dtype=np.float64)

    hist = predict_match_history(raw, dummy_predict, ckpt)
    # Period 1 (0..15 -> 7 windows: 9..15), Period 2 (46..57 -> 3 windows: 55..57) => 10 rows.
    assert len(hist) == 10
    assert hist[hist["period_value"] == 1]["minute"].tolist() == list(range(9, 16))
    assert hist[hist["period_value"] == 2]["minute"].tolist() == [55, 56, 57]
    assert np.allclose(hist["p_any_goal"], 1.0 - (1.0 - hist["p_home"]) * (1.0 - hist["p_away"]))


def test_resolve_default_model_path_finds_existing_checkpoint_and_raises_clear_torch_err() -> None:
    """Verifies default model path resolves to models/LSTM/lstm_goal_predictor.pt on disk."""
    resolved = resolve_default_model_path()
    assert resolved.exists()
    assert resolved.name == "lstm_goal_predictor.pt"

    with pytest.raises(FileNotFoundError, match="Trained LSTM checkpoint not found"):
        load_lstm_checkpoint("models/lstm/does_not_exist.pt")


@pytest.mark.skipif(not RAW_FILES, reason=f"No raw JSON files in {RAW_DIR}")
def test_predict_live_on_real_raw_json_snapshot() -> None:
    """Runs live snapshot and history extraction on a real cached WhoScored JSON."""
    raw_data = json.loads(RAW_FILES[0].read_text(encoding="utf-8"))
    ckpt = _dummy_ckpt(seq_len=10, horizon=10)

    def dummy_predict(scaled_x: np.ndarray) -> np.ndarray:
        return np.zeros(len(scaled_x), dtype=np.float64)

    pred = predict_match_snapshot(raw_data, dummy_predict, ckpt, drop_incomplete_minute=False)
    assert pred.home.window_end_minute - pred.home.window_start_minute == 9
    assert 0.0 < pred.any_goal_probability < 1.0

    hist = predict_match_history(raw_data, dummy_predict, ckpt)
    assert not hist.empty
    assert set(hist["period_value"].unique()).issubset({1, 2})
