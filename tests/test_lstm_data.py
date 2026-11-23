"""
test_lstm_data.py
=================
Unit tests for ``src.lstm_data`` verifying causal feature construction, own-goal
attribution, halftime isolation, direct summation of ``team_stats`` increments,
and leak-free match-level splitting/scaling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lstm_data import (
    FEATURE_NAMES,
    build_minute_panel_from_frames,
    build_sequences,
    load_dataset_bundle,
    save_dataset_bundle,
    split_and_scale,
)

EVENT_COLUMNS: tuple[str, ...] = (
    "match_id",
    "period_value",
    "team_id",
    "expanded_minute",
    "event_type",
    "outcome_type",
    "is_touch",
    "is_shot",
    "is_goal",
    "is_own_goal",
    "is_big_chance",
    "is_key_pass",
    "is_penalty",
    "is_red_card",
    "is_second_yellow_card",
    "body_part",
    "x",
    "y",
    "end_x",
    "end_y",
)


def _make_metadata(
    match_id: int = 1,
    home_id: int = 100,
    away_id: int = 200,
) -> pd.DataFrame:
    """Creates a minimal single-match metadata DataFrame."""
    return pd.DataFrame([
        {
            "match_id": match_id,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_name": f"Home_{match_id}",
            "away_team_name": f"Away_{match_id}",
        },
    ])


def _make_event(**overrides: object) -> dict[str, object]:
    """Creates a full event dictionary with safe defaults for every queried column."""
    row: dict[str, object] = {
        "match_id": 1,
        "period_value": 1,
        "team_id": 100,
        "expanded_minute": 0,
        "event_type": "Pass",
        "outcome_type": "Successful",
        "is_touch": True,
        "is_shot": False,
        "is_goal": False,
        "is_own_goal": False,
        "is_big_chance": False,
        "is_key_pass": False,
        "is_penalty": False,
        "is_red_card": False,
        "is_second_yellow_card": False,
        "body_part": "RightFoot",
        "x": 50.0,
        "y": 50.0,
        "end_x": 55.0,
        "end_y": 50.0,
    }
    row.update(overrides)
    return row


def _empty_team_stats() -> pd.DataFrame:
    """Returns an empty ``team_stats`` DataFrame with expected column types."""
    return pd.DataFrame({
        "match_id": pd.Series(dtype="int64"),
        "team_id": pd.Series(dtype="int64"),
        "stat_name": pd.Series(dtype="str"),
        "minute": pd.Series(dtype="int64"),
        "value": pd.Series(dtype="float64"),
    })


def test_own_goal_credited_to_opponent() -> None:
    """
    Verifies that an own goal by Home (team 100) at minute 20 counts as a future
    goal for Away (team 200) in minutes 10..19 and updates ``score_diff`` from minute 20+.
    """
    events_df = pd.DataFrame([
        _make_event(expanded_minute=0, team_id=100),
        _make_event(
            expanded_minute=20,
            team_id=100,
            event_type="Goal",
            is_shot=False,
            is_goal=True,
            is_own_goal=True,
        ),
        _make_event(expanded_minute=30, team_id=200),
    ])
    panel = build_minute_panel_from_frames(
        events_df, _empty_team_stats(), _make_metadata(), horizon_minutes=10,
    )

    home = panel[panel["team_id"] == 100].set_index("t")
    away = panel[panel["team_id"] == 200].set_index("t")

    # Minutes 10..19 have minute 20 strictly inside (t, t + 10].
    assert (away.loc[10:19, "goal_next_m"] == 1).all()
    assert (home.loc[10:19, "goal_next_m"] == 0).all()

    # Before minute 20 score_diff is 0; from minute 20 onward Away leads +1 and Home trails -1.
    assert np.allclose(away.loc[0:19, "score_diff"], 0.0)
    assert np.allclose(away.loc[20:30, "score_diff"], 1.0)
    assert np.allclose(home.loc[20:30, "score_diff"], -1.0)


def test_label_strictly_future_and_causal() -> None:
    """
    Verifies that a goal at minute t does NOT mark minute t itself as ``goal_next_m == 1``,
    and that future events at t+1 do not leak backward into features at minute t.
    """
    events_base = [
        _make_event(expanded_minute=0, team_id=100, x=40.0, end_x=45.0),
        _make_event(
            expanded_minute=15,
            team_id=100,
            event_type="Goal",
            is_shot=True,
            is_goal=True,
            x=90.0,
            y=50.0,
        ),
        _make_event(expanded_minute=25, team_id=200),
    ]
    events_with_extra_future = [
        *events_base,
        _make_event(
            expanded_minute=14,
            team_id=100,
            x=92.0,
            y=50.0,
            is_shot=True,
            is_big_chance=True,
            is_key_pass=True,
        ),
    ]

    panel_a = build_minute_panel_from_frames(
        pd.DataFrame(events_base), _empty_team_stats(), _make_metadata(), horizon_minutes=5,
    )
    panel_b = build_minute_panel_from_frames(
        pd.DataFrame(events_with_extra_future),
        _empty_team_stats(),
        _make_metadata(),
        horizon_minutes=5,
    )

    home_a = panel_a[panel_a["team_id"] == 100].set_index("t")
    home_b = panel_b[panel_b["team_id"] == 100].set_index("t")

    # Goal at minute 15 triggers minutes 10..14, NOT minute 9 and NOT minute 15.
    assert home_a.loc[9, "goal_next_m"] == 0
    assert (home_a.loc[10:14, "goal_next_m"] == 1).all()
    assert home_a.loc[15, "goal_next_m"] == 0

    # Extra event at minute 14 must have zero impact on any feature up through minute 13.
    feat_cols = list(FEATURE_NAMES)
    pd.testing.assert_frame_equal(home_a.loc[0:13, feat_cols], home_b.loc[0:13, feat_cols])


def test_halftime_boundary_isolation() -> None:
    """
    Verifies that a goal early in Period 2 does not mark late Period 1 rows as positive,
    and that ``build_sequences`` never creates a window spanning across halftime.
    """
    events_df = pd.DataFrame([
        _make_event(period_value=1, expanded_minute=0, team_id=100),
        _make_event(period_value=1, expanded_minute=45, team_id=200),
        _make_event(period_value=2, expanded_minute=46, team_id=100),
        _make_event(
            period_value=2,
            expanded_minute=48,
            team_id=100,
            event_type="Goal",
            is_shot=True,
            is_goal=True,
            x=90.0,
            y=50.0,
        ),
        _make_event(period_value=2, expanded_minute=65, team_id=200),
    ])
    panel = build_minute_panel_from_frames(
        events_df, _empty_team_stats(), _make_metadata(), horizon_minutes=10,
    )

    home_p1 = panel[(panel["team_id"] == 100) & (panel["period_value"] == 1)].set_index("t")
    home_p2 = panel[(panel["team_id"] == 100) & (panel["period_value"] == 2)].set_index("t")

    assert (home_p1["goal_next_m"] == 0).all()
    assert not bool(home_p1.loc[36:45, "has_full_horizon"].any())
    assert home_p2.loc[46, "goal_next_m"] == 1
    assert home_p2.loc[47, "goal_next_m"] == 1

    x, y, meta = build_sequences(panel, sequence_length=10)
    assert x.shape[0] == len(y) == len(meta)
    # Period 1 valid windows end at minute 35 (35 + 10 <= 45); minutes 0..8 lack 10m history.
    p1_minutes = meta[meta["period_value"] == 1]["minute"].unique()
    assert int(p1_minutes.min()) == 9
    assert int(p1_minutes.max()) == 35
    # Period 2 starts at minute 46, so first 10-minute sequence ends at 46 + 9 = 55.
    p2_minutes = meta[meta["period_value"] == 2]["minute"].unique()
    assert int(p2_minutes.min()) == 55


def test_team_stats_summed_not_differenced() -> None:
    """
    Verifies that decreasing per-minute ``possession`` and ``clearances`` rows in ``team_stats``
    are summed directly rather than differenced with ``LAG``.
    """
    events_df = pd.DataFrame([
        _make_event(expanded_minute=0, team_id=100),
        _make_event(expanded_minute=5, team_id=200),
    ])
    team_stats_df = pd.DataFrame([
        # Home possession decreases (10 -> 5 -> 2): sum must equal 17.
        {"match_id": 1, "team_id": 100, "stat_name": "possession", "minute": 0, "value": 10.0},
        {"match_id": 1, "team_id": 100, "stat_name": "possession", "minute": 1, "value": 5.0},
        {"match_id": 1, "team_id": 100, "stat_name": "possession", "minute": 2, "value": 2.0},
        # Away possession at minute 0 is 3: total = 17 + 3 = 20 -> Home share = 17/20 = 0.85.
        {"match_id": 1, "team_id": 200, "stat_name": "possession", "minute": 0, "value": 3.0},
        # Clearances: Away 4 (2 + 2), Home 1 -> diff at t=3 is (4 - 1) / 4 = 0.75.
        {"match_id": 1, "team_id": 200, "stat_name": "clearances", "minute": 1, "value": 2.0},
        {"match_id": 1, "team_id": 200, "stat_name": "clearances", "minute": 3, "value": 2.0},
        {"match_id": 1, "team_id": 100, "stat_name": "clearances", "minute": 2, "value": 1.0},
    ])
    panel = build_minute_panel_from_frames(
        events_df, team_stats_df, _make_metadata(), horizon_minutes=2,
    )
    home = panel[panel["team_id"] == 100].set_index("t")

    assert float(home.loc[2, "poss_share_cum"]) == pytest.approx(17.0 / 20.0, rel=1e-6)
    assert float(home.loc[3, "clearances_diff_rate"]) == pytest.approx((4.0 - 1.0) / 4.0, rel=1e-6)


def test_split_by_match_and_train_only_scaling() -> None:
    """
    Verifies zero match overlap across splits, train-only scaler fitting, finite float32 tensors,
    and expected shapes ``(N, L, 20)``.
    """
    events_rows = []
    meta_rows = []
    for mid in range(1, 7):
        home_id, away_id = mid * 10 + 1, mid * 10 + 2
        meta_rows.append(_make_metadata(mid, home_id, away_id).iloc[0].to_dict())
        # Give match 6 extreme coordinates so val/test outliers cannot shift train scaler stats.
        x_val = 95.0 if mid == 6 else 55.0
        events_rows.extend([
            _make_event(match_id=mid, period_value=1, team_id=home_id, expanded_minute=0, x=x_val),
            _make_event(match_id=mid, period_value=1, team_id=away_id, expanded_minute=25, x=50.0),
            _make_event(
                match_id=mid,
                period_value=1,
                team_id=home_id,
                expanded_minute=18,
                event_type="Goal",
                is_shot=True,
                is_goal=True,
                x=90.0,
            ),
        ])

    panel = build_minute_panel_from_frames(
        pd.DataFrame(events_rows),
        _empty_team_stats(),
        pd.DataFrame(meta_rows),
        horizon_minutes=5,
    )
    x, y, meta = build_sequences(panel, sequence_length=6)
    bundle = split_and_scale(x, y, meta, val_ratio=0.17, test_ratio=0.17, seed=42)

    train_m = set(bundle.train.meta["match_id"].unique())
    val_m = set(bundle.val.meta["match_id"].unique())
    test_m = set(bundle.test.meta["match_id"].unique())
    assert train_m and val_m and test_m
    assert train_m.isdisjoint(val_m)
    assert train_m.isdisjoint(test_m)
    assert val_m.isdisjoint(test_m)

    for split in (bundle.train, bundle.val, bundle.test):
        assert split.x.ndim == 3
        assert split.x.shape[1:] == (6, len(FEATURE_NAMES))
        assert split.x.dtype == np.float32
        assert split.y.dtype == np.float32
        assert np.all(np.isfinite(split.x))
        assert set(np.unique(split.y)).issubset({0.0, 1.0})

    # Verify train mean/std matches raw train slice directly (no leakage from val/test).
    train_mask = meta["match_id"].isin(train_m).to_numpy()
    expected_mean = np.mean(x[train_mask], axis=(0, 1))
    np.testing.assert_allclose(bundle.feature_mean, expected_mean, rtol=1e-5, atol=1e-5)


def test_save_and_load_dataset_bundle_roundtrip(tmp_path: Path) -> None:
    """
    Verifies that ``save_dataset_bundle`` and ``load_dataset_bundle`` preserve arrays,
    scaler statistics, and metadata DataFrames without Python object pickling.
    """
    events_rows = []
    meta_rows = []
    for mid in range(1, 7):
        home_id, away_id = mid * 10 + 1, mid * 10 + 2
        meta_rows.append(_make_metadata(mid, home_id, away_id).iloc[0].to_dict())
        events_rows.extend([
            _make_event(match_id=mid, period_value=1, team_id=home_id, expanded_minute=0),
            _make_event(match_id=mid, period_value=1, team_id=away_id, expanded_minute=20),
            _make_event(
                match_id=mid,
                period_value=1,
                team_id=home_id,
                expanded_minute=14,
                event_type="Goal",
                is_shot=True,
                is_goal=True,
                x=90.0,
            ),
        ])

    panel = build_minute_panel_from_frames(
        pd.DataFrame(events_rows),
        _empty_team_stats(),
        pd.DataFrame(meta_rows),
        horizon_minutes=4,
    )
    x, y, meta = build_sequences(panel, sequence_length=5)
    bundle = split_and_scale(x, y, meta, val_ratio=0.17, test_ratio=0.17, seed=42)
    bundle.horizon_minutes = 4

    archive_path = save_dataset_bundle(bundle, tmp_path / "bundle.npz")
    with np.load(archive_path, allow_pickle=False) as raw_archive:
        assert "train_x" in raw_archive

    loaded = load_dataset_bundle(archive_path)
    assert loaded.feature_names == bundle.feature_names
    assert loaded.sequence_length == bundle.sequence_length
    assert loaded.horizon_minutes == bundle.horizon_minutes
    np.testing.assert_allclose(loaded.feature_mean, bundle.feature_mean)
    np.testing.assert_allclose(loaded.feature_std, bundle.feature_std)

    for expected, actual in (
        (bundle.train, loaded.train),
        (bundle.val, loaded.val),
        (bundle.test, loaded.test),
    ):
        np.testing.assert_array_equal(actual.x, expected.x)
        np.testing.assert_array_equal(actual.y, expected.y)
        pd.testing.assert_frame_equal(actual.meta, expected.meta)
