"""
lstm_data.py
============
Simple, leak-free data preparation for predicting whether a team will score
within the next ``M`` minutes using an LSTM.

Only three tables produced by the scraper pipeline are needed:
    1. ``events``     — pitch coordinates, touches, passes, shots, cards, goals.
    2. ``team_stats`` — per-minute ``possession`` counts and ``clearances``.
    3. ``metadata``   — home/away team IDs and names for each match.

Why this compact 20-feature set?
    Across 240 matches in ``output/combined/``, combining spatial pressure from
    ``events`` with ``team_stats`` possession share and clearances captured
    nearly all of the predictive signal (5-fold match-grouped ROC-AUC ~0.595),
    while dumping ~100 redundant features from all tables hurt generalization.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Compact, causal 20-feature catalogue (verified per-minute at time t)
# ---------------------------------------------------------------------------

FEATURE_NAMES: tuple[str, ...] = (
    # 1. Minute-t dynamic activity (10 features)
    "box_touches",
    "opp_box_touches",
    "box_entries",
    "final_third_touches",
    "passes_final_third_acc",
    "shots_in_box",
    "xg_proxy",
    "key_passes",
    "high_recoveries",
    "opp_own_third_turnovers",
    # 2. Match-to-date running dominance up to minute t (6 features)
    "box_touches_rate",
    "box_entries_diff_rate",
    "poss_share_cum",
    "field_tilt_cum",
    "avg_x_diff_cum",
    "clearances_diff_rate",
    # 3. Live game state & clock at minute t (4 features)
    "score_diff",
    "man_advantage",
    "minute_norm",
    "is_home",
)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "box_touches": "Own touches inside opponent penalty box in minute t (x>=83, 21<=y<=79).",
    "opp_box_touches": "Opponent touches inside our penalty box in minute t.",
    "box_entries": (
        "Own events ending in opponent penalty box in minute t (end_x>=83, 21<=end_y<=79)."
    ),
    "final_third_touches": "Own touches in attacking third in minute t (x>=66).",
    "passes_final_third_acc": "Own successful passes ending in attacking third in minute t.",
    "shots_in_box": "Own shots taken from inside the penalty box in minute t.",
    "xg_proxy": "Geometric shot-quality sum in minute t (distance, big chance, header, penalty).",
    "key_passes": "Own key passes creating a shot in minute t.",
    "high_recoveries": "Own ball recoveries in attacking half (x>=60) in minute t.",
    "opp_own_third_turnovers": (
        "Opponent failed passes or dispossessions in their own third (x<=35)."
    ),
    "box_touches_rate": "Cumulative own penalty-box touches per minute played up to t.",
    "box_entries_diff_rate": "Cumulative (own - opponent) box entries per minute played up to t.",
    "poss_share_cum": "Cumulative possession share from team_stats: own / (own + opp).",
    "field_tilt_cum": "Cumulative share of accurate final-third passes: own / (own + opp).",
    "avg_x_diff_cum": "Running match-to-date difference in average event x (own - opponent).",
    "clearances_diff_rate": (
        "Cumulative (opp - own) clearances per minute (high = opponent under siege)."
    ),
    "score_diff": (
        "Live goal difference (goals_for - goals_against) at minute t (own goals flipped)."
    ),
    "man_advantage": "Red-card advantage at minute t (opponent red cards - own red cards).",
    "minute_norm": "Expanded match minute divided by 90.0.",
    "is_home": "1.0 if this row is for the home team, 0.0 for the away team.",
}

META_COLUMNS: tuple[str, ...] = (
    "match_id",
    "team_id",
    "team_name",
    "opp_team_name",
    "is_home",
    "period_value",
    "minute",
    "score_diff",
)


@dataclass
class SequenceSplit:
    """One train, validation, or test split ready for PyTorch DataLoader."""

    x: np.ndarray  # shape: (num_samples, sequence_length, num_features), float32
    y: np.ndarray  # shape: (num_samples,), float32 in {0.0, 1.0}
    meta: pd.DataFrame  # alignment metadata (match_id, team_id, period_value, minute, ...)


@dataclass
class LSTMDatasetBundle:
    """Complete bundle returned by ``prepare_lstm_dataset``."""

    train: SequenceSplit
    val: SequenceSplit
    test: SequenceSplit
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    sequence_length: int
    horizon_minutes: int


# ---------------------------------------------------------------------------
# Static SQL templates (parameterized with `?` — never f-string interpolated)
# ---------------------------------------------------------------------------

_CREATE_PER_MATCH_EVENTS_SQL = """
CREATE OR REPLACE TABLE events AS
SELECT
    match_id, period_value, team_id, expanded_minute, event_type, outcome_type,
    is_touch, is_shot, is_goal, is_own_goal, is_big_chance, is_key_pass,
    is_penalty, is_red_card, is_second_yellow_card, body_part,
    x, y, end_x, end_y
FROM read_csv_auto(?, union_by_name=True, ignore_errors=False)
"""

_CREATE_PER_MATCH_TEAM_STATS_SQL = """
CREATE OR REPLACE TABLE team_stats AS
SELECT match_id, team_id, stat_name, minute, value
FROM read_csv_auto(?, union_by_name=True, ignore_errors=False)
"""

_CREATE_PER_MATCH_METADATA_SQL = """
CREATE OR REPLACE TABLE metadata AS
SELECT match_id, home_team_id, away_team_id, home_team_name, away_team_name
FROM read_csv_auto(?, union_by_name=True, ignore_errors=False)
"""

_CREATE_COMBINED_TABLE_TEMPLATE = {
    "events": """
        CREATE OR REPLACE TABLE events AS
        SELECT
            match_id, period_value, team_id, expanded_minute, event_type, outcome_type,
            is_touch, is_shot, is_goal, is_own_goal, is_big_chance, is_key_pass,
            is_penalty, is_red_card, is_second_yellow_card, body_part,
            x, y, end_x, end_y
        FROM read_csv_auto(?, header=True)
    """,
    "team_stats": """
        CREATE OR REPLACE TABLE team_stats AS
        SELECT match_id, team_id, stat_name, minute, value
        FROM read_csv_auto(?, header=True)
    """,
    "metadata": """
        CREATE OR REPLACE TABLE metadata AS
        SELECT match_id, home_team_id, away_team_id, home_team_name, away_team_name
        FROM read_csv_auto(?, header=True)
    """,
}

# Important data nuances handled inside _PANEL_SQL:
# 1. Only regular halves (period_value IN (1, 2)) are included; extra time and
#    penalty shootouts are excluded.
# 2. `team_stats` values (`possession`, `clearances`) are ALREADY per-minute
#    increments, so we SUM them directly (never take LAG differences).
# 3. `team_stats.minute` matches `events.expanded_minute`.
# 4. Own goals (`is_own_goal = TRUE`) are credited to `opp_team_id` so the
#    scoring beneficiary gets both the future goal target and `score_diff` +1.
_PANEL_SQL = """
WITH half_bounds AS (
    SELECT
        CAST(match_id AS BIGINT) AS match_id,
        CAST(period_value AS INT) AS period_value,
        MIN(CAST(expanded_minute AS INT)) AS min_t,
        MAX(CAST(expanded_minute AS INT)) AS max_t
    FROM events
    WHERE CAST(period_value AS INT) IN (1, 2) AND expanded_minute IS NOT NULL
    GROUP BY 1, 2
),
teams AS (
    SELECT
        CAST(match_id AS BIGINT) AS match_id,
        CAST(home_team_id AS BIGINT) AS team_id,
        CAST(away_team_id AS BIGINT) AS opp_team_id,
        CAST(home_team_name AS VARCHAR) AS team_name,
        CAST(away_team_name AS VARCHAR) AS opp_team_name,
        1.0 AS is_home
    FROM metadata
    UNION ALL
    SELECT
        CAST(match_id AS BIGINT) AS match_id,
        CAST(away_team_id AS BIGINT) AS team_id,
        CAST(home_team_id AS BIGINT) AS opp_team_id,
        CAST(away_team_name AS VARCHAR) AS team_name,
        CAST(home_team_name AS VARCHAR) AS opp_team_name,
        0.0 AS is_home
    FROM metadata
),
grid AS (
    SELECT
        hb.match_id,
        hb.period_value,
        tm.team_id,
        tm.opp_team_id,
        tm.team_name,
        tm.opp_team_name,
        tm.is_home,
        gs.t
    FROM half_bounds hb
    JOIN teams tm
        ON hb.match_id = tm.match_id
    CROSS JOIN LATERAL generate_series(hb.min_t, hb.max_t) AS gs(t)
),
ev_goals AS (
    SELECT
        CAST(e.match_id AS BIGINT) AS match_id,
        CAST(e.period_value AS INT) AS period_value,
        CASE
            WHEN COALESCE(CAST(e.is_own_goal AS BOOLEAN), FALSE) THEN tm.opp_team_id
            ELSE CAST(e.team_id AS BIGINT)
        END AS scoring_team_id,
        CAST(e.expanded_minute AS INT) AS t,
        COUNT(*) AS goals_scored
    FROM events e
    JOIN teams tm
        ON CAST(e.match_id AS BIGINT) = tm.match_id
       AND CAST(e.team_id AS BIGINT) = tm.team_id
    WHERE COALESCE(CAST(e.is_goal AS BOOLEAN), FALSE)
      AND CAST(e.period_value AS INT) IN (1, 2)
      AND e.expanded_minute IS NOT NULL
    GROUP BY 1, 2, 3, 4
),
ev_min AS (
    SELECT
        CAST(match_id AS BIGINT) AS match_id,
        CAST(period_value AS INT) AS period_value,
        CAST(team_id AS BIGINT) AS team_id,
        CAST(expanded_minute AS INT) AS t,
        AVG(CAST(x AS DOUBLE)) AS ev_avg_x,
        SUM(CASE
            WHEN COALESCE(CAST(is_touch AS BOOLEAN), FALSE)
             AND x >= 83.0 AND y BETWEEN 21.0 AND 79.0 THEN 1 ELSE 0
        END) AS box_touches,
        SUM(CASE
            WHEN end_x >= 83.0 AND end_y BETWEEN 21.0 AND 79.0 THEN 1 ELSE 0
        END) AS box_entries,
        SUM(CASE
            WHEN COALESCE(CAST(is_touch AS BOOLEAN), FALSE) AND x >= 66.0 THEN 1 ELSE 0
        END) AS final_third_touches,
        SUM(CASE
            WHEN event_type = 'Pass' AND outcome_type = 'Successful' AND end_x >= 66.0
            THEN 1 ELSE 0
        END) AS passes_final_third_acc,
        SUM(CASE
            WHEN COALESCE(CAST(is_shot AS BOOLEAN), FALSE)
             AND x >= 83.0 AND y BETWEEN 21.0 AND 79.0 THEN 1 ELSE 0
        END) AS shots_in_box,
        SUM(CASE
            WHEN COALESCE(CAST(is_shot AS BOOLEAN), FALSE) THEN
                CASE
                    WHEN COALESCE(CAST(is_penalty AS BOOLEAN), FALSE) THEN 0.76
                    ELSE 1.0 / (
                        1.0 + EXP(-(
                            -0.6
                            - 0.12 * SQRT(
                                POWER((100.0 - COALESCE(CAST(x AS DOUBLE), 85.0)) * 1.05, 2)
                                + POWER((50.0 - COALESCE(CAST(y AS DOUBLE), 50.0)) * 0.68, 2)
                            )
                            + 1.4 * CAST(COALESCE(CAST(is_big_chance AS BOOLEAN), FALSE) AS INT)
                            - 0.6 * CASE WHEN body_part = 'Head' THEN 1 ELSE 0 END
                        ))
                    )
                END
            ELSE 0.0
        END) AS xg_proxy,
        SUM(CAST(COALESCE(CAST(is_key_pass AS BOOLEAN), FALSE) AS INT)) AS key_passes,
        SUM(CASE
            WHEN event_type = 'BallRecovery' AND x >= 60.0 THEN 1 ELSE 0
        END) AS high_recoveries,
        SUM(CASE
            WHEN (
                (event_type = 'Pass' AND outcome_type = 'Unsuccessful')
                OR event_type = 'Dispossessed'
            ) AND x <= 35.0 THEN 1 ELSE 0
        END) AS own_third_turnovers,
        SUM(CASE
            WHEN COALESCE(CAST(is_red_card AS BOOLEAN), FALSE)
              OR COALESCE(CAST(is_second_yellow_card AS BOOLEAN), FALSE) THEN 1 ELSE 0
        END) AS red_cards
    FROM events
    WHERE CAST(period_value AS INT) IN (1, 2) AND expanded_minute IS NOT NULL
    GROUP BY 1, 2, 3, 4
),
ts_min AS (
    SELECT
        CAST(match_id AS BIGINT) AS match_id,
        CAST(team_id AS BIGINT) AS team_id,
        CAST(minute AS INT) AS t,
        COALESCE(MAX(CASE WHEN stat_name = 'possession' THEN CAST(value AS DOUBLE) END), 0.0)
            AS poss_cnt,
        COALESCE(MAX(CASE WHEN stat_name = 'clearances' THEN CAST(value AS DOUBLE) END), 0.0)
            AS clearances
    FROM team_stats
    WHERE stat_name IN ('possession', 'clearances') AND minute IS NOT NULL
    GROUP BY 1, 2, 3
),
team_cum AS (
    SELECT
        g.match_id,
        g.period_value,
        g.team_id,
        g.opp_team_id,
        g.team_name,
        g.opp_team_name,
        g.is_home,
        g.t,
        CAST(COALESCE(eg.goals_scored, 0) AS INT) AS goals_scored,
        CAST(COALESCE(ev.box_touches, 0) AS DOUBLE) AS box_touches,
        CAST(COALESCE(ev.box_entries, 0) AS DOUBLE) AS box_entries,
        CAST(COALESCE(ev.final_third_touches, 0) AS DOUBLE) AS final_third_touches,
        CAST(COALESCE(ev.passes_final_third_acc, 0) AS DOUBLE) AS passes_final_third_acc,
        CAST(COALESCE(ev.shots_in_box, 0) AS DOUBLE) AS shots_in_box,
        CAST(COALESCE(ev.xg_proxy, 0.0) AS DOUBLE) AS xg_proxy,
        CAST(COALESCE(ev.key_passes, 0) AS DOUBLE) AS key_passes,
        CAST(COALESCE(ev.high_recoveries, 0) AS DOUBLE) AS high_recoveries,
        CAST(COALESCE(ev.own_third_turnovers, 0) AS DOUBLE) AS own_third_turnovers,
        SUM(COALESCE(ev.box_touches, 0)) OVER wm AS box_touches_cum,
        SUM(COALESCE(ev.box_entries, 0)) OVER wm AS box_entries_cum,
        SUM(COALESCE(ev.passes_final_third_acc, 0)) OVER wm AS passes_f3_acc_cum,
        SUM(COALESCE(ts.poss_cnt, 0.0)) OVER wm AS poss_cum,
        SUM(COALESCE(ts.clearances, 0.0)) OVER wm AS clearances_cum,
        COALESCE(AVG(ev.ev_avg_x) OVER wm, 50.0) AS avg_x_cum,
        SUM(COALESCE(eg.goals_scored, 0)) OVER wm AS goals_for_cum,
        SUM(COALESCE(ev.red_cards, 0)) OVER wm AS red_cards_cum
    FROM grid g
    LEFT JOIN ev_goals eg
        ON g.match_id = eg.match_id
       AND g.period_value = eg.period_value
       AND g.team_id = eg.scoring_team_id
       AND g.t = eg.t
    LEFT JOIN ev_min ev
        ON g.match_id = ev.match_id
       AND g.period_value = ev.period_value
       AND g.team_id = ev.team_id
       AND g.t = ev.t
    LEFT JOIN ts_min ts
        ON g.match_id = ts.match_id
       AND g.team_id = ts.team_id
       AND g.t = ts.t
    WINDOW wm AS (
        PARTITION BY g.match_id, g.team_id
        ORDER BY g.period_value, g.t
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
)
SELECT
    me.match_id,
    me.period_value,
    me.team_id,
    me.team_name,
    me.opp_team_name,
    me.t,
    me.goals_scored,
    -- Minute-t dynamic activity (10 features)
    me.box_touches,
    opp.box_touches AS opp_box_touches,
    me.box_entries,
    me.final_third_touches,
    me.passes_final_third_acc,
    me.shots_in_box,
    me.xg_proxy,
    me.key_passes,
    me.high_recoveries,
    opp.own_third_turnovers AS opp_own_third_turnovers,
    -- Match-to-date running dominance (6 features)
    me.box_touches_cum / GREATEST(me.t + 1.0, 1.0) AS box_touches_rate,
    (me.box_entries_cum - opp.box_entries_cum) / GREATEST(me.t + 1.0, 1.0)
        AS box_entries_diff_rate,
    CASE
        WHEN (me.poss_cum + opp.poss_cum) > 0.0
        THEN me.poss_cum / (me.poss_cum + opp.poss_cum)
        ELSE 0.5
    END AS poss_share_cum,
    CASE
        WHEN (me.passes_f3_acc_cum + opp.passes_f3_acc_cum) > 0.0
        THEN me.passes_f3_acc_cum / (me.passes_f3_acc_cum + opp.passes_f3_acc_cum)
        ELSE 0.5
    END AS field_tilt_cum,
    me.avg_x_cum - opp.avg_x_cum AS avg_x_diff_cum,
    (opp.clearances_cum - me.clearances_cum) / GREATEST(me.t + 1.0, 1.0)
        AS clearances_diff_rate,
    -- Live game state & clock (4 features)
    CAST(me.goals_for_cum - opp.goals_for_cum AS DOUBLE) AS score_diff,
    CAST(opp.red_cards_cum - me.red_cards_cum AS DOUBLE) AS man_advantage,
    CAST(me.t AS DOUBLE) / 90.0 AS minute_norm,
    me.is_home
FROM team_cum me
JOIN team_cum opp
    ON me.match_id = opp.match_id
   AND me.period_value = opp.period_value
   AND me.t = opp.t
   AND me.opp_team_id = opp.team_id
ORDER BY me.match_id, me.team_id, me.period_value, me.t
"""


# ---------------------------------------------------------------------------
# Data loading, panel construction, target labeling, and sequence slicing
# ---------------------------------------------------------------------------


def _has_all_per_match_tables(root: Path) -> bool:
    """Checks whether per-match events, team_stats, and metadata CSVs exist."""
    return (
        any(root.glob("match_*_events.csv"))
        and any(root.glob("match_*_team_stats.csv"))
        and any(root.glob("match_*_metadata.csv"))
    )


def load_match_tables(
    data_dir: Path | str = "output",
    use_all_matches: bool = True,
) -> duckdb.DuckDBPyConnection:
    """
    Loads ``events``, ``team_stats``, and ``metadata`` CSVs into an in-memory DuckDB connection.

    Args:
        data_dir: Root export directory containing ``match_*_*.csv`` and/or ``combined/``.
        use_all_matches: When ``True``, loads every per-match CSV matching
            ``match_*_{events,team_stats,metadata}.csv`` (e.g. all 351 matches) if present,
            falling back to ``combined/combined_*.csv``. When ``False``, loads ``combined/``.

    Returns:
        Open in-memory ``duckdb.DuckDBPyConnection`` with ``events``, ``team_stats``, and
        ``metadata`` tables registered.

    Raises:
        FileNotFoundError: If neither per-match nor combined CSV tables can be found.
    """
    root = Path(data_dir)
    con = duckdb.connect()

    if use_all_matches and _has_all_per_match_tables(root):
        con.execute(_CREATE_PER_MATCH_EVENTS_SQL, [str(root / "match_*_events.csv")])
        con.execute(_CREATE_PER_MATCH_TEAM_STATS_SQL, [str(root / "match_*_team_stats.csv")])
        con.execute(_CREATE_PER_MATCH_METADATA_SQL, [str(root / "match_*_metadata.csv")])
        return con

    combined_dir = root / "combined"
    for table_name, create_sql in _CREATE_COMBINED_TABLE_TEMPLATE.items():
        csv_path = combined_dir / f"combined_{table_name}.csv"
        if not csv_path.exists():
            con.close()
            raise FileNotFoundError(
                f"Missing '{csv_path}'. Run `python -m src.main` first or pass a valid `data_dir`.",
            )
        con.execute(create_sql, [str(csv_path)])

    return con


def add_future_goal_labels(panel: pd.DataFrame, horizon_minutes: int = 10) -> pd.DataFrame:
    """
    Adds binary target ``goal_next_m`` and boundary flag ``has_full_horizon`` to a minute panel.

    For each minute ``t`` inside a half, ``goal_next_m == 1`` if and only if the team
    (including own goals credited in their favour) scores strictly in ``(t, t + horizon_minutes]``
    within the **same** half. Goals at minute ``t`` itself are excluded (past information),
    and Period 1 never looks ahead into Period 2.

    Args:
        panel: Minute-level DataFrame produced by ``_PANEL_SQL`` containing ``match_id``,
            ``team_id``, ``period_value``, ``t``, and ``goals_scored``.
        horizon_minutes: Prediction window length ``M`` in minutes (must be >= 1).

    Returns:
        Sorted copy of ``panel`` with integer columns ``goal_next_m`` and
        ``has_full_horizon`` (1 when ``t + horizon_minutes <= half_max_t``, else 0).
    """
    if horizon_minutes < 1:
        raise ValueError(f"horizon_minutes must be >= 1, got {horizon_minutes}")

    sorted_panel = panel.sort_values(["match_id", "team_id", "period_value", "t"]).reset_index(
        drop=True,
    )
    goals = sorted_panel["goals_scored"].to_numpy(dtype=np.int32)
    minutes = sorted_panel["t"].to_numpy(dtype=np.int32)

    goal_next_m = np.zeros(len(sorted_panel), dtype=np.int8)
    has_full_horizon = np.zeros(len(sorted_panel), dtype=bool)

    grouped = sorted_panel.groupby(["match_id", "team_id", "period_value"], sort=False).groups
    for idx in grouped.values():
        half_pos = np.asarray(list(idx), dtype=np.int64)
        half_goals = goals[half_pos]
        half_t = minutes[half_pos]
        half_max_t = int(half_t[-1])

        for local_i in range(len(half_pos)):
            future_slice = half_goals[local_i + 1 : local_i + 1 + horizon_minutes]
            goal_next_m[half_pos[local_i]] = 1 if np.any(future_slice > 0) else 0
            within_half = int(half_t[local_i]) + horizon_minutes <= half_max_t
            has_full_horizon[half_pos[local_i]] = within_half

    sorted_panel["goal_next_m"] = goal_next_m
    sorted_panel["has_full_horizon"] = has_full_horizon
    return sorted_panel


def build_minute_panel(
    con: duckdb.DuckDBPyConnection,
    horizon_minutes: int = 10,
) -> pd.DataFrame:
    """
    Computes the minute-by-minute causal feature panel and future-goal labels from DuckDB tables.

    Args:
        con: DuckDB connection containing ``events``, ``team_stats``, and ``metadata`` tables.
        horizon_minutes: Prediction horizon ``M`` in minutes (must be >= 1).

    Returns:
        DataFrame with one row per ``(match_id, team_id, period_value, t)``, all 20 causal
        features, ``goal_next_m``, and ``has_full_horizon``.
    """
    if horizon_minutes < 1:
        raise ValueError(f"horizon_minutes must be >= 1, got {horizon_minutes}")

    panel = con.execute(_PANEL_SQL).fetchdf()
    return add_future_goal_labels(panel, horizon_minutes=horizon_minutes)


def build_minute_panel_from_frames(
    events_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    horizon_minutes: int = 10,
) -> pd.DataFrame:
    """
    Convenience wrapper that builds the causal minute panel from in-memory pandas DataFrames.

    Args:
        events_df: Events table with columns matching ``events`` CSV export.
        team_stats_df: Team stats table (``match_id``, ``team_id``, ``stat_name``, ``minute``,
            ``value``).
        metadata_df: Match metadata table (``match_id``, ``home_team_id``, ``away_team_id``,
            ``home_team_name``, ``away_team_name``).
        horizon_minutes: Prediction horizon ``M`` in minutes (must be >= 1).

    Returns:
        DataFrame returned by ``build_minute_panel``.
    """
    con = duckdb.connect()
    try:
        con.register("events", events_df)
        con.register("team_stats", team_stats_df)
        con.register("metadata", metadata_df)
        return build_minute_panel(con, horizon_minutes=horizon_minutes)
    finally:
        con.close()


def _extract_half_sequences(
    half_features: np.ndarray,
    half_targets: np.ndarray,
    half_minutes: np.ndarray,
    half_valid: np.ndarray,
    half_positions: np.ndarray,
    sequence_length: int,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Extracts contiguous ``sequence_length`` windows within a single half."""
    seq_x: list[np.ndarray] = []
    seq_y: list[float] = []
    meta_rows: list[int] = []

    span = sequence_length - 1
    for i in range(span, len(half_positions)):
        if not half_valid[i]:
            continue
        # Guard against missing minutes inside a half.
        if int(half_minutes[i]) - int(half_minutes[i - span]) != span:
            continue
        seq_x.append(half_features[i - span : i + 1])
        seq_y.append(float(half_targets[i]))
        meta_rows.append(int(half_positions[i]))

    return seq_x, seq_y, meta_rows


def build_sequences(
    panel: pd.DataFrame,
    sequence_length: int = 10,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Slices a minute panel into fixed-length LSTM sequences strictly inside each half.

    Every sample ``i`` represents the sequence of minutes ``[t - sequence_length + 1 .. t]``
    and predicts ``goal_next_m`` in ``(t, t + M]``. Sequences never cross halftime.

    Args:
        panel: Output DataFrame from ``build_minute_panel``.
        sequence_length: Number of consecutive minutes per LSTM input sequence (``L >= 1``).
        feature_names: Ordered feature columns to include along the last tensor axis.

    Returns:
        Tuple ``(x, y, meta)`` where:
            - ``x`` has shape ``(N, sequence_length, len(feature_names))`` and dtype ``float32``.
            - ``y`` has shape ``(N,)`` and dtype ``float32``.
            - ``meta`` is a DataFrame aligned row-by-row with ``x`` and ``y``.
    """
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    sorted_panel = panel.sort_values(["match_id", "team_id", "period_value", "t"]).reset_index(
        drop=True,
    )
    feature_matrix = sorted_panel[list(feature_names)].to_numpy(dtype=np.float32)
    targets = sorted_panel["goal_next_m"].to_numpy(dtype=np.float32)
    minutes = sorted_panel["t"].to_numpy(dtype=np.int32)
    valid = sorted_panel["has_full_horizon"].to_numpy(dtype=bool)

    all_x: list[np.ndarray] = []
    all_y: list[float] = []
    meta_indices: list[int] = []

    grouped = sorted_panel.groupby(["match_id", "team_id", "period_value"], sort=False).groups
    for idx in grouped.values():
        pos = np.asarray(list(idx), dtype=np.int64)
        hx, hy, hm = _extract_half_sequences(
            feature_matrix[pos],
            targets[pos],
            minutes[pos],
            valid[pos],
            pos,
            sequence_length,
        )
        all_x.extend(hx)
        all_y.extend(hy)
        meta_indices.extend(hm)

    if not all_x:
        empty_x = np.empty((0, sequence_length, len(feature_names)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        empty_meta = sorted_panel.iloc[0:0].rename(columns={"t": "minute"})[list(META_COLUMNS)]
        return empty_x, empty_y, empty_meta

    x_arr = np.stack(all_x, axis=0).astype(np.float32)
    y_arr = np.asarray(all_y, dtype=np.float32)
    meta_df = (
        sorted_panel.iloc[meta_indices]
        .rename(columns={"t": "minute"})[list(META_COLUMNS)]
        .reset_index(drop=True)
    )
    return x_arr, y_arr, meta_df


def _split_match_ids(
    unique_matches: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[int], set[int], set[int]]:
    """Randomly partitions unique ``match_id`` values into train, val, and test sets."""
    n_matches = len(unique_matches)
    if n_matches < 3:
        raise ValueError(f"Need at least 3 matches to split train/val/test, got {n_matches}")
    if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1.0")

    shuffled = unique_matches.copy()
    np.random.default_rng(seed).shuffle(shuffled)

    n_test = max(1, round(n_matches * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, round(n_matches * val_ratio)) if val_ratio > 0 else 0
    if n_test + n_val >= n_matches:
        n_test = 1 if test_ratio > 0 else 0
        n_val = 1 if val_ratio > 0 else 0

    test_ids = {int(mid) for mid in shuffled[:n_test]}
    val_ids = {int(mid) for mid in shuffled[n_test : n_test + n_val]}
    train_ids = {int(mid) for mid in shuffled[n_test + n_val :]}
    return train_ids, val_ids, test_ids


def _scale_features(
    x_arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip_z: float,
) -> np.ndarray:
    """Applies z-score standardization, replaces non-finite values with 0, and clips."""
    scaled = (x_arr - mean) / std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scaled, -clip_z, clip_z).astype(np.float32)


def split_and_scale(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    clip_z: float = 5.0,
) -> LSTMDatasetBundle:
    """
    Splits sequences by ``match_id`` and standardizes features using train statistics only.

    Splitting by ``match_id`` prevents nearby minutes from the same match from appearing in
    both training and evaluation sets. Fitting mean/std only on ``train`` prevents any data
    leakage from validation or test matches.

    Args:
        x: Sequence tensor of shape ``(N, L, F)``.
        y: Target vector of shape ``(N,)``.
        meta: Aligned metadata DataFrame containing ``match_id``.
        val_ratio: Fraction of matches assigned to validation.
        test_ratio: Fraction of matches assigned to test.
        seed: Random seed controlling the match-level shuffle.
        clip_z: Symmetric z-score clipping threshold after scaling.

    Returns:
        ``LSTMDatasetBundle`` holding scaled ``train``, ``val``, and ``test`` splits plus the
        training scaler statistics.
    """
    unique_matches = np.sort(meta["match_id"].unique())
    train_ids, val_ids, test_ids = _split_match_ids(unique_matches, val_ratio, test_ratio, seed)

    match_col = meta["match_id"].to_numpy(dtype=np.int64)
    train_mask = np.isin(match_col, list(train_ids))
    val_mask = np.isin(match_col, list(val_ids))
    test_mask = np.isin(match_col, list(test_ids))

    x_train_raw = x[train_mask]
    mean = np.nanmean(x_train_raw, axis=(0, 1))
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)

    std = np.nanstd(x_train_raw, axis=(0, 1))
    std = np.where((std > 1e-8) & np.isfinite(std), std, 1.0).astype(np.float32)

    train_split = SequenceSplit(
        x=_scale_features(x_train_raw, mean, std, clip_z),
        y=y[train_mask].astype(np.float32),
        meta=meta.loc[train_mask].reset_index(drop=True),
    )
    val_split = SequenceSplit(
        x=_scale_features(x[val_mask], mean, std, clip_z),
        y=y[val_mask].astype(np.float32),
        meta=meta.loc[val_mask].reset_index(drop=True),
    )
    test_split = SequenceSplit(
        x=_scale_features(x[test_mask], mean, std, clip_z),
        y=y[test_mask].astype(np.float32),
        meta=meta.loc[test_mask].reset_index(drop=True),
    )

    seq_len = int(x.shape[1]) if x.ndim == 3 else 0
    return LSTMDatasetBundle(
        train=train_split,
        val=val_split,
        test=test_split,
        feature_names=FEATURE_NAMES,
        feature_mean=mean,
        feature_std=std,
        sequence_length=seq_len,
        horizon_minutes=0,
    )


def prepare_lstm_dataset(
    data_dir: Path | str = "output",
    sequence_length: int = 10,
    horizon_minutes: int = 10,
    use_all_matches: bool = True,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> LSTMDatasetBundle:
    """
    End-to-end pipeline: loads CSVs, builds causal panel & sequences, splits by match, and scales.

    Args:
        data_dir: Directory containing scraper CSV outputs (default ``"output"``).
        sequence_length: Number of consecutive past minutes ``L`` fed into the LSTM.
        horizon_minutes: Future prediction window ``M`` in minutes (target: goal in ``(t, t+M]``).
        use_all_matches: Load all per-match CSVs in ``data_dir`` (recommended) or fall back to
            ``combined/``.
        val_ratio: Fraction of matches held out for validation.
        test_ratio: Fraction of matches held out for testing.
        seed: Random seed for reproducible match splitting.

    Returns:
        ``LSTMDatasetBundle`` ready to wrap in PyTorch ``TensorDataset`` / ``DataLoader``.
    """
    con = load_match_tables(data_dir=data_dir, use_all_matches=use_all_matches)
    try:
        panel = build_minute_panel(con, horizon_minutes=horizon_minutes)
    finally:
        con.close()

    x, y, meta = build_sequences(
        panel,
        sequence_length=sequence_length,
        feature_names=FEATURE_NAMES,
    )
    bundle = split_and_scale(
        x=x,
        y=y,
        meta=meta,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    bundle.horizon_minutes = horizon_minutes
    return bundle


_STRING_META_COLS = frozenset({"team_name", "opp_team_name"})


def _pack_split(prefix: str, split: SequenceSplit) -> dict[str, np.ndarray]:
    """Serializes one ``SequenceSplit`` into pickle-free NumPy arrays for ``.npz`` export."""
    payload: dict[str, np.ndarray] = {
        f"{prefix}_x": split.x.astype(np.float32),
        f"{prefix}_y": split.y.astype(np.float32),
    }
    for col in META_COLUMNS:
        key = f"{prefix}_meta_{col}"
        if col in _STRING_META_COLS:
            # Convert via list[str] so NumPy always stores fixed-width Unicode (<U...), not object.
            values = [str(val) for val in split.meta[col].fillna("")]
            payload[key] = np.asarray(values, dtype=str)
        else:
            payload[key] = split.meta[col].to_numpy()
    return payload


def _unpack_split(data: Mapping[str, np.ndarray], prefix: str) -> SequenceSplit:
    """Reconstructs a ``SequenceSplit`` from arrays loaded out of a ``.npz`` archive."""
    meta_dict: dict[str, object] = {}
    for col in META_COLUMNS:
        arr = data[f"{prefix}_meta_{col}"]
        if col in _STRING_META_COLS:
            meta_dict[col] = [str(val) for val in arr.tolist()]
        else:
            meta_dict[col] = arr
    return SequenceSplit(
        x=data[f"{prefix}_x"].astype(np.float32),
        y=data[f"{prefix}_y"].astype(np.float32),
        meta=pd.DataFrame(meta_dict, columns=list(META_COLUMNS)),
    )


def save_dataset_bundle(bundle: LSTMDatasetBundle, path: Path | str) -> Path:
    """
    Saves an ``LSTMDatasetBundle`` as a portable, compressed ``.npz`` file (zero pickle).

    All string metadata columns are stored as fixed-width Unicode arrays so the archive can
    be loaded safely with ``allow_pickle=False`` across Python versions (e.g. in Google Colab).

    Args:
        bundle: Prepared dataset bundle returned by ``prepare_lstm_dataset``.
        path: Destination ``.npz`` file path (parent directories are created automatically).

    Returns:
        Resolved ``Path`` to the written ``.npz`` archive.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "feature_names": np.asarray([str(name) for name in bundle.feature_names], dtype=str),
        "feature_mean": bundle.feature_mean.astype(np.float32),
        "feature_std": bundle.feature_std.astype(np.float32),
        "sequence_length": np.asarray(bundle.sequence_length, dtype=np.int64),
        "horizon_minutes": np.asarray(bundle.horizon_minutes, dtype=np.int64),
    }
    for prefix, split in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
        payload.update(_pack_split(prefix, split))

    np.savez_compressed(out_path, **payload)
    return out_path


def load_dataset_bundle(path: Path | str) -> LSTMDatasetBundle:
    """
    Loads an ``LSTMDatasetBundle`` saved by ``save_dataset_bundle`` using ``allow_pickle=False``.

    Args:
        path: Path to the ``.npz`` dataset archive.

    Returns:
        Reconstructed ``LSTMDatasetBundle`` ready for PyTorch training.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        return LSTMDatasetBundle(
            train=_unpack_split(data, "train"),
            val=_unpack_split(data, "val"),
            test=_unpack_split(data, "test"),
            feature_names=tuple(str(name) for name in data["feature_names"].tolist()),
            feature_mean=data["feature_mean"].astype(np.float32),
            feature_std=data["feature_std"].astype(np.float32),
            sequence_length=int(data["sequence_length"]),
            horizon_minutes=int(data["horizon_minutes"]),
        )


def _parse_cli_args() -> argparse.Namespace:
    """Parses CLI arguments for exporting a compact ``.npz`` bundle for Google Colab."""
    cli = argparse.ArgumentParser(
        description="Build and save a compressed LSTM dataset (.npz) ready for Google Colab.",
    )
    cli.add_argument(
        "--data-dir",
        default="output",
        help="Directory containing match_*_*.csv or combined/ CSVs (default: 'output').",
    )
    cli.add_argument(
        "--output",
        default="output/lstm_dataset.npz",
        help="Output .npz archive path (default: 'output/lstm_dataset.npz').",
    )
    cli.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Consecutive past minutes per LSTM sequence L (default: 10).",
    )
    cli.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Future prediction horizon M in minutes (default: 10).",
    )
    cli.add_argument(
        "--combined-only",
        action="store_true",
        help="Read only output/combined/ instead of all per-match CSVs.",
    )
    cli.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for match-level train/val/test split (default: 42).",
    )
    return cli.parse_args()


def main() -> None:
    """CLI entry point for building and exporting ``lstm_dataset.npz``."""
    args = _parse_cli_args()
    bundle = prepare_lstm_dataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        horizon_minutes=args.horizon,
        use_all_matches=not args.combined_only,
        seed=args.seed,
    )
    out_path = save_dataset_bundle(bundle, args.output)
    size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Saved LSTM dataset bundle to: {out_path.resolve()} ({size_mb:.2f} MB)")
    for name, split in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
        print(
            f"  {name:<5}: x={split.x.shape}, y={split.y.shape}, "
            f"matches={split.meta['match_id'].nunique()}, goal_rate={split.y.mean():.3%}",
        )


__all__ = [
    "FEATURE_DESCRIPTIONS",
    "FEATURE_NAMES",
    "META_COLUMNS",
    "LSTMDatasetBundle",
    "SequenceSplit",
    "add_future_goal_labels",
    "build_minute_panel",
    "build_minute_panel_from_frames",
    "build_sequences",
    "load_dataset_bundle",
    "load_match_tables",
    "prepare_lstm_dataset",
    "save_dataset_bundle",
    "split_and_scale",
]


if __name__ == "__main__":
    main()
