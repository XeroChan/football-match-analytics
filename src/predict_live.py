"""
predict_live.py
===============
Live and offline match goal-probability predictor using a trained PyTorch LSTM checkpoint.

How the pipeline works on an ongoing match
------------------------------------------
1. **Scrape from minute 0 to the current minute**: WhoScored's Match Centre payload
   (``matchCentreData``) always contains the match timeline from kickoff (``0'``) up to
   the current live minute ``t_now``. We need the full history ``0 .. t_now`` because 8 of
   the 20 model features track *cumulative* match context up to ``t`` (for example
   ``poss_share_cum``, ``box_touches_rate``, ``field_tilt_cum``, and live ``score_diff``).
2. **One 10-minute window per live prediction**: The LSTM itself does *not* require the
   entire match to be sliced into overlapping windows during live inference. To predict
   ``P(goal in next M minutes)`` right now, we extract the single latest ``L = 10``-minute
   window ``[t - 9 .. t]`` inside the current regular half for Home and Away (skipping the
   currently in-progress minute by default so partial counts do not under-predict).
3. **Full timeline mode (``--history``)**: Optionally slices every ``L``-minute window
   accumulated so far in the match (strictly within each half, never crossing halftime)
   so you can inspect how goal probabilities evolved minute by minute up to ``t_now``.

Checkpoint location
-------------------
By default, the script looks for a trained ``.pt`` checkpoint saved from ``06_lstm.ipynb``
or ``07_lstm_colab.ipynb`` in ``models/lstm/lstm_goal_predictor.pt`` (and automatically
matches ``models/LSTM/lstm_goal_predictor.pt`` as well).

CLI usage examples
------------------
1. Single live prediction from a WhoScored Match Centre URL::

    python -m src.predict_live --url "https://www.whoscored.com/matches/1821123/live"

2. Live watch mode (keeps one Chrome window open and refreshes every 60 seconds)::

    python -m src.predict_live --url "https://www.whoscored.com/matches/1821123/live" --watch

3. Print the full minute-by-minute probability timeline up to the current minute::

    python -m src.predict_live --url "https://www.whoscored.com/matches/1821123/live" --history

4. Offline test on a saved raw JSON backup (no browser launched)::

    python -m src.predict_live --from-json data/raw/match_1821123_*.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from src.lstm_data import build_minute_panel_from_frames
from src.parser import parse_match
from src.scraper import WhoScoredScraper, WhoScoredScraperError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CANDIDATES: tuple[Path, ...] = (
    Path("models/lstm/lstm_goal_predictor.pt"),
    Path("models/LSTM/lstm_goal_predictor.pt"),
)

_EMPTY_TEAM_STATS_DTYPES: dict[str, str] = {
    "match_id": "int64",
    "team_id": "int64",
    "stat_name": "str",
    "minute": "int64",
    "value": "float64",
}

_HISTORY_COLUMNS: list[str] = [
    "period_value",
    "minute",
    "home_team",
    "away_team",
    "score_diff",
    "p_home",
    "p_away",
    "p_any_goal",
]


@dataclass
class LiveTeamWindow:
    """Extracted raw (unscaled) feature window ``(L, F)`` for one team at minute ``t``."""

    team_name: str
    team_id: int
    is_home: bool
    period_value: int
    window_start_minute: int
    window_end_minute: int
    score_diff: int
    raw_features: np.ndarray


@dataclass
class TeamPrediction:
    """Calibrated goal-probability prediction for a single team over ``(t, t + M]``."""

    team_name: str
    team_id: int
    is_home: bool
    period_value: int
    window_start_minute: int
    window_end_minute: int
    score_diff: int
    raw_logit: float
    goal_probability: float


@dataclass
class LiveMatchPrediction:
    """Combined home, away, and any-goal prediction for a live or offline match snapshot."""

    match_id: int
    horizon_minutes: int
    sequence_length: int
    home: TeamPrediction
    away: TeamPrediction
    any_goal_probability: float


def resolve_default_model_path(explicit_path: Path | str | None = None) -> Path:
    """
    Resolves the path to the trained PyTorch ``.pt`` checkpoint.

    Args:
        explicit_path: User-supplied checkpoint path. When ``None``, searches
            ``DEFAULT_MODEL_CANDIDATES`` (``models/lstm/...`` and ``models/LSTM/...``).

    Returns:
        Resolved ``Path`` pointing to the checkpoint location.
    """
    if explicit_path is not None:
        return Path(explicit_path)
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_MODEL_CANDIDATES[0]


def extract_match_id_from_url(url_or_id: str) -> int:
    """
    Extracts the numeric WhoScored ``match_id`` from a Match Centre URL or integer string.

    Args:
        url_or_id: Full WhoScored URL (e.g.
            ``"https://www.whoscored.com/matches/1821123/live/..."``) or numeric ID string.

    Returns:
        Integer ``match_id``.

    Raises:
        ValueError: If no numeric match ID can be parsed from ``url_or_id``.
    """
    cleaned = str(url_or_id).strip()
    if cleaned.isdigit():
        return int(cleaned)

    match = re.search(r"/matches/(\d+)", cleaned, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    raise ValueError(
        f"Could not extract WhoScored match ID from '{url_or_id}'. "
        "Pass a URL like 'https://www.whoscored.com/matches/1821123/live' or a numeric ID.",
    )


def _build_live_panel(raw_data: dict[str, Any], horizon_minutes: int) -> pd.DataFrame:
    """Parses raw ``matchCentreData`` and builds the minute-by-minute 20-feature causal panel."""
    dataset = parse_match(raw_data)
    if dataset.events_df.empty or dataset.metadata_df.empty:
        raise ValueError("Match payload contains no events or metadata yet.")

    team_stats_df = dataset.team_stats_df
    if team_stats_df.empty:
        team_stats_df = pd.DataFrame({
            col: pd.Series(dtype=dtype) for col, dtype in _EMPTY_TEAM_STATS_DTYPES.items()
        })

    panel = build_minute_panel_from_frames(
        dataset.events_df,
        team_stats_df,
        dataset.metadata_df,
        horizon_minutes=horizon_minutes,
    )
    if panel.empty:
        raise ValueError("No regular-half minutes (period 1 or 2) are available yet.")
    return panel


def _select_team_window(
    half_panel: pd.DataFrame,
    side_label: str,
    is_home_team: bool,
    feature_names: Sequence[str],
    sequence_length: int,
    drop_incomplete_minute: bool,
) -> LiveTeamWindow:
    """Slices the single latest contiguous ``sequence_length``-minute window for one team."""
    side_mask = (
        half_panel["is_home"] >= 0.5 if is_home_team else half_panel["is_home"] < 0.5
    )
    team_df = half_panel[side_mask].sort_values("t")
    if drop_incomplete_minute and len(team_df) > 1:
        team_df = team_df.iloc[:-1]

    if len(team_df) < sequence_length:
        period = int(half_panel["period_value"].iloc[0])
        raise ValueError(
            f"Not enough completed minutes for {side_label} in half {period}: "
            f"found {len(team_df)}, need {sequence_length}.",
        )

    window = team_df.tail(sequence_length)
    start_t = int(window["t"].iloc[0])
    end_t = int(window["t"].iloc[-1])
    if end_t - start_t != sequence_length - 1:
        raise ValueError(
            f"Latest {sequence_length} rows for {side_label} are not contiguous minutes "
            f"({start_t}..{end_t}).",
        )

    last_row = window.iloc[-1]
    return LiveTeamWindow(
        team_name=str(last_row["team_name"]),
        team_id=int(last_row["team_id"]),
        is_home=is_home_team,
        period_value=int(last_row["period_value"]),
        window_start_minute=start_t,
        window_end_minute=end_t,
        score_diff=round(float(last_row["score_diff"])),
        raw_features=window[list(feature_names)].to_numpy(dtype=np.float32),
    )


def extract_latest_windows(
    raw_data: dict[str, Any],
    feature_names: Sequence[str],
    sequence_length: int = 10,
    horizon_minutes: int = 10,
    drop_incomplete_minute: bool = True,
) -> dict[str, LiveTeamWindow]:
    """
    Extracts the latest ``sequence_length``-minute window in the current half for Home and Away.

    Cumulative features are computed over the full match from ``0'`` up to the current minute,
    then only the latest ``L``-minute window inside the active half is returned for inference.

    Args:
        raw_data: Raw WhoScored ``matchCentreData`` dictionary.
        feature_names: Ordered feature names stored inside the trained checkpoint.
        sequence_length: Number of consecutive minutes ``L`` expected by the LSTM.
        horizon_minutes: Horizon ``M`` saved in the checkpoint (passed to panel builder).
        drop_incomplete_minute: When ``True`` (recommended for live matches), ignores the
            final in-progress minute so partial event counts do not bias the prediction down.

    Returns:
        Dictionary ``{"home": LiveTeamWindow, "away": LiveTeamWindow}``.

    Raises:
        ValueError: If the match has no regular-half events or fewer than ``sequence_length``
            completed minutes in the current half.
    """
    panel = _build_live_panel(raw_data, horizon_minutes=horizon_minutes)
    current_period = int(panel["period_value"].max())
    half_panel = panel[panel["period_value"] == current_period]

    return {
        "home": _select_team_window(
            half_panel, "home", True, feature_names, sequence_length, drop_incomplete_minute,
        ),
        "away": _select_team_window(
            half_panel, "away", False, feature_names, sequence_length, drop_incomplete_minute,
        ),
    }


def extract_all_windows(
    raw_data: dict[str, Any],
    feature_names: Sequence[str],
    sequence_length: int = 10,
    horizon_minutes: int = 10,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Slices every valid ``sequence_length``-minute window accumulated so far in each regular half.

    Unlike training ``build_sequences``, this does **not** discard the most recent ``M`` minutes,
    allowing you to inspect probabilities right up to the latest available minute.

    Args:
        raw_data: Raw WhoScored ``matchCentreData`` dictionary.
        feature_names: Ordered feature names stored inside the trained checkpoint.
        sequence_length: Number of consecutive minutes ``L`` per window.
        horizon_minutes: Horizon ``M`` saved in the checkpoint.

    Returns:
        Tuple ``(raw_x, meta_df)`` where ``raw_x`` has shape ``(N, sequence_length, F)`` and
        ``meta_df`` contains alignment metadata for every window.
    """
    panel = _build_live_panel(raw_data, horizon_minutes=horizon_minutes)
    panel = panel.sort_values(["match_id", "team_id", "period_value", "t"]).reset_index(drop=True)

    feat_mat = panel[list(feature_names)].to_numpy(dtype=np.float32)
    minutes = panel["t"].to_numpy(dtype=np.int32)
    span = sequence_length - 1

    windows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    grouped = panel.groupby(["match_id", "team_id", "period_value"], sort=False).groups

    for idx in grouped.values():
        pos = np.asarray(list(idx), dtype=np.int64)
        for local_i in range(span, len(pos)):
            end_pos = int(pos[local_i])
            start_pos = int(pos[local_i - span])
            if int(minutes[end_pos]) - int(minutes[start_pos]) != span:
                continue
            row = panel.iloc[end_pos]
            windows.append(feat_mat[pos[local_i - span : local_i + 1]])
            meta_rows.append({
                "match_id": int(row["match_id"]),
                "team_id": int(row["team_id"]),
                "team_name": str(row["team_name"]),
                "is_home": float(row["is_home"]),
                "period_value": int(row["period_value"]),
                "window_start_minute": int(minutes[start_pos]),
                "minute": int(minutes[end_pos]),
                "score_diff": round(float(row["score_diff"])),
            })

    if not windows:
        empty_x = np.empty((0, sequence_length, len(feature_names)), dtype=np.float32)
        return empty_x, pd.DataFrame(columns=[
            "match_id", "team_id", "team_name", "is_home",
            "period_value", "window_start_minute", "minute", "score_diff",
        ])

    return np.stack(windows, axis=0).astype(np.float32), pd.DataFrame(meta_rows)


def scale_sequence(
    raw_features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    clip_z: float = 5.0,
) -> np.ndarray:
    """
    Standardizes a ``(L, F)`` or ``(N, L, F)`` array with the training scaler statistics.

    Args:
        raw_features: Unscaled feature tensor.
        feature_mean: Per-feature mean loaded from the checkpoint.
        feature_std: Per-feature standard deviation loaded from the checkpoint.
        clip_z: Symmetric z-score clipping threshold (default ``5.0``).

    Returns:
        Scaled ``float32`` NumPy array with the same shape as ``raw_features``.
    """
    mean = np.asarray(feature_mean, dtype=np.float32)
    std = np.asarray(feature_std, dtype=np.float32)
    safe_std = np.where((std > 1e-8) & np.isfinite(std), std, 1.0).astype(np.float32)
    scaled = (raw_features.astype(np.float32) - mean) / safe_std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scaled, -clip_z, clip_z).astype(np.float32)


def calibrate_logit(
    logit: float | np.ndarray,
    platt_a: float,
    platt_b: float,
) -> float | np.ndarray:
    """
    Converts raw model logits into calibrated probabilities via Platt scaling.

    Args:
        logit: Raw logit scalar or NumPy array returned by ``GoalPredictorLSTM``.
        platt_a: Slope parameter fitted on validation logits.
        platt_b: Intercept parameter fitted on validation logits.

    Returns:
        Calibrated probability (or array of probabilities) in ``(0, 1)``.
    """
    probs = expit(platt_a * np.asarray(logit, dtype=np.float64) + platt_b)
    if np.ndim(logit) == 0:
        return float(probs)
    return probs.astype(np.float64)


def load_lstm_checkpoint(
    model_path: Path | str | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """
    Loads the trained ``GoalPredictorLSTM`` weights and metadata from a ``.pt`` checkpoint.

    PyTorch is imported lazily here so feature extraction, URL parsing, ``pydoc``, and
    non-torch unit tests work even in environments without ``torch`` installed.

    Args:
        model_path: Path to the ``.pt`` checkpoint. Defaults to
            ``models/lstm/lstm_goal_predictor.pt`` (auto-resolving ``models/LSTM/...``).

    Returns:
        Tuple ``(predict_logits_fn, checkpoint_dict)`` where ``predict_logits_fn`` accepts a
        scaled NumPy array of shape ``(batch, L, F)`` and returns raw logits ``(batch,)``.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ImportError: If ``torch`` is not installed in the active Python environment.
    """
    resolved = resolve_default_model_path(model_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Trained LSTM checkpoint not found at '{resolved}'. "
            "Place your .pt file in 'models/lstm/' or pass --model <path>.",
        )

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to run inference on '.pt' checkpoints. "
            "Install it in your .venv with: pip install torch",
        ) from exc

    class GoalPredictorLSTM(nn.Module):
        """Architecture matching ``notebooks/06_lstm.ipynb`` and ``07_lstm_colab.ipynb``."""

        def __init__(
            self,
            num_features: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h_n, _) = self.lstm(x)
            return self.head(self.dropout(h_n[-1])).squeeze(-1)

    ckpt: dict[str, Any] = torch.load(resolved, map_location="cpu", weights_only=False)
    model = GoalPredictorLSTM(
        num_features=len(ckpt["feature_names"]),
        hidden_size=int(ckpt["hidden_size"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def predict_logits(scaled_x: np.ndarray) -> np.ndarray:
        tensor_x = torch.tensor(scaled_x, dtype=torch.float32)
        with torch.no_grad():
            logits = model(tensor_x)
        return logits.cpu().numpy().astype(np.float64)

    return predict_logits, ckpt


def predict_match_snapshot(
    raw_data: dict[str, Any],
    predict_logits_fn: Callable[[np.ndarray], np.ndarray],
    ckpt: dict[str, Any],
    drop_incomplete_minute: bool = True,
) -> LiveMatchPrediction:
    """
    Runs a live next-``M``-minute goal prediction on the latest window of a match.

    Args:
        raw_data: Raw WhoScored ``matchCentreData`` dictionary.
        predict_logits_fn: Callable mapping scaled ``(batch, L, F)`` arrays to logits ``(batch,)``.
        ckpt: Checkpoint dictionary containing ``feature_names``, ``feature_mean``,
            ``feature_std``, ``sequence_length``, ``horizon_minutes``, ``platt_a``, ``platt_b``.
        drop_incomplete_minute: Ignore the last in-progress minute (recommended for live polling).

    Returns:
        ``LiveMatchPrediction`` with calibrated probabilities for Home, Away, and Either Team.
    """
    seq_len = int(ckpt["sequence_length"])
    horizon = int(ckpt["horizon_minutes"])
    windows = extract_latest_windows(
        raw_data,
        feature_names=ckpt["feature_names"],
        sequence_length=seq_len,
        horizon_minutes=horizon,
        drop_incomplete_minute=drop_incomplete_minute,
    )

    batch_raw = np.stack([windows["home"].raw_features, windows["away"].raw_features], axis=0)
    batch_scaled = scale_sequence(batch_raw, ckpt["feature_mean"], ckpt["feature_std"])
    logits = predict_logits_fn(batch_scaled)

    platt_a = float(ckpt["platt_a"])
    platt_b = float(ckpt["platt_b"])
    p_home = float(calibrate_logit(float(logits[0]), platt_a, platt_b))
    p_away = float(calibrate_logit(float(logits[1]), platt_a, platt_b))
    p_any = 1.0 - (1.0 - p_home) * (1.0 - p_away)

    home_win, away_win = windows["home"], windows["away"]
    return LiveMatchPrediction(
        match_id=int(raw_data.get("matchId") or 0),
        horizon_minutes=horizon,
        sequence_length=seq_len,
        home=TeamPrediction(
            team_name=home_win.team_name,
            team_id=home_win.team_id,
            is_home=True,
            period_value=home_win.period_value,
            window_start_minute=home_win.window_start_minute,
            window_end_minute=home_win.window_end_minute,
            score_diff=home_win.score_diff,
            raw_logit=float(logits[0]),
            goal_probability=p_home,
        ),
        away=TeamPrediction(
            team_name=away_win.team_name,
            team_id=away_win.team_id,
            is_home=False,
            period_value=away_win.period_value,
            window_start_minute=away_win.window_start_minute,
            window_end_minute=away_win.window_end_minute,
            score_diff=away_win.score_diff,
            raw_logit=float(logits[1]),
            goal_probability=p_away,
        ),
        any_goal_probability=p_any,
    )


def predict_match_history(
    raw_data: dict[str, Any],
    predict_logits_fn: Callable[[np.ndarray], np.ndarray],
    ckpt: dict[str, Any],
) -> pd.DataFrame:
    """
    Scores every valid ``L``-minute window observed so far in the match.

    Args:
        raw_data: Raw WhoScored ``matchCentreData`` dictionary.
        predict_logits_fn: Callable mapping scaled ``(batch, L, F)`` arrays to logits ``(batch,)``.
        ckpt: Checkpoint dictionary loaded from ``.pt``.

    Returns:
        DataFrame with one row per ``(period_value, minute)`` containing ``home_team``,
        ``away_team``, ``score_diff``, ``p_home``, ``p_away``, and ``p_any_goal``.
    """
    raw_x, meta_df = extract_all_windows(
        raw_data,
        feature_names=ckpt["feature_names"],
        sequence_length=int(ckpt["sequence_length"]),
        horizon_minutes=int(ckpt["horizon_minutes"]),
    )
    if len(meta_df) == 0:
        return pd.DataFrame(columns=_HISTORY_COLUMNS)

    scaled_x = scale_sequence(raw_x, ckpt["feature_mean"], ckpt["feature_std"])
    logits = predict_logits_fn(scaled_x)
    probs = calibrate_logit(logits, float(ckpt["platt_a"]), float(ckpt["platt_b"]))

    scored = meta_df.copy()
    scored["prob"] = probs

    home_df = scored[scored["is_home"] >= 0.5].rename(
        columns={"team_name": "home_team", "prob": "p_home"},
    )
    away_df = scored[scored["is_home"] < 0.5].rename(
        columns={"team_name": "away_team", "prob": "p_away"},
    )
    merged = home_df[["period_value", "minute", "home_team", "score_diff", "p_home"]].merge(
        away_df[["period_value", "minute", "away_team", "p_away"]],
        on=["period_value", "minute"],
        how="inner",
    )
    merged["p_any_goal"] = 1.0 - (1.0 - merged["p_home"]) * (1.0 - merged["p_away"])
    return merged[_HISTORY_COLUMNS].sort_values(["period_value", "minute"]).reset_index(drop=True)


def format_prediction(pred: LiveMatchPrediction) -> str:
    """Formats a ``LiveMatchPrediction`` into a concise human-readable terminal block."""
    h, a = pred.home, pred.away
    window_str = f"{h.window_start_minute}'-{h.window_end_minute}'"
    next_str = f"{h.window_end_minute + 1}'-{h.window_end_minute + pred.horizon_minutes}'"
    header = (
        f"Match {pred.match_id} | Half {h.period_value} | Window {window_str} "
        f"-> Next {pred.horizon_minutes} min ({next_str})"
    )
    return "\n".join([
        "=" * 72,
        header,
        f"  Home ({h.team_name:<22}): {h.goal_probability:6.2%}  (score_diff: {h.score_diff:+d})",
        f"  Away ({a.team_name:<22}): {a.goal_probability:6.2%}  (score_diff: {a.score_diff:+d})",
        f"  Any team scores next {pred.horizon_minutes}m     : {pred.any_goal_probability:6.2%}",
        "=" * 72,
    ])


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for live or offline LSTM match prediction."""
    cli = argparse.ArgumentParser(
        description=(
            "Scrape a live WhoScored match (or load an offline JSON) and predict goal "
            "probabilities over the next M minutes using a trained PyTorch LSTM checkpoint."
        ),
    )
    source_group = cli.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--url",
        "-u",
        help="WhoScored Match Centre URL or numeric match ID.",
    )
    source_group.add_argument(
        "--from-json",
        "-j",
        help="Path to an offline raw match JSON file (skips launching Chrome).",
    )
    cli.add_argument(
        "--model",
        "-m",
        default=None,
        help="Path to .pt checkpoint (default: models/lstm/lstm_goal_predictor.pt).",
    )
    cli.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help="Keep Chrome open and poll the live match repeatedly.",
    )
    cli.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds when --watch is enabled (default: 60).",
    )
    cli.add_argument(
        "--history",
        action="store_true",
        help="Print minute-by-minute predictions for all 10-minute windows accumulated so far.",
    )
    cli.add_argument(
        "--include-current-minute",
        action="store_true",
        help="Include the latest (possibly incomplete) minute in live snapshot predictions.",
    )
    cli.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (visible mode is recommended to avoid Cloudflare).",
    )
    return cli.parse_args(argv)


def _run_offline(
    args: argparse.Namespace,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    ckpt: dict[str, Any],
) -> None:
    """Executes prediction on a saved local raw JSON file."""
    raw_data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    if args.history:
        hist = predict_match_history(raw_data, predict_fn, ckpt)
        print(hist.to_string(index=False))
        return
    pred = predict_match_snapshot(raw_data, predict_fn, ckpt, drop_incomplete_minute=False)
    print(format_prediction(pred))


def _run_live(
    args: argparse.Namespace,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    ckpt: dict[str, Any],
) -> None:
    """Scrapes WhoScored via Chrome and prints either one snapshot, history, or watch loop."""
    match_id = extract_match_id_from_url(args.url)
    drop_incomplete = not args.include_current_minute

    with WhoScoredScraper(headless=args.headless) as scraper:
        while True:
            try:
                raw_data = scraper.scrape_match(match_id)
                if args.history and not args.watch:
                    hist = predict_match_history(raw_data, predict_fn, ckpt)
                    print(hist.to_string(index=False))
                else:
                    pred = predict_match_snapshot(
                        raw_data, predict_fn, ckpt, drop_incomplete_minute=drop_incomplete,
                    )
                    print(format_prediction(pred))
            except (WhoScoredScraperError, ValueError) as exc:
                logger.warning("Prediction skipped for matchId=%s: %s", match_id, exc)

            if not args.watch:
                break
            time.sleep(max(10, args.interval))


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for ``python -m src.predict_live``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_cli_args(argv)
    predict_fn, ckpt = load_lstm_checkpoint(args.model)

    if args.from_json:
        _run_offline(args, predict_fn, ckpt)
    else:
        _run_live(args, predict_fn, ckpt)


__all__ = [
    "DEFAULT_MODEL_CANDIDATES",
    "LiveMatchPrediction",
    "LiveTeamWindow",
    "TeamPrediction",
    "calibrate_logit",
    "extract_all_windows",
    "extract_latest_windows",
    "extract_match_id_from_url",
    "format_prediction",
    "load_lstm_checkpoint",
    "main",
    "parse_cli_args",
    "predict_match_history",
    "predict_match_snapshot",
    "resolve_default_model_path",
    "scale_sequence",
]


if __name__ == "__main__":
    main()
