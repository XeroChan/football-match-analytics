"""
test_parser_coverage.py
=======================
Checks that every key path in the raw `matchCentreData` JSON backups (`data/raw/*_raw.json`)
is read by `MatchDataParser.parse_all()`, and verifies that qualifier values and merged
`incidentEvents` fields are preserved without silent drops or conflicts.

How it works:
    The raw JSON is wrapped in `TrackingDict`, a `dict` subclass that records every key whose
    value is read (`.get`, `[...]`, `.items()`, `.values()`). Mere key enumeration (`.keys()`,
    `iter(d)`) does NOT mark keys as read. After parsing, every key path in the file is
    compared with the recorded paths.

Path notation:
    `$` is the root, `.key` is a dict key, `[]` is any list element and `<n>` is any numeric
    dict key (minutes, IDs, periods), e.g. `$.home.players[].stats.ratings.<n>`.

If a path is skipped on purpose, add it to `INTENTIONALLY_SKIPPED` along with the reason.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from src.parser import MatchDataParser
from src.parsers.qualifiers import extract_qualifiers_info

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILES = sorted(RAW_DIR.glob("*_raw.json"))

# Exact paths (or path patterns) the parser is allowed to ignore. Matched with `re.fullmatch`.
INTENTIONALLY_SKIPPED: dict[str, str] = {
    r".*\.\$idx": "WhoScored internal array index",
    r".*\.\$len": "WhoScored internal array length",
    r"\$\.(home|away)\.stats\.minutesWithStats": (
        "Redundant minute index list; derivable from team_stats_df['minute']"
    ),
}

_NUMERIC_KEY = re.compile(r"^-?\d+$")


def _norm(key: Any) -> str:
    return "<n>" if _NUMERIC_KEY.match(str(key)) else str(key)


class TrackingDict(dict):
    """
    `dict` that records which of its keys had their values read into a shared `accessed` set.
    Key enumeration (`keys()`, `__iter__`) intentionally does NOT mark keys as read.
    When `.items()` or `.values()` is called on a parent dict that contains nested dicts,
    only child dicts whose contents are subsequently accessed (or leaf dicts whose scalar
    values are consumed) are marked as read.
    """

    def __init__(self, data: dict[str, Any], path: str, accessed: set[str]) -> None:
        super().__init__({k: _wrap(v, f"{path}.{_norm(k)}", accessed) for k, v in data.items()})
        self._path = path
        self._accessed = accessed

    def _hit(self, key: Any) -> None:
        if self._path != "$":
            self._accessed.add(self._path)
        if super().__contains__(key):
            self._accessed.add(f"{self._path}.{_norm(key)}")

    def _hit_leaf_entries(self) -> None:
        if self._path != "$":
            self._accessed.add(self._path)
        if not any(isinstance(v, TrackingDict) for v in dict.values(self)):
            for key in dict.keys(self):
                self._hit(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self._hit(key)
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        self._hit(key)
        return super().__getitem__(key)

    def items(self):  # type: ignore[override]
        self._hit_leaf_entries()
        return super().items()

    def values(self):  # type: ignore[override]
        self._hit_leaf_entries()
        return super().values()


def _wrap(value: Any, path: str, accessed: set[str]) -> Any:
    if isinstance(value, dict):
        return TrackingDict(value, path, accessed)
    if isinstance(value, list):
        return [_wrap(v, f"{path}[]", accessed) for v in value]
    return value


def _all_paths(value: Any, path: str, out: set[str]) -> set[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{_norm(key)}"
            out.add(child_path)
            _all_paths(child, child_path, out)
    elif isinstance(value, list):
        for child in value:
            _all_paths(child, f"{path}[]", out)
    return out


def _is_skipped(path: str) -> bool:
    return any(re.fullmatch(pattern, path) for pattern in INTENTIONALLY_SKIPPED)


def find_unparsed_paths(raw: dict[str, Any]) -> list[str]:
    """Returns every key path in `raw` that `MatchDataParser.parse_all()` never reads."""
    accessed: set[str] = set()
    MatchDataParser().parse_all(_wrap(raw, "$", accessed))
    all_paths = _all_paths(raw, "$", set())
    return sorted(p for p in all_paths - accessed if not _is_skipped(p))


def test_tracking_dict_detects_unread_keys() -> None:
    """Sanity check: reading some keys or enumerating `.keys()` does not mask unread keys."""
    accessed: set[str] = set()
    sample = {"a": 1, "b": {"c": 2, "d": 3}, "lst": [{"e": 1}]}
    data = _wrap(sample, "$", accessed)
    _ = data.get("a"), data["b"]["c"], list(data["b"].keys()), [x for x in data["lst"]]
    raw_paths = _all_paths(sample, "$", set())
    assert sorted(raw_paths - accessed) == ["$.b.d", "$.lst[].e"]


@pytest.mark.skipif(not RAW_FILES, reason=f"No raw JSON files in {RAW_DIR}")
def test_merged_incident_events_detects_unread_keys() -> None:
    """
    Regression check: injecting an unknown field into an `incidentEvents` entry or a
    team `stats` dict must be caught by `find_unparsed_paths` (not masked by `.items()`).
    """
    raw = copy.deepcopy(json.loads(RAW_FILES[0].read_text(encoding="utf-8")))
    raw["home"]["incidentEvents"][0]["unhandledIncidentField"] = "surprise"
    raw["home"]["stats"]["unhandledTeamStatScalar"] = 42

    missing = find_unparsed_paths(raw)
    assert "$.home.incidentEvents[].unhandledIncidentField" in missing
    assert "$.home.stats.unhandledTeamStatScalar" in missing


def test_qualifiers_preserve_all_names_type_ids_and_values() -> None:
    """
    Checks that `extract_qualifiers_info` preserves every qualifier displayName, numeric
    type ID, and key-value entry in `qualifiers_json` alongside dedicated columns.
    """
    qualifiers = [
        {"type": {"displayName": "PassEndX", "value": 140}, "value": "72.4"},
        {"type": {"displayName": "PassEndY", "value": 141}, "value": "31.8"},
        {"type": {"displayName": "KeyPass", "value": 210}},
        {"type": {"displayName": "LeftFoot", "value": 72}},
        {"type": {"displayName": "StandingSave", "value": 178}},
    ]
    info = extract_qualifiers_info(qualifiers)
    assert info["qualifier_pass_end_x"] == "72.4"
    assert info["qualifier_pass_end_y"] == "31.8"
    assert info["is_key_pass"] is True
    assert info["body_part"] == "LeftFoot"
    assert info["qualifier_names"] == "KeyPass|LeftFoot|PassEndX|PassEndY|StandingSave"
    assert info["qualifier_type_ids"] == "72|140|141|178|210"
    assert json.loads(info["qualifiers_json"]) == {
        "PassEndX": "72.4",
        "PassEndY": "31.8",
        "KeyPass": True,
        "LeftFoot": True,
        "StandingSave": True,
    }


@pytest.mark.skipif(not RAW_FILES, reason=f"No raw JSON files in {RAW_DIR}")
def test_incident_events_do_not_conflict_with_primary_events() -> None:
    """
    Verifies that when an event ID appears in both `events[]` and `home/away.incidentEvents[]`,
    shared scalar fields do not hold conflicting non-null values that would be silently shadowed.
    """
    raw = json.loads(RAW_FILES[0].read_text(encoding="utf-8"))
    events_by_id = {
        ev["id"]: ev
        for ev in raw.get("events") or []
        if isinstance(ev, dict) and ev.get("id") is not None
    }
    for side in ("home", "away"):
        for inc in (raw.get(side) or {}).get("incidentEvents") or []:
            ev = events_by_id.get(inc.get("id"))
            if not ev:
                continue
            for key, inc_val in inc.items():
                if key.startswith("$") or isinstance(inc_val, (dict, list)) or inc_val is None:
                    continue
                ev_val = ev.get(key)
                if ev_val is not None:
                    assert ev_val == inc_val, f"Conflicting '{key}' on event id={inc.get('id')}"


@pytest.mark.skipif(not RAW_FILES, reason=f"No raw JSON files in {RAW_DIR}")
@pytest.mark.parametrize("raw_file", RAW_FILES, ids=[f.stem for f in RAW_FILES])
def test_parser_reads_every_raw_json_field(raw_file: Path) -> None:
    raw = json.loads(raw_file.read_text(encoding="utf-8"))
    missing = find_unparsed_paths(raw)
    assert not missing, (
        f"{len(missing)} raw JSON key path(s) in {raw_file.name} are never read by the parser "
        "(handle them in src/parsers/ or add them to INTENTIONALLY_SKIPPED):\n  "
        + "\n  ".join(missing)
    )


if __name__ == "__main__":
    # `python tests/test_parser_coverage.py` -> union report across all raw files.
    union: set[str] = set()
    for f in RAW_FILES:
        union.update(find_unparsed_paths(json.loads(f.read_text(encoding="utf-8"))))
    print(f"{len(union)} unparsed key path(s) across {len(RAW_FILES)} file(s):")
    for p in sorted(union):
        print("  ", p)
