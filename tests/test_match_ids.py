"""
test_match_ids.py
=================
Checks that configured WhoScored match IDs in `src.match_ids` contain no duplicates
within any competition, across competitions, or in `DEFAULT_MATCH_IDS`.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from src.match_ids import DEFAULT_MATCH_IDS, MATCH_IDS_BY_COMPETITION


def test_no_duplicate_match_ids_within_competition() -> None:
    """Checks that no competition list in `MATCH_IDS_BY_COMPETITION` repeats a match ID."""
    duplicates: list[str] = []
    for competition, match_ids in MATCH_IDS_BY_COMPETITION.items():
        counts = Counter(match_ids)
        repeated = sorted((mid, cnt) for mid, cnt in counts.items() if cnt > 1)
        if repeated:
            details = ", ".join(f"{mid} (x{cnt})" for mid, cnt in repeated)
            duplicates.append(f"{competition}: {details}")

    assert not duplicates, (
        "Duplicate match ID(s) found within competitions:\n  " + "\n  ".join(duplicates)
    )


def test_no_duplicate_match_ids_across_competitions() -> None:
    """Checks that no match ID appears in more than one competition."""
    competitions_by_id: dict[int, list[str]] = defaultdict(list)
    for competition, match_ids in MATCH_IDS_BY_COMPETITION.items():
        for match_id in set(match_ids):
            competitions_by_id[match_id].append(competition)

    cross_duplicates = sorted(
        (mid, comps) for mid, comps in competitions_by_id.items() if len(comps) > 1
    )
    assert not cross_duplicates, (
        "Match ID(s) found in multiple competitions:\n  "
        + "\n  ".join(f"{mid}: {', '.join(comps)}" for mid, comps in cross_duplicates)
    )


def test_default_match_ids_has_no_duplicates() -> None:
    """Checks that the flattened `DEFAULT_MATCH_IDS` list contains only unique IDs."""
    counts = Counter(DEFAULT_MATCH_IDS)
    duplicates = sorted((mid, cnt) for mid, cnt in counts.items() if cnt > 1)
    assert not duplicates, (
        "Duplicate match ID(s) found in DEFAULT_MATCH_IDS:\n  "
        + "\n  ".join(f"{mid} (x{cnt})" for mid, cnt in duplicates)
    )
