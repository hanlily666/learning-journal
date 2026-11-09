"""Scores for one evaluation run, from the judge's matches between items and my labels."""

from dataclasses import dataclass
from typing import Any

FIELDS = ("concept", "gap", "explore", "source")


@dataclass
class Judged:
    session_id: str
    expected_label: str
    predicted_labels: list[str]  # one per segment (none for the baseline, which doesn't classify)
    labels: list[dict[str, Any]]  # my labelled items: field, text
    predicted: list[dict[str, Any]]  # extracted items: field, text, segment
    matches: list[tuple[int, int]]  # (predicted index, label index), as the judge saw them
    duplicates: list[tuple[int, int]]  # pairs of predicted items that say the same thing


def _ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def score(sessions: list[Judged]) -> dict[str, Any]:
    """Extraction is scored on the sessions I labelled AI, so the agent and the baseline are
    compared on the same sessions. Items pulled from other sessions are counted apart."""
    counts = {f: {"predicted": 0, "labelled": 0, "matched": 0} for f in FIELDS}
    duplicates = off_topic = 0
    for s in sessions:
        if s.expected_label != "ai":
            off_topic += len(s.predicted)
            continue
        for item in s.predicted:
            counts[item["field"]]["predicted"] += 1
        for item in s.labels:
            counts[item["field"]]["labelled"] += 1
        # One-to-one, same field only: the judge can't double-count.
        used_p, used_l = set(), set()
        for p, l in s.matches:
            if (p in used_p or l in used_l or not 0 <= p < len(s.predicted)
                    or not 0 <= l < len(s.labels)
                    or s.predicted[p]["field"] != s.labels[l]["field"]):
                continue
            used_p.add(p)
            used_l.add(l)
            counts[s.predicted[p]["field"]]["matched"] += 1
        duplicates += sum(1 for a, b in set(map(tuple, map(sorted, s.duplicates)))
                          if 0 <= a < len(s.predicted) and 0 <= b < len(s.predicted)
                          and s.predicted[a]["segment"] != s.predicted[b]["segment"])
    fields = {f: {**c, "precision": _ratio(c["matched"], c["predicted"]),
                  "recall": _ratio(c["matched"], c["labelled"])} for f, c in counts.items()}
    labelled = [(s.expected_label, p) for s in sessions for p in s.predicted_labels]
    correct = sum(1 for expected, predicted in labelled if expected == predicted)
    return {"sessions": len(sessions), "classification_accuracy": _ratio(correct, len(labelled)),
            "fields": fields, "duplicates_across_segments": duplicates,
            "items_from_non_ai_sessions": off_topic}
