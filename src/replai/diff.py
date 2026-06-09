"""Align two runs' spans and report where they diverged.

Spans are matched by (name, type) using a longest-common-subsequence alignment,
so reordered or inserted/removed steps line up sensibly. Nesting isn't considered
in the alignment — runs are compared as their flattened, time-ordered step lists.
"""
from __future__ import annotations

_FIELDS = ("output", "model", "tokens_in", "tokens_out", "error")


def diff_runs(spans_a: list[dict], spans_b: list[dict]) -> list[dict]:
    """Return aligned rows comparing two ordered span lists.

    Each row: {"status": same|changed|added|removed, "a": span|None,
               "b": span|None, "changes": {field: [old, new]}}.
    """
    keys_a = [_key(s) for s in spans_a]
    keys_b = [_key(s) for s in spans_b]
    rows: list[dict] = []
    i = j = 0
    for mi, mj in _lcs(keys_a, keys_b) + [(len(spans_a), len(spans_b))]:
        while i < mi:
            rows.append({"status": "removed", "a": spans_a[i], "b": None, "changes": {}})
            i += 1
        while j < mj:
            rows.append({"status": "added", "a": None, "b": spans_b[j], "changes": {}})
            j += 1
        if mi < len(spans_a) and mj < len(spans_b):
            a, b = spans_a[mi], spans_b[mj]
            changes = _field_changes(a, b)
            rows.append({
                "status": "changed" if changes else "same",
                "a": a, "b": b, "changes": changes,
            })
            i, j = mi + 1, mj + 1
    return rows


def summarize(spans: list[dict]) -> dict:
    """Aggregate totals for a run's spans."""
    return {
        "steps": len(spans),
        "tokens_in": sum(s.get("tokens_in") or 0 for s in spans),
        "tokens_out": sum(s.get("tokens_out") or 0 for s in spans),
        "errors": sum(1 for s in spans if s.get("error")),
    }


def _key(span: dict):
    return (span.get("name"), span.get("type"))


def _field_changes(a: dict, b: dict) -> dict:
    return {f: [a.get(f), b.get(f)] for f in _FIELDS if a.get(f) != b.get(f)}


def _lcs(a: list, b: list) -> list[tuple[int, int]]:
    """Aligned (i, j) index pairs of the longest common subsequence of a and b."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    pairs = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs
