from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .models import Run, Span

DEFAULT_DB = Path(os.environ.get("REPLAI_DB", str(Path.home() / ".replai" / "replai.db")))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT,
    start REAL,
    end REAL,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    parent_id TEXT,
    name TEXT,
    type TEXT,
    start REAL,
    end REAL,
    input TEXT,
    output TEXT,
    error TEXT,
    model TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    metadata TEXT,
    raw TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id);
"""


def _enc(value) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _dec(value):
    return json.loads(value) if value else None


class Store:
    def __init__(self, path: Optional[str | Path] = None, redact=None):
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._redact = redact
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(spans)")}
        if "raw" not in cols:
            self._conn.execute("ALTER TABLE spans ADD COLUMN raw TEXT")

    def save_run(self, run: Run) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (id, name, start, end, metadata) VALUES (?, ?, ?, ?, ?)",
                (run.id, run.name, run.start, run.end, _enc(run.metadata)),
            )
            self._conn.commit()

    def save_span(self, span: Span) -> None:
        inp = self._apply_redact(span.input)
        out = self._apply_redact(span.output)
        raw = self._apply_redact(span.raw)
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO spans
                   (id, run_id, parent_id, name, type, start, end,
                    input, output, error, model, tokens_in, tokens_out, metadata, raw)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    span.id, span.run_id, span.parent_id, span.name, span.type, span.start, span.end,
                    _enc(inp), _enc(out), span.error, span.model,
                    span.tokens_in, span.tokens_out, _enc(span.metadata), _enc(raw),
                ),
            )
            self._conn.commit()

    def _apply_redact(self, value):
        if self._redact is None:
            return value
        try:
            return self._redact(value)
        except Exception:
            return "<redaction failed>"  # fail closed — never persist the raw value

    def runs(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY start DESC").fetchall()
        return [dict(r) for r in rows]

    def run(self, run_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def spans(self, run_id: str, with_raw: bool = False) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM spans WHERE run_id = ? ORDER BY start", (run_id,)
            ).fetchall()
        result = []
        for row in rows:
            span = dict(row)
            span["input"] = _dec(span["input"])
            span["output"] = _dec(span["output"])
            span["metadata"] = _dec(span["metadata"])
            if with_raw:
                span["raw"] = _dec(span["raw"])
            else:
                span.pop("raw", None)  # large SDK dump — only replay needs it
            span["duration_ms"] = (
                round((span["end"] - span["start"]) * 1000, 1) if span["end"] else None
            )
            result.append(span)
        return result

    def close(self) -> None:
        self._conn.close()
