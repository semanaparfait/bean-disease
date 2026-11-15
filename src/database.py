"""SQLite persistence: uploaded training data, retraining runs and a prediction log."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config as C

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    label TEXT NOT NULL,
    path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    used_in_training INTEGER NOT NULL DEFAULT 0,
    training_run_id INTEGER
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    n_new_images INTEGER,
    n_total_images INTEGER,
    metrics TEXT,
    message TEXT,
    model_version TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    latency_ms REAL NOT NULL,
    model_version TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    with _lock:
        conn = sqlite3.connect(C.DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- uploads
def add_upload(filename: str, label: str, path: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO uploads (filename, label, path, uploaded_at) VALUES (?, ?, ?, ?)",
            (filename, label, path, now()),
        )
        return cur.lastrowid


def pending_uploads() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM uploads WHERE used_in_training = 0 ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def mark_uploads_used(ids: list[int], run_id: int, new_paths: dict[int, str]) -> None:
    with connect() as conn:
        for i in ids:
            conn.execute(
                "UPDATE uploads SET used_in_training = 1, training_run_id = ?, path = ? WHERE id = ?",
                (run_id, new_paths.get(i), i),
            )


def upload_summary() -> dict:
    with connect() as conn:
        rows = conn.execute(
            "SELECT label, used_in_training, COUNT(*) AS n FROM uploads GROUP BY label, used_in_training"
        ).fetchall()
    out = {"pending": {c: 0 for c in C.CLASS_NAMES}, "used": {c: 0 for c in C.CLASS_NAMES}}
    for r in rows:
        out["used" if r["used_in_training"] else "pending"][r["label"]] = r["n"]
    out["pending_total"] = sum(out["pending"].values())
    out["used_total"] = sum(out["used"].values())
    return out


# ---------------------------------------------------------------- training runs
def start_run(trigger: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO training_runs (trigger, status, started_at) VALUES (?, 'running', ?)",
            (trigger, now()),
        )
        return cur.lastrowid


def finish_run(run_id: int, status: str, message: str, metrics: dict | None = None,
               n_new: int | None = None, n_total: int | None = None, model_version: str | None = None):
    with connect() as conn:
        conn.execute(
            """UPDATE training_runs SET status=?, finished_at=?, message=?, metrics=?, n_new_images=?,
               n_total_images=?, model_version=? WHERE id=?""",
            (status, now(), message, json.dumps(metrics) if metrics else None, n_new, n_total, model_version, run_id),
        )


def list_runs(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    runs = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else None
        runs.append(d)
    return runs


def get_run(run_id: int) -> dict | None:
    runs = [r for r in list_runs(1000) if r["id"] == run_id]
    return runs[0] if runs else None


# ---------------------------------------------------------------- predictions
def log_prediction(predicted_class: str, confidence: float, latency_ms: float, model_version: str):
    with connect() as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, predicted_class, confidence, latency_ms, model_version)"
            " VALUES (?, ?, ?, ?, ?)",
            (now(), predicted_class, confidence, latency_ms, model_version),
        )


def prediction_summary() -> dict:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*), AVG(latency_ms), AVG(confidence) FROM predictions").fetchone()
        by_cls = conn.execute(
            "SELECT predicted_class, COUNT(*) AS n FROM predictions GROUP BY predicted_class"
        ).fetchall()
    return {
        "total": total[0],
        "avg_latency_ms": round(total[1], 1) if total[1] else None,
        "avg_confidence": round(total[2], 3) if total[2] else None,
        "by_class": {r["predicted_class"]: r["n"] for r in by_cls},
    }
