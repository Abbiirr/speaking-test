"""SQLite persistence for IELTS Speaking practice history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from speaking_test.models import AttemptRecord, SessionRecord

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "history.db"


def _get_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            mode TEXT NOT NULL,
            overall_band REAL DEFAULT 0.0,
            attempt_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            part INTEGER DEFAULT 0,
            topic TEXT DEFAULT '',
            question_text TEXT DEFAULT '',
            transcript TEXT DEFAULT '',
            duration REAL DEFAULT 0.0,
            overall_band REAL DEFAULT 0.0,
            fluency_coherence REAL DEFAULT 0.0,
            lexical_resource REAL DEFAULT 0.0,
            grammatical_range REAL DEFAULT 0.0,
            pronunciation REAL DEFAULT 0.0,
            speech_rate REAL DEFAULT 0.0,
            pause_ratio REAL DEFAULT 0.0,
            pronunciation_confidence REAL DEFAULT 0.0,
            examiner_feedback TEXT DEFAULT '',
            grammar_corrections TEXT DEFAULT '',
            vocabulary_upgrades TEXT DEFAULT '',
            improvement_tips TEXT DEFAULT '',
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)


_conn: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _get_connection()
        _init_db(_conn)
    return _conn


def create_session(mode: str) -> int:
    conn = get_db()
    ts = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO sessions (timestamp, mode) VALUES (?, ?)",
        (ts, mode),
    )
    conn.commit()
    return cursor.lastrowid


def save_attempt(record: AttemptRecord) -> int:
    conn = get_db()
    if not record.timestamp:
        record.timestamp = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO attempts (
            session_id, timestamp, part, topic, question_text, transcript,
            duration, overall_band, fluency_coherence, lexical_resource,
            grammatical_range, pronunciation, speech_rate, pause_ratio,
            pronunciation_confidence, examiner_feedback,
            grammar_corrections, vocabulary_upgrades, improvement_tips
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record.session_id,
            record.timestamp,
            record.part,
            record.topic,
            record.question_text,
            record.transcript,
            record.duration,
            record.overall_band,
            record.fluency_coherence,
            record.lexical_resource,
            record.grammatical_range,
            record.pronunciation,
            record.speech_rate,
            record.pause_ratio,
            record.pronunciation_confidence,
            record.examiner_feedback,
            record.grammar_corrections,
            record.vocabulary_upgrades,
            record.improvement_tips,
        ),
    )
    conn.commit()

    # Update session aggregates
    _update_session_stats(record.session_id)
    return cursor.lastrowid


def _update_session_stats(session_id: int) -> None:
    conn = get_db()
    row = conn.execute(
        "SELECT COUNT(*) as cnt, AVG(overall_band) as avg_band "
        "FROM attempts WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE sessions SET attempt_count = ?, overall_band = ? WHERE id = ?",
            (row["cnt"], round(row["avg_band"] * 2) / 2, session_id),
        )
        conn.commit()


def get_band_trend(limit: int = 50) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT timestamp, overall_band FROM attempts "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_criterion_trends(limit: int = 50) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT timestamp, fluency_coherence, lexical_resource, "
        "grammatical_range, pronunciation FROM attempts "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_weak_areas() -> dict[str, float]:
    conn = get_db()
    row = conn.execute(
        "SELECT "
        "AVG(fluency_coherence) as fluency_coherence, "
        "AVG(lexical_resource) as lexical_resource, "
        "AVG(grammatical_range) as grammatical_range, "
        "AVG(pronunciation) as pronunciation "
        "FROM (SELECT * FROM attempts ORDER BY id DESC LIMIT 20)",
    ).fetchone()
    if not row or row["fluency_coherence"] is None:
        return {}
    return {
        "Fluency & Coherence": round(row["fluency_coherence"], 1),
        "Lexical Resource": round(row["lexical_resource"], 1),
        "Grammar": round(row["grammatical_range"], 1),
        "Pronunciation": round(row["pronunciation"], 1),
    }


def get_recent_sessions(limit: int = 20) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_attempts_for_session(session_id: int) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM attempts WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        # Parse JSON fields
        for field in ("grammar_corrections", "vocabulary_upgrades", "improvement_tips"):
            val = d.get(field, "")
            if val:
                try:
                    d[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        result.append(d)
    return result
