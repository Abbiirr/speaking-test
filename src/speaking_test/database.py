"""SQLite persistence for IELTS practice history (speaking + writing)."""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from speaking_test.models import AttemptRecord, SessionRecord

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "history.db"
CSV_PATH = Path(__file__).resolve().parent.parent.parent / "questions_answers_updated.csv"


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
            band9_answer TEXT DEFAULT '',
            strengths TEXT DEFAULT '',
            pronunciation_warnings TEXT DEFAULT '',
            source TEXT DEFAULT '',
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)

    # Questions table — seeded from CSV once
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            part INTEGER NOT NULL,
            topic TEXT DEFAULT '',
            question_text TEXT NOT NULL,
            cue_card TEXT DEFAULT '',
            source TEXT DEFAULT '',
            band9_answer TEXT DEFAULT '',
            answer_variant TEXT DEFAULT ''
        )
    """)

    # --- Writing module tables ---
    conn.executescript("""
        -- PDF document tracking (hash-based idempotency)
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_hash TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            doc_type TEXT NOT NULL DEFAULT 'cambridge_book',
            page_count INTEGER NOT NULL,
            parser TEXT NOT NULL DEFAULT 'pymupdf',
            parser_version TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );

        -- Per-page extracted text
        CREATE TABLE IF NOT EXISTS document_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            page_no INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(id),
            UNIQUE (doc_id, page_no)
        );

        -- Extracted assets (full-page images)
        CREATE TABLE IF NOT EXISTS document_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            page_no INTEGER NOT NULL,
            asset_type TEXT NOT NULL DEFAULT 'image',
            file_path TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            meta_json TEXT DEFAULT '',
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );

        -- Writing question bank
        CREATE TABLE IF NOT EXISTS writing_prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_type TEXT NOT NULL,
            task_type INTEGER NOT NULL,
            topic TEXT DEFAULT '',
            prompt_text TEXT NOT NULL,
            source_doc_id INTEGER,
            source_page_no INTEGER,
            chart_asset_id INTEGER,
            task1_data_json TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (source_doc_id) REFERENCES documents(id),
            FOREIGN KEY (chart_asset_id) REFERENCES document_assets(id)
        );

        -- Model/sample answers for calibration
        CREATE TABLE IF NOT EXISTS writing_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id INTEGER NOT NULL,
            band REAL NOT NULL,
            essay_text TEXT NOT NULL,
            examiner_notes TEXT DEFAULT '',
            source TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (prompt_id) REFERENCES writing_prompts(id)
        );

        -- Writing attempt records
        CREATE TABLE IF NOT EXISTS writing_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            prompt_id INTEGER NOT NULL,
            task_type INTEGER NOT NULL,
            essay_text TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            task_score REAL NOT NULL,
            coherence_score REAL NOT NULL,
            lexical_score REAL NOT NULL,
            grammar_score REAL NOT NULL,
            overall_band REAL NOT NULL,
            examiner_feedback TEXT DEFAULT '',
            paragraph_feedback TEXT DEFAULT '',
            grammar_corrections TEXT DEFAULT '',
            vocabulary_upgrades TEXT DEFAULT '',
            improvement_tips TEXT DEFAULT '',
            provider TEXT NOT NULL,
            raw_json TEXT DEFAULT '',
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (prompt_id) REFERENCES writing_prompts(id)
        );
    """)

    # FTS5 full-text search over page text (separate statement — virtual tables
    # cannot be created inside executescript on some SQLite builds)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS document_pages_fts
        USING fts5(text, doc_id UNINDEXED, page_no UNINDEXED,
                   content='document_pages', content_rowid='id')
    """)

    # Migrate existing databases: add new columns if missing
    _new_columns = [
        ("band9_answer", "TEXT DEFAULT ''"),
        ("strengths", "TEXT DEFAULT ''"),
        ("pronunciation_warnings", "TEXT DEFAULT ''"),
        ("source", "TEXT DEFAULT ''"),
    ]
    for col_name, col_type in _new_columns:
        try:
            conn.execute(f"ALTER TABLE attempts ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists


def _seed_questions(conn: sqlite3.Connection) -> None:
    """Import CSV questions into the questions table (once, if empty)."""
    row = conn.execute("SELECT COUNT(*) as cnt FROM questions").fetchone()
    if row["cnt"] > 0:
        return  # Already seeded

    if not CSV_PATH.exists():
        return

    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append((
                int(r["part"]),
                r.get("topic", "").strip(),
                r["question"].strip(),
                r.get("cue_card", "").strip(),
                r.get("source", "").strip(),
                r.get("band9_answer", "").strip(),
                r.get("answer_variant", "").strip(),
            ))

    conn.executemany(
        "INSERT INTO questions (part, topic, question_text, cue_card, source, "
        "band9_answer, answer_variant) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


_conn: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _get_connection()
        _init_db(_conn)
        _seed_questions(_conn)
    return _conn


# ---------------------------------------------------------------------------
# Question loading from DB
# ---------------------------------------------------------------------------

def get_all_questions_from_db() -> list[dict]:
    """Load all unique questions from the DB, one random answer variant per question.

    Returns list of dicts with keys: part, topic, question_text, cue_card, source,
    band9_answer.
    """
    conn = get_db()
    # Get all distinct (part, question_text) groups with their answers
    rows = conn.execute(
        "SELECT part, topic, question_text, cue_card, source, band9_answer, answer_variant "
        "FROM questions ORDER BY part, question_text, answer_variant"
    ).fetchall()

    # Group by (part, question_text), pick one random variant per question
    import random
    groups: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        key = (r["part"], r["question_text"])
        if key not in groups:
            groups[key] = []
        groups[key].append(dict(r))

    result = []
    for (_part, _text), variants in groups.items():
        chosen = random.choice(variants)
        result.append({
            "part": chosen["part"],
            "topic": chosen["topic"],
            "question_text": chosen["question_text"],
            "cue_card": chosen["cue_card"],
            "source": chosen["source"],
            "band9_answer": chosen["band9_answer"],
        })
    return result


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
            grammar_corrections, vocabulary_upgrades, improvement_tips,
            band9_answer, strengths, pronunciation_warnings, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            record.band9_answer,
            record.strengths,
            record.pronunciation_warnings,
            record.source,
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
        for field in (
            "grammar_corrections", "vocabulary_upgrades", "improvement_tips",
            "strengths", "pronunciation_warnings",
        ):
            val = d.get(field, "")
            if val:
                try:
                    d[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        result.append(d)
    return result


def get_detailed_weaknesses(limit: int = 50) -> dict:
    """Aggregate weakness data from recent attempts (no LLM calls).

    Returns a dict with:
    - grammar_errors: list of (original, corrected, count) tuples — most frequent
    - basic_words: list of (word, count) tuples — most common words to upgrade
    - criterion_trends: dict of criterion -> {"avg": float, "direction": str}
    - recurring_tips: list of (tip, count) tuples — most repeated tips
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT grammar_corrections, vocabulary_upgrades, improvement_tips, "
        "fluency_coherence, lexical_resource, grammatical_range, pronunciation, "
        "id FROM attempts ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()

    if not rows:
        return {}

    # Parse grammar errors
    from collections import Counter
    grammar_counter: Counter = Counter()
    for r in rows:
        gc_raw = r["grammar_corrections"] or ""
        if gc_raw:
            try:
                corrections = json.loads(gc_raw) if isinstance(gc_raw, str) else gc_raw
                if isinstance(corrections, list):
                    for item in corrections:
                        if isinstance(item, dict):
                            orig = item.get("original", "").strip()
                            corr = item.get("corrected", "").strip()
                            if orig and corr:
                                grammar_counter[(orig, corr)] += 1
            except (json.JSONDecodeError, TypeError):
                pass

    # Parse basic words to upgrade
    word_counter: Counter = Counter()
    for r in rows:
        vu_raw = r["vocabulary_upgrades"] or ""
        if vu_raw:
            try:
                upgrades = json.loads(vu_raw) if isinstance(vu_raw, str) else vu_raw
                if isinstance(upgrades, list):
                    for item in upgrades:
                        if isinstance(item, dict):
                            word = item.get("basic_word", "").strip().lower()
                            if word:
                                word_counter[word] += 1
            except (json.JSONDecodeError, TypeError):
                pass

    # Parse recurring tips
    tip_counter: Counter = Counter()
    for r in rows:
        tips_raw = r["improvement_tips"] or ""
        if tips_raw:
            try:
                tips = json.loads(tips_raw) if isinstance(tips_raw, str) else tips_raw
                if isinstance(tips, list):
                    for tip in tips:
                        if isinstance(tip, str) and tip.strip():
                            tip_counter[tip.strip()] += 1
            except (json.JSONDecodeError, TypeError):
                pass

    # Criterion trends — compare first half vs second half of attempts
    criteria = {
        "Fluency & Coherence": "fluency_coherence",
        "Lexical Resource": "lexical_resource",
        "Grammar": "grammatical_range",
        "Pronunciation": "pronunciation",
    }
    criterion_trends = {}
    rows_list = list(reversed(rows))  # oldest first
    mid = len(rows_list) // 2
    for label, col in criteria.items():
        all_vals = [r[col] for r in rows_list if r[col] is not None and r[col] > 0]
        if not all_vals:
            continue
        avg = round(sum(all_vals) / len(all_vals), 1)
        if mid > 0 and len(rows_list) >= 4:
            first_half = [r[col] for r in rows_list[:mid] if r[col] and r[col] > 0]
            second_half = [r[col] for r in rows_list[mid:] if r[col] and r[col] > 0]
            if first_half and second_half:
                avg1 = sum(first_half) / len(first_half)
                avg2 = sum(second_half) / len(second_half)
                diff = avg2 - avg1
                if diff > 0.3:
                    direction = "improving"
                elif diff < -0.3:
                    direction = "declining"
                else:
                    direction = "stable"
            else:
                direction = "insufficient data"
        else:
            direction = "insufficient data"
        criterion_trends[label] = {"avg": avg, "direction": direction}

    return {
        "grammar_errors": [
            {"original": k[0], "corrected": k[1], "count": v}
            for k, v in grammar_counter.most_common(5)
        ],
        "basic_words": [
            {"word": k, "count": v}
            for k, v in word_counter.most_common(5)
        ],
        "criterion_trends": criterion_trends,
        "recurring_tips": [
            {"tip": k, "count": v}
            for k, v in tip_counter.most_common(5)
        ],
    }


# ---------------------------------------------------------------------------
# Writing module — database functions
# ---------------------------------------------------------------------------

def save_writing_attempt(session_id: int, attempt_data: dict) -> int:
    """Insert a writing attempt record and update session stats."""
    logger.info(
        "Saving writing attempt: session=%d, task=%d, band=%.1f, words=%d",
        session_id, attempt_data.get("task_type", 0),
        attempt_data.get("overall_band", 0), attempt_data.get("word_count", 0),
    )
    conn = get_db()
    ts = attempt_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO writing_attempts (
            session_id, timestamp, prompt_id, task_type, essay_text, word_count,
            task_score, coherence_score, lexical_score, grammar_score, overall_band,
            examiner_feedback, paragraph_feedback, grammar_corrections,
            vocabulary_upgrades, improvement_tips, provider, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            ts,
            attempt_data["prompt_id"],
            attempt_data["task_type"],
            attempt_data["essay_text"],
            attempt_data["word_count"],
            attempt_data["task_score"],
            attempt_data["coherence_score"],
            attempt_data["lexical_score"],
            attempt_data["grammar_score"],
            attempt_data["overall_band"],
            attempt_data.get("examiner_feedback", ""),
            attempt_data.get("paragraph_feedback", ""),
            attempt_data.get("grammar_corrections", ""),
            attempt_data.get("vocabulary_upgrades", ""),
            attempt_data.get("improvement_tips", ""),
            attempt_data.get("provider", ""),
            attempt_data.get("raw_json", ""),
        ),
    )
    conn.commit()
    _update_session_stats(session_id)
    return cursor.lastrowid


def get_writing_attempts(session_id: int | None = None) -> list[dict]:
    """Query writing attempts, optionally filtered by session."""
    conn = get_db()
    if session_id is not None:
        rows = conn.execute(
            "SELECT * FROM writing_attempts WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM writing_attempts ORDER BY id DESC LIMIT 100"
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        for field in ("paragraph_feedback", "grammar_corrections",
                       "vocabulary_upgrades", "improvement_tips"):
            val = d.get(field, "")
            if val:
                try:
                    d[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        result.append(d)
    return result


def get_all_writing_prompts() -> list[dict]:
    """Return all prompts from the writing_prompts table."""
    conn = get_db()
    rows = conn.execute(
        "SELECT wp.*, da.file_path AS chart_image_path "
        "FROM writing_prompts wp "
        "LEFT JOIN document_assets da ON wp.chart_asset_id = da.id "
        "ORDER BY wp.id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_writing_prompt_by_id(prompt_id: int) -> dict | None:
    """Single prompt lookup with resolved chart image path."""
    conn = get_db()
    row = conn.execute(
        "SELECT wp.*, da.file_path AS chart_image_path "
        "FROM writing_prompts wp "
        "LEFT JOIN document_assets da ON wp.chart_asset_id = da.id "
        "WHERE wp.id = ?",
        (prompt_id,),
    ).fetchone()
    return dict(row) if row else None


def search_document_pages(query: str, limit: int = 20) -> list[dict]:
    """FTS5 search over ingested document pages with snippets."""
    logger.debug("FTS search: query=%r, limit=%d", query, limit)
    conn = get_db()
    rows = conn.execute(
        "SELECT dp.doc_id, dp.page_no, d.file_name, "
        "snippet(document_pages_fts, 0, '<b>', '</b>', '...', 40) AS snippet "
        "FROM document_pages_fts fts "
        "JOIN document_pages dp ON dp.id = fts.rowid "
        "JOIN documents d ON d.id = dp.doc_id "
        "WHERE document_pages_fts MATCH ? "
        "ORDER BY rank LIMIT ?",
        (query, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_document_list() -> list[dict]:
    """List all ingested documents."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM documents ORDER BY ingested_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_document_page_assets(doc_id: int, page_no: int) -> list[dict]:
    """Get image assets for a specific document page."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM document_assets "
        "WHERE doc_id = ? AND page_no = ? ORDER BY id",
        (doc_id, page_no),
    ).fetchall()
    return [dict(r) for r in rows]


def get_writing_samples(prompt_id: int) -> list[dict]:
    """Get model/sample answers for a writing prompt."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM writing_samples WHERE prompt_id = ? ORDER BY band DESC",
        (prompt_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_writing_criterion_trends(limit: int = 50) -> list[dict]:
    """Writing band trends over time (4 criteria + overall)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT timestamp, task_score, coherence_score, lexical_score, "
        "grammar_score, overall_band FROM writing_attempts "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_writing_weaknesses() -> dict:
    """Aggregate writing weakness analysis from recent attempts."""
    conn = get_db()
    rows = conn.execute(
        "SELECT grammar_corrections, vocabulary_upgrades, improvement_tips, "
        "task_score, coherence_score, lexical_score, grammar_score, id "
        "FROM writing_attempts ORDER BY id DESC LIMIT 50"
    ).fetchall()

    if not rows:
        return {}

    from collections import Counter

    grammar_counter: Counter = Counter()
    word_counter: Counter = Counter()
    tip_counter: Counter = Counter()

    for r in rows:
        gc_raw = r["grammar_corrections"] or ""
        if gc_raw:
            try:
                corrections = json.loads(gc_raw) if isinstance(gc_raw, str) else gc_raw
                if isinstance(corrections, list):
                    for item in corrections:
                        if isinstance(item, dict):
                            orig = item.get("original", "").strip()
                            corr = item.get("corrected", "").strip()
                            if orig and corr:
                                grammar_counter[(orig, corr)] += 1
            except (json.JSONDecodeError, TypeError):
                pass

        vu_raw = r["vocabulary_upgrades"] or ""
        if vu_raw:
            try:
                upgrades = json.loads(vu_raw) if isinstance(vu_raw, str) else vu_raw
                if isinstance(upgrades, list):
                    for item in upgrades:
                        if isinstance(item, dict):
                            word = item.get("basic_word", "").strip().lower()
                            if word:
                                word_counter[word] += 1
            except (json.JSONDecodeError, TypeError):
                pass

        tips_raw = r["improvement_tips"] or ""
        if tips_raw:
            try:
                tips = json.loads(tips_raw) if isinstance(tips_raw, str) else tips_raw
                if isinstance(tips, list):
                    for tip in tips:
                        if isinstance(tip, str) and tip.strip():
                            tip_counter[tip.strip()] += 1
            except (json.JSONDecodeError, TypeError):
                pass

    criteria = {
        "Task Achievement": "task_score",
        "Coherence & Cohesion": "coherence_score",
        "Lexical Resource": "lexical_score",
        "Grammar": "grammar_score",
    }
    criterion_trends = {}
    rows_list = list(reversed(rows))
    mid = len(rows_list) // 2
    for label, col in criteria.items():
        all_vals = [r[col] for r in rows_list if r[col] is not None and r[col] > 0]
        if not all_vals:
            continue
        avg = round(sum(all_vals) / len(all_vals), 1)
        if mid > 0 and len(rows_list) >= 4:
            first_half = [r[col] for r in rows_list[:mid] if r[col] and r[col] > 0]
            second_half = [r[col] for r in rows_list[mid:] if r[col] and r[col] > 0]
            if first_half and second_half:
                avg1 = sum(first_half) / len(first_half)
                avg2 = sum(second_half) / len(second_half)
                diff = avg2 - avg1
                if diff > 0.3:
                    direction = "improving"
                elif diff < -0.3:
                    direction = "declining"
                else:
                    direction = "stable"
            else:
                direction = "insufficient data"
        else:
            direction = "insufficient data"
        criterion_trends[label] = {"avg": avg, "direction": direction}

    return {
        "grammar_errors": [
            {"original": k[0], "corrected": k[1], "count": v}
            for k, v in grammar_counter.most_common(5)
        ],
        "basic_words": [
            {"word": k, "count": v}
            for k, v in word_counter.most_common(5)
        ],
        "criterion_trends": criterion_trends,
        "recurring_tips": [
            {"tip": k, "count": v}
            for k, v in tip_counter.most_common(5)
        ],
    }
