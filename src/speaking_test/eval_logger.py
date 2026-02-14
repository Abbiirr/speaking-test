"""File-based evaluation logger for model comparison.

Writes one JSON file per evaluation into a date/time-ordered folder hierarchy:

    logs/evaluations/YYYY-MM-DD/HHMMSS_mode/eval_NNN.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Base directory — same parent as the existing logs/app.log
_BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "evaluations"
)

# session_id -> folder path
_session_dirs: dict[int, str] = {}

# session_id -> next eval sequence number
_session_counters: dict[int, int] = {}


def init_eval_session(session_id: int, mode: str) -> str:
    """Create a date/time_mode folder for this session and register it.

    Returns the absolute path to the session folder.
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")
    folder_name = f"{time_str}_{mode}"

    session_dir = os.path.join(_BASE_DIR, date_str, folder_name)
    os.makedirs(session_dir, exist_ok=True)

    _session_dirs[session_id] = session_dir
    _session_counters[session_id] = 1

    logger.info("Eval session %d initialized: %s", session_id, session_dir)
    return session_dir


def log_evaluation(session_id: int, data: dict) -> str | None:
    """Write the next sequentially numbered eval JSON into the session folder.

    Returns the file path written, or None if the session was not initialized.
    """
    session_dir = _session_dirs.get(session_id)
    if session_dir is None:
        logger.warning(
            "log_evaluation called for uninitialized session %d — skipping",
            session_id,
        )
        return None

    seq = _session_counters.get(session_id, 1)
    filename = f"eval_{seq:03d}.json"
    filepath = os.path.join(session_dir, filename)

    _session_counters[session_id] = seq + 1

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Eval logged: %s", filepath)
    return filepath
