#!/usr/bin/env python3
"""Load writing prompts from seed JSON into the database."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speaking_test.database import get_db


def main():
    seed_path = Path(__file__).parent / "seed_writing_prompts.json"
    if not seed_path.exists():
        print(f"ERROR: {seed_path} not found. Create it first.")
        sys.exit(1)

    with open(seed_path, encoding="utf-8") as f:
        prompts = json.load(f)

    conn = get_db()
    ts = datetime.now(timezone.utc).isoformat()
    inserted = 0
    skipped = 0
    samples_inserted = 0

    for entry in prompts:
        # Deduplicate by prompt_text + task_type
        existing = conn.execute(
            "SELECT id FROM writing_prompts WHERE prompt_text = ? AND task_type = ?",
            (entry["prompt_text"], entry["task_type"]),
        ).fetchone()
        if existing:
            prompt_id = existing["id"]
            skipped += 1
        else:
            cursor = conn.execute(
                "INSERT INTO writing_prompts (test_type, task_type, topic, prompt_text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (entry["test_type"], entry["task_type"], entry.get("topic", ""),
                 entry["prompt_text"], ts),
            )
            prompt_id = cursor.lastrowid
            inserted += 1

        # Insert samples/model answers
        for sample in entry.get("samples", []):
            # Check for duplicate (compare first 200 chars of essay)
            essay_prefix = sample["essay_text"][:200]
            existing_sample = conn.execute(
                "SELECT id FROM writing_samples WHERE prompt_id = ? AND essay_text LIKE ?",
                (prompt_id, essay_prefix + "%"),
            ).fetchone()
            if not existing_sample:
                conn.execute(
                    "INSERT INTO writing_samples (prompt_id, band, essay_text, examiner_notes, source, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (prompt_id, sample["band"], sample["essay_text"],
                     sample.get("examiner_notes", ""), entry.get("source", ""), ts),
                )
                samples_inserted += 1

    conn.commit()
    print(f"Done: {inserted} prompts inserted, {skipped} skipped (duplicates), {samples_inserted} samples inserted")


if __name__ == "__main__":
    main()
