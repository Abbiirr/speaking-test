#!/usr/bin/env python3
"""Verify writing prompts were seeded correctly into the database."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speaking_test.database import get_db


def main():
    conn = get_db()

    # Count prompts by type
    print("=== Writing Prompts Summary ===")
    rows = conn.execute(
        "SELECT test_type, task_type, COUNT(*) as cnt "
        "FROM writing_prompts GROUP BY test_type, task_type ORDER BY test_type, task_type"
    ).fetchall()
    total = 0
    for r in rows:
        print(f"  {r['test_type']:10s} Task {r['task_type']}: {r['cnt']} prompts")
        total += r["cnt"]
    print(f"  {'TOTAL':10s}        : {total} prompts")

    # Count samples
    print("\n=== Writing Samples Summary ===")
    rows = conn.execute(
        "SELECT ws.source, COUNT(*) as cnt, AVG(ws.band) as avg_band "
        "FROM writing_samples ws GROUP BY ws.source ORDER BY ws.source"
    ).fetchall()
    sample_total = 0
    for r in rows:
        print(f"  {r['source']:40s}: {r['cnt']} samples (avg band {r['avg_band']:.1f})")
        sample_total += r["cnt"]
    print(f"  {'TOTAL':40s}: {sample_total} samples")

    # Show a few prompts as spot-check
    print("\n=== Sample Prompts (first 3) ===")
    rows = conn.execute(
        "SELECT id, test_type, task_type, topic, substr(prompt_text, 1, 80) as preview "
        "FROM writing_prompts ORDER BY id LIMIT 3"
    ).fetchall()
    for r in rows:
        print(f"  #{r['id']} [{r['test_type']}/Task{r['task_type']}] {r['topic']}")
        print(f"    {r['preview']}...")

    # Check for prompts with samples
    print("\n=== Prompts with samples ===")
    row = conn.execute(
        "SELECT COUNT(DISTINCT prompt_id) as cnt FROM writing_samples"
    ).fetchone()
    print(f"  {row['cnt']} prompts have at least one sample/model answer")

    # Check for potential issues
    print("\n=== Quality Checks ===")
    # Empty prompts
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM writing_prompts WHERE length(prompt_text) < 20"
    ).fetchone()
    if row["cnt"] > 0:
        print(f"  WARNING: {row['cnt']} prompts have very short text (<20 chars)")
    else:
        print("  OK: All prompts have reasonable length")

    # Duplicate check
    row = conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT prompt_text) as dups FROM writing_prompts"
    ).fetchone()
    if row["dups"] > 0:
        print(f"  WARNING: {row['dups']} duplicate prompt texts found")
    else:
        print("  OK: No duplicate prompts")

    # Garbled text check
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM writing_prompts WHERE prompt_text LIKE '%\x92%' OR prompt_text LIKE '%\x93%'"
    ).fetchone()
    if row["cnt"] > 0:
        print(f"  WARNING: {row['cnt']} prompts may have garbled characters")
    else:
        print("  OK: No garbled characters detected")

    print("\nDone.")


if __name__ == "__main__":
    main()
