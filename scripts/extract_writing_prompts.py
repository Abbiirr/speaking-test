#!/usr/bin/env python3
"""Extract writing prompts and sample answers from ingested PDF pages.

Runs AFTER ingest_pdf.py — searches extracted text for IELTS Writing sections
using FTS and pattern matching.

Usage:
    uv run python scripts/extract_writing_prompts.py
    uv run python scripts/extract_writing_prompts.py --test-type academic
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger("speaking_test.extract_writing_prompts")

from speaking_test.database import get_db


# Patterns to detect writing task headers
TASK_HEADER_PATTERNS = [
    re.compile(r"WRITING\s+TASK\s+([12])", re.IGNORECASE),
    re.compile(r"Writing\s+Task\s+([12])"),
    re.compile(r"TASK\s+([12])\s*\n", re.IGNORECASE),
]

# Patterns for model/sample answers
SAMPLE_ANSWER_PATTERNS = [
    re.compile(r"(?:Model|Sample)\s+[Aa]nswer", re.IGNORECASE),
    re.compile(r"Band\s+(\d(?:\.\d)?)\s+(?:answer|essay|response)", re.IGNORECASE),
]

# Patterns to detect section boundaries (stop extracting prompt text here)
SECTION_BOUNDARY = re.compile(
    r"(?:READING|LISTENING|SPEAKING|WRITING\s+TASK\s+[12]|PART\s+\d|TEST\s+\d)",
    re.IGNORECASE,
)


def _extract_prompt_text(page_text: str, task_match: re.Match) -> str:
    """Extract the prompt text following a task header match."""
    start = task_match.end()
    remaining = page_text[start:].strip()

    # Find the next section boundary
    boundary = SECTION_BOUNDARY.search(remaining)
    if boundary:
        remaining = remaining[:boundary.start()].strip()

    # Clean up: remove excessive whitespace
    lines = [ln.strip() for ln in remaining.splitlines() if ln.strip()]
    return "\n".join(lines)


def _find_chart_asset(conn, doc_id: int, page_no: int) -> int | None:
    """Find the page image asset for Task 1 chart display."""
    row = conn.execute(
        "SELECT id FROM document_assets "
        "WHERE doc_id = ? AND page_no = ? AND asset_type = 'image' "
        "ORDER BY id LIMIT 1",
        (doc_id, page_no),
    ).fetchone()
    return row["id"] if row else None


def _prompt_exists(conn, doc_id: int, page_no: int, task_type: int) -> bool:
    """Check if a prompt from this source already exists."""
    row = conn.execute(
        "SELECT id FROM writing_prompts "
        "WHERE source_doc_id = ? AND source_page_no = ? AND task_type = ?",
        (doc_id, page_no, task_type),
    ).fetchone()
    return row is not None


def extract_prompts(conn, test_type: str = "academic") -> dict:
    """Scan all ingested pages for writing prompts and insert them."""
    ts = datetime.now(timezone.utc).isoformat()

    # Get all pages
    pages = conn.execute(
        "SELECT dp.id, dp.doc_id, dp.page_no, dp.text, d.file_name "
        "FROM document_pages dp "
        "JOIN documents d ON d.id = dp.doc_id "
        "ORDER BY dp.doc_id, dp.page_no"
    ).fetchall()

    prompts_found = 0
    prompts_skipped = 0
    samples_found = 0

    for page in pages:
        text = page["text"]
        doc_id = page["doc_id"]
        page_no = page["page_no"]

        # Search for task headers
        for pattern in TASK_HEADER_PATTERNS:
            for match in pattern.finditer(text):
                task_type = int(match.group(1))
                prompt_text = _extract_prompt_text(text, match)

                if not prompt_text or len(prompt_text) < 20:
                    continue

                if _prompt_exists(conn, doc_id, page_no, task_type):
                    prompts_skipped += 1
                    continue

                # For Task 1, link the page image as chart
                chart_asset_id = None
                if task_type == 1:
                    chart_asset_id = _find_chart_asset(conn, doc_id, page_no)

                # Detect topic from first line
                first_line = prompt_text.split("\n")[0][:80]
                topic = first_line if len(first_line) < 60 else ""

                conn.execute(
                    "INSERT INTO writing_prompts (test_type, task_type, topic, "
                    "prompt_text, source_doc_id, source_page_no, chart_asset_id, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (test_type, task_type, topic, prompt_text,
                     doc_id, page_no, chart_asset_id, ts),
                )
                prompts_found += 1
                logger.info(
                    "Found Task %d prompt on page %d of %s",
                    task_type, page_no, page["file_name"],
                )
                print(
                    f"  Found Task {task_type} prompt on page {page_no} "
                    f"of {page['file_name']}"
                )

        # Search for sample/model answers (look for answer sections)
        for pattern in SAMPLE_ANSWER_PATTERNS:
            for match in pattern.finditer(text):
                answer_text = text[match.end():].strip()
                # Trim to reasonable length
                boundary = SECTION_BOUNDARY.search(answer_text)
                if boundary:
                    answer_text = answer_text[:boundary.start()].strip()

                if len(answer_text) < 50:
                    continue

                # Try to extract band score
                band = 0.0
                band_match = re.search(r"Band\s+(\d(?:\.\d)?)", match.group(0))
                if band_match:
                    band = float(band_match.group(1))

                # Find the most recent prompt for this document
                recent_prompt = conn.execute(
                    "SELECT id FROM writing_prompts "
                    "WHERE source_doc_id = ? ORDER BY id DESC LIMIT 1",
                    (doc_id,),
                ).fetchone()

                if recent_prompt:
                    # Check for duplicate
                    existing = conn.execute(
                        "SELECT id FROM writing_samples "
                        "WHERE prompt_id = ? AND essay_text = ?",
                        (recent_prompt["id"], answer_text[:200]),
                    ).fetchone()

                    if not existing:
                        conn.execute(
                            "INSERT INTO writing_samples (prompt_id, band, "
                            "essay_text, source, created_at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (recent_prompt["id"], band, answer_text,
                             page["file_name"], ts),
                        )
                        samples_found += 1

    conn.commit()
    return {
        "prompts_found": prompts_found,
        "prompts_skipped": prompts_skipped,
        "samples_found": samples_found,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract writing prompts from ingested PDFs"
    )
    parser.add_argument(
        "--test-type", default="academic", choices=["academic", "gt"],
        help="IELTS test type (default: academic)",
    )
    args = parser.parse_args()

    conn = get_db()

    # Check that documents exist
    doc_count = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()["cnt"]
    if doc_count == 0:
        print(
            "No documents found in database. Run ingest_pdf.py first:\n"
            "  uv run --extra pdf python scripts/ingest_pdf.py pdf/",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Scanning {doc_count} document(s) for writing prompts...\n")
    result = extract_prompts(conn, test_type=args.test_type)

    print(f"\nResults:")
    print(f"  Prompts found:   {result['prompts_found']}")
    print(f"  Prompts skipped: {result['prompts_skipped']} (already in DB)")
    print(f"  Samples found:   {result['samples_found']}")


if __name__ == "__main__":
    main()
