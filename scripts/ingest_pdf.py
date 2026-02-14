#!/usr/bin/env python3
"""Standalone PDF ingestion script — reads PDFs, extracts text + page images.

Usage:
    uv run --extra pdf python scripts/ingest_pdf.py pdf/
    uv run --extra pdf python scripts/ingest_pdf.py pdf/cambridge-ielts-19.pdf

This script is NOT part of the speaking_test package. It imports pymupdf
(which is an optional dev dependency) and writes to the shared SQLite DB
plus the pdf_extracted/ filesystem directory.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger("speaking_test.ingest_pdf")

try:
    import pymupdf  # noqa: F401
except ImportError:
    print(
        "ERROR: pymupdf is not installed. Install it with:\n"
        "  uv pip install pymupdf>=1.25.0\n"
        "  # or: uv sync --extra pdf",
        file=sys.stderr,
    )
    sys.exit(1)

from speaking_test.database import get_db

PARSER_NAME = "pymupdf"
PARSER_VERSION = pymupdf.VersionBind  # e.g. "1.25.1"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "pdf_extracted"


def _slugify(name: str) -> str:
    """Convert a filename to a safe directory slug."""
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_pdf(pdf_path: Path, conn: sqlite3.Connection) -> dict:
    """Ingest a single PDF file. Returns summary dict."""
    logger.info("Ingesting PDF: %s", pdf_path.name)
    pdf_bytes = pdf_path.read_bytes()
    doc_hash = _sha256(pdf_bytes)

    # Check for duplicate (hash + parser_version)
    existing = conn.execute(
        "SELECT id FROM documents WHERE doc_hash = ? AND parser_version = ?",
        (doc_hash, PARSER_VERSION),
    ).fetchone()
    if existing:
        logger.info("Skipping %s — already ingested (hash=%s)", pdf_path.name, doc_hash[:12])
        return {"file": pdf_path.name, "status": "skipped", "reason": "already ingested"}

    doc = pymupdf.open(str(pdf_path))
    page_count = len(doc)
    slug = _slugify(pdf_path.name)
    out_dir = OUTPUT_DIR / slug
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()

    # Insert document record
    cursor = conn.execute(
        "INSERT INTO documents (doc_hash, file_name, doc_type, page_count, "
        "parser, parser_version, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (doc_hash, pdf_path.name, "cambridge_book", page_count,
         PARSER_NAME, PARSER_VERSION, ts),
    )
    doc_id = cursor.lastrowid

    pages_md_lines: list[str] = []
    pages_json_list: list[dict] = []
    images_saved = 0

    for page_no in range(page_count):
        page = doc[page_no]
        text = page.get_text()

        # Save text to DB
        conn.execute(
            "INSERT INTO document_pages (doc_id, page_no, text) VALUES (?, ?, ?)",
            (doc_id, page_no, text),
        )

        # Populate FTS index
        page_row = conn.execute(
            "SELECT id FROM document_pages WHERE doc_id = ? AND page_no = ?",
            (doc_id, page_no),
        ).fetchone()
        if page_row:
            conn.execute(
                "INSERT INTO document_pages_fts (rowid, text, doc_id, page_no) "
                "VALUES (?, ?, ?, ?)",
                (page_row["id"], text, doc_id, page_no),
            )

        # Render full-page image
        pix = page.get_pixmap(dpi=144)
        img_name = f"page_{page_no:04d}.png"
        img_path = img_dir / img_name
        pix.save(str(img_path))
        images_saved += 1

        # Store relative path from project root
        rel_path = img_path.relative_to(
            Path(__file__).resolve().parent.parent
        ).as_posix()

        conn.execute(
            "INSERT INTO document_assets (doc_id, page_no, asset_type, file_path, "
            "width, height) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, page_no, "image", rel_path, pix.width, pix.height),
        )

        # Build markdown + JSON
        pages_md_lines.append(f"## Page {page_no}\n\n{text}\n")
        pages_json_list.append({"page_no": page_no, "text": text})

    conn.commit()
    doc.close()

    # Write filesystem outputs
    (out_dir / "pages.md").write_text("\n".join(pages_md_lines), encoding="utf-8")
    (out_dir / "pages.json").write_text(
        json.dumps(pages_json_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "Ingested %s: doc_id=%d, pages=%d, images=%d",
        pdf_path.name, doc_id, page_count, images_saved,
    )
    return {
        "file": pdf_path.name,
        "status": "ingested",
        "doc_id": doc_id,
        "pages": page_count,
        "images": images_saved,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest IELTS PDFs into the database")
    parser.add_argument(
        "paths", nargs="+", type=Path,
        help="PDF file(s) or directory containing PDFs",
    )
    args = parser.parse_args()

    # Collect all PDF paths
    pdf_files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            pdf_files.extend(sorted(p.glob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdf_files.append(p)
        else:
            print(f"WARNING: skipping {p} (not a PDF or directory)", file=sys.stderr)

    if not pdf_files:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)

    conn = get_db()
    print(f"Found {len(pdf_files)} PDF file(s) to process.\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name} ... ", end="", flush=True)
        result = ingest_pdf(pdf_path, conn)
        if result["status"] == "skipped":
            print(f"SKIPPED ({result['reason']})")
        else:
            print(f"OK — {result['pages']} pages, {result['images']} images")

    print("\nDone. Output written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
