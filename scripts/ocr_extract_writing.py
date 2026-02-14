#!/usr/bin/env python3
"""OCR scanned PDFs to find writing task pages. Outputs text to stdout.

Requires Tesseract OCR installed:
  winget install UB-Mannheim.TesseractOCR

Run one PDF at a time (OCR is slow — ~5-10 sec/page):
  uv run --extra pdf python scripts/ocr_extract_writing.py "Cambridge IELTS 10"

Or run all:
  uv run --extra pdf python scripts/ocr_extract_writing.py
"""
import pymupdf
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WRITING_PATTERN = re.compile(
    r"WRITING\s+TASK\s+[12]|MODEL\s+ANSWER|SAMPLE\s+ANSWER|Band\s+\d",
    re.IGNORECASE,
)

# Image-only PDFs (get_text() returns empty/garbled)
IMAGE_PDFS = [
    "Cambridge IELTS4.02.pdf",
    "Cambridge IELTS4.03.pdf",
    "Cambridge IELTS4.04.pdf",
    "Cambridge IELTS 5 with Answers.pdf",
    "Cambridge ielts 6 test1.pdf",
    "Cambridge ielts 6 test2.pdf",
    "Cambridge ielts 6 test4.pdf",
    "Cambridge IELTS 7 (Book).pdf",
    "Cambridge IELTS 8.pdf",
    "Cambridge IELTS 9.pdf",
    "Cambridge IELTS 10.pdf",
    "Cambridge IELTS 11 - Clear PDF Version.pdf",
    "Cambridge IELTS 12 PDF.pdf",
    "Cambridge IELTS 14.pdf",
    "cambridge_ielts_15.pdf",
    "Cambridge IELTS 15 Gen .pdf",
    "Cambridge IELTS 16 (Academic).pdf",
    "Cambridge IELTS 17 (Academic).pdf",
    "Cambridge IELTS 19 Academic.pdf",
    "IELTS Cambridge 13- pdf.pdf",
    "Cambridge ielts6 Answers.pdf",
    "cambridge 4 ans key.pdf",
]


def ocr_pdf(pdf_path: str):
    doc = pymupdf.open(pdf_path)
    name = Path(pdf_path).name
    print(f"\n{'='*60}")
    print(f"# {name} ({len(doc)} pages)")
    print(f"{'='*60}")

    for i in range(len(doc)):
        page = doc[i]
        # Run OCR via pymupdf's built-in Tesseract binding
        try:
            tp = page.get_textpage_ocr(language="eng", dpi=200)
            text = page.get_text("text", textpage=tp)
        except Exception as e:
            print(f"  OCR error on page {i}: {e}", file=sys.stderr)
            continue

        # Filter for writing-related pages
        if WRITING_PATTERN.search(text):
            print(f"\n=== {name} PAGE {i} ===")
            print(text)

        # Progress indicator to stderr
        if (i + 1) % 10 == 0:
            print(f"  ... {name} page {i+1}/{len(doc)}", file=sys.stderr)

    doc.close()
    print(f"  Done: {name}", file=sys.stderr)


def main():
    pdf_dir = Path(__file__).resolve().parent.parent / "pdf"
    target = sys.argv[1] if len(sys.argv) > 1 else None

    for name in IMAGE_PDFS:
        if target and target.lower() not in name.lower():
            continue
        path = pdf_dir / name
        if path.exists():
            ocr_pdf(str(path))
        else:
            print(f"SKIP: {path} not found", file=sys.stderr)


if __name__ == "__main__":
    main()
