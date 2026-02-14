#!/usr/bin/env python3
"""Extract writing-related pages from text-native PDFs.

Outputs structured text to stdout AND saves to pdf_extracted/{slug}/:
  - writing_pages.md   — human-readable markdown
  - writing_pages.json — machine-readable JSON
  - images/page_NNNN.png — page images for writing-related pages (Task 1 charts)

Run: uv run --extra pdf python scripts/extract_text_writing.py
"""
import json
import pymupdf
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Text-native PDFs where get_text() works directly
TEXT_NATIVE_PDFS = [
    ("Cambridge IELTS 3.pdf", "Cambridge IELTS 3"),
    ("Cambridge IELTS 18.pdf", "Cambridge IELTS 18"),
    ("IELTS1.pdf", "IELTS 1"),
    ("ielts2.pdf", "IELTS 2"),
    ("Cambridge IELTS4.01.pdf", "Cambridge IELTS 4 (Test 1)"),
]

WRITING_PATTERN = re.compile(
    r"WRITING TASK [12]|MODEL ANSWER|SAMPLE ANSWER|Model answer|Sample answer|"
    r"General Training: Writing|General Training Writing",
    re.IGNORECASE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "pdf_extracted"


def _slugify(name: str) -> str:
    """Convert a PDF filename to a directory slug."""
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug


def extract_pdf(pdf_path: str, label: str):
    doc = pymupdf.open(pdf_path)
    slug = _slugify(Path(pdf_path).name)
    out_dir = EXTRACTED_DIR / slug
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Collect writing pages
    md_lines: list[str] = []
    json_pages: list[dict] = []

    header = f"# {label} — {Path(pdf_path).name} ({len(doc)} pages)"
    md_lines.append(f"{'='*60}")
    md_lines.append(header)
    md_lines.append(f"{'='*60}")

    # Also print to stdout
    print(f"\n{'='*60}")
    print(f"# {label} — {Path(pdf_path).name} ({len(doc)} pages)")
    print(f"{'='*60}")

    total_pages = len(doc)

    for i in range(total_pages):
        text = doc[i].get_text()
        if WRITING_PATTERN.search(text):
            # Clean up common PDF encoding artifacts
            text = text.replace("\u2019", "'").replace("\u2018", "'")
            text = text.replace("\u201c", '"').replace("\u201d", '"')

            md_lines.append(f"\n=== PAGE {i} ===")
            md_lines.append(text)

            json_pages.append({"page": i, "text": text})

            print(f"\n=== PAGE {i} ===")
            print(text)

            # Save page image (useful for Task 1 charts/graphs)
            pix = doc[i].get_pixmap(dpi=150)
            img_path = img_dir / f"page_{i:04d}.png"
            pix.save(str(img_path))

    doc.close()

    # Write markdown file
    md_path = out_dir / "writing_pages.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Write JSON file
    json_path = out_dir / "writing_pages.json"
    json_data = {
        "label": label,
        "filename": Path(pdf_path).name,
        "slug": slug,
        "page_count": total_pages,
        "writing_pages": json_pages,
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  -> Saved {len(json_pages)} writing pages to {out_dir}/", file=sys.stderr)


def main():
    pdf_dir = PROJECT_ROOT / "pdf"

    target = sys.argv[1] if len(sys.argv) > 1 else None

    for filename, label in TEXT_NATIVE_PDFS:
        if target and target not in filename and target not in label:
            continue
        path = pdf_dir / filename
        if path.exists():
            extract_pdf(str(path), label)
        else:
            print(f"SKIP: {path} not found", file=sys.stderr)


if __name__ == "__main__":
    main()
