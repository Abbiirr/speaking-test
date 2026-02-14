#!/usr/bin/env python3
"""Build seed_writing_prompts.json by sending INDIVIDUAL pages to Ollama.

Each Ollama call handles exactly one page (~1-3k chars). No long-text issues.

Phase 1: Classify + extract prompts (one page at a time)
Phase 2: Classify + extract answers (one page at a time)
Phase 3: Match answers to prompts, merge, write JSON

Run:
  .venv/Scripts/python.exe scripts/build_seed_json.py
  .venv/Scripts/python.exe scripts/build_seed_json.py --model qwen3:14b
  .venv/Scripts/python.exe scripts/build_seed_json.py --only ielts1 cambridge-ielts-3
"""
import argparse
import json
import re
import requests
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EXTRACTED = PROJECT / "pdf_extracted"
OUT = PROJECT / "scripts" / "seed_writing_prompts.json"
OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_MODEL = "dengcao/Qwen3-30B-A3B-Instruct-2507"

BOOKS = {
    "cambridge-ielts-3": "Cambridge IELTS 3",
    "cambridge-ielts-18": "Cambridge IELTS 18",
    "ielts1": "IELTS 1",
    "ielts2": "IELTS 2",
    "cambridge-ielts4-01": "Cambridge IELTS 4",
}

# ── Regex pre-classification (no LLM needed) ──────────────────────────

PROMPT_RE = re.compile(r"WRITING TASK [12]", re.I)
ANSWER_RE = re.compile(
    r"(?:MODEL|SAMPLE)\s*ANSWER|M\s*O\s*D\s*E\s*L\s*A\s*N\s*S\s*W\s*E\s*R"
    r"|Model and sample answers|Sample Writing answers"
    r"|candidate who achieved a Band|model has been prepared by an examiner",
    re.I,
)
SKIP_RE = re.compile(
    r"^(?:Contents|Introduction|Sample answer sheets|Acknowledgements)",
    re.I | re.M,
)


def classify_page(text: str) -> str:
    """Classify a page as 'prompt', 'answer', or 'skip' using regex only."""
    has_answer = bool(ANSWER_RE.search(text))
    has_prompt = bool(PROMPT_RE.search(text))
    # Pages that are clearly just TOC/intro/answer sheets
    if not has_prompt and not has_answer:
        return "skip"
    if has_answer:
        return "answer"
    if has_prompt:
        return "prompt"
    return "skip"


# ── Ollama calls ──────────────────────────────────────────────────────

def call_ollama(model: str, system: str, user: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def parse_json(text: str) -> dict | list | None:
    """Extract JSON from Ollama response, handling think tags and fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # Try markdown fence
    m = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    m = re.search(r"([\[{].*[\]}])", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


# ── Phase 1: Extract prompts (one page at a time) ────────────────────

PROMPT_SYSTEM = """You extract IELTS writing task prompts from a single PDF page.
Output ONLY a JSON object (no markdown, no explanation):
{
  "task_type": 1 or 2,
  "test_label": "Test 1" or "GT Test A" or "Practice Test 3" etc.,
  "test_type": "academic" or "gt",
  "topic": "Short 5-10 word description",
  "prompt_text": "Full prompt text, cleaned of page numbers and headers"
}

Rules:
- GT Task 1 = letter writing. Academic Task 1 = describe data/chart/diagram/map.
- Task 2 = essay (both academic and GT).
- Clean prompt_text: remove page numbers, "Test 1\\n28", URLs, but keep all instructions.
- If the page has BOTH Task 1 AND Task 2, output a JSON ARRAY of two objects.
- If the page has no valid prompt (just intro text), output: {"skip": true}"""


def extract_prompt(model: str, book_label: str, page_num: int, text: str) -> list[dict]:
    """Send one prompt page to Ollama, return list of prompt dicts."""
    user = f"Book: {book_label}\nPage {page_num}:\n\n{text}"
    print(f"    Page {page_num} (prompt)...", end=" ", flush=True)
    t0 = time.time()
    resp = call_ollama(model, PROMPT_SYSTEM, user)
    elapsed = time.time() - t0
    result = parse_json(resp)
    if result is None:
        print(f"parse error ({elapsed:.0f}s)")
        return []
    if isinstance(result, dict):
        if result.get("skip"):
            print(f"skipped ({elapsed:.0f}s)")
            return []
        result = [result]
    print(f"{len(result)} prompt(s) ({elapsed:.0f}s)")
    # Add source
    for r in result:
        r["source"] = f"{book_label}, {r.pop('test_label', 'Unknown')}"
        r.setdefault("samples", [])
    return result


# ── Phase 2: Extract answers (one page at a time) ────────────────────

ANSWER_SYSTEM = """You extract IELTS model/sample answers from a single PDF page.
Output ONLY a JSON array of answer objects found on this page:
[{
  "test_label": "Test 1" or "GT Test A" etc.,
  "task_type": 1 or 2,
  "band": 5.0,
  "essay_text": "The full essay text ONLY (no examiner comments mixed in)",
  "examiner_notes": "The examiner's comment about this essay",
  "type": "sample" or "model"
}]

Rules:
- "model" = written by an examiner as a good example. Assign band 8.0.
- "sample" = written by a candidate. Use the stated Band score.
- essay_text = ONLY the essay. No "Here is the examiner's comment" text.
- examiner_notes = the comment about the essay quality.
- A page may have 1 or 2 answers (e.g. Task 1 answer and start of Task 2 answer).
- If no valid answer on this page, output: []
- Look for headers like "TEST 1, WRITING TASK 1" or "TEST A, WRITING TASK 2 (GENERAL TRAINING)" to identify which test/task."""


def extract_answer(model: str, book_label: str, page_num: int, text: str) -> list[dict]:
    """Send one answer page to Ollama, return list of answer dicts."""
    user = f"Book: {book_label}\nPage {page_num}:\n\n{text}"
    print(f"    Page {page_num} (answer)...", end=" ", flush=True)
    t0 = time.time()
    resp = call_ollama(model, ANSWER_SYSTEM, user)
    elapsed = time.time() - t0
    result = parse_json(resp)
    if result is None or (isinstance(result, list) and len(result) == 0):
        print(f"none ({elapsed:.0f}s)")
        return []
    if isinstance(result, dict):
        result = [result]
    # Add source
    for r in result:
        r["source"] = f"{book_label}, {r.pop('test_label', 'Unknown')}"
    print(f"{len(result)} answer(s) ({elapsed:.0f}s)")
    return result


# ── Phase 3: Match answers to prompts ────────────────────────────────

def match_answers(prompts: list[dict], answers: list[dict]):
    """Attach each answer to its matching prompt."""
    for ans in answers:
        src = ans.get("source", "")
        task = ans.get("task_type")
        for p in prompts:
            if p["source"] == src and p["task_type"] == task:
                p.setdefault("samples", []).append({
                    "band": ans.get("band", 7.0),
                    "essay_text": ans.get("essay_text", ""),
                    "examiner_notes": ans.get("examiner_notes", ""),
                    "type": ans.get("type", "model"),
                })
                break
        else:
            # Try fuzzy match on test label
            for p in prompts:
                if p["task_type"] == task and _source_fuzzy_match(p["source"], src):
                    p.setdefault("samples", []).append({
                        "band": ans.get("band", 7.0),
                        "essay_text": ans.get("essay_text", ""),
                        "examiner_notes": ans.get("examiner_notes", ""),
                        "type": ans.get("type", "model"),
                    })
                    break


def _source_fuzzy_match(source_a: str, source_b: str) -> bool:
    """Check if two sources refer to the same book+test despite naming differences."""
    # Normalize: "IELTS 1, Practice Test 4" vs "IELTS 1, Test 4"
    def norm(s):
        s = s.lower().replace("practice ", "").replace("  ", " ")
        return s
    return norm(source_a) == norm(source_b)


# ── Main ──────────────────────────────────────────────────────────────

def process_book(model: str, slug: str) -> tuple[list[dict], list[dict]]:
    """Process one book page by page. Returns (prompts, answers)."""
    label = BOOKS[slug]
    json_path = EXTRACTED / slug / "writing_pages.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    pages = data["writing_pages"]
    print(f"\n{'='*60}")
    print(f"{label} — {len(pages)} extracted pages")
    print(f"{'='*60}")

    # Classify all pages first (no LLM)
    classified = []
    for wp in pages:
        cls = classify_page(wp["text"])
        classified.append((wp["page"], wp["text"], cls))
        if cls != "skip":
            print(f"  Page {wp['page']:4d}: {cls}")

    # Phase 1: Extract prompts
    prompts = []
    print(f"\n  Phase 1: Extracting prompts...")
    for page_num, text, cls in classified:
        if cls == "prompt":
            result = extract_prompt(model, label, page_num, text)
            prompts.extend(result)

    # Phase 2: Extract answers
    answers = []
    print(f"\n  Phase 2: Extracting answers...")
    for page_num, text, cls in classified:
        if cls == "answer":
            result = extract_answer(model, label, page_num, text)
            answers.extend(result)

    print(f"\n  Result: {len(prompts)} prompts, {len(answers)} answers")
    return prompts, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--only", nargs="*", help="Process only these book slugs")
    args = parser.parse_args()

    slugs = args.only if args.only else list(BOOKS.keys())
    model = args.model
    print(f"Model: {model}")
    print(f"Books: {', '.join(slugs)}")

    # Load existing results for books we're NOT reprocessing
    existing = []
    if args.only and OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            existing = json.load(f)
        reprocess_labels = {BOOKS[s] for s in slugs if s in BOOKS}
        existing = [p for p in existing if not any(p.get("source", "").startswith(lbl) for lbl in reprocess_labels)]

    all_prompts = list(existing)
    all_answers = []

    for slug in slugs:
        if slug not in BOOKS:
            print(f"Unknown slug: {slug}")
            continue
        prompts, answers = process_book(model, slug)
        all_prompts.extend(prompts)
        all_answers.extend(answers)

    # Phase 3: Match answers to prompts
    print(f"\n{'='*60}")
    print("Phase 3: Matching answers to prompts...")
    match_answers(all_prompts, all_answers)

    # Summary
    total_samples = sum(len(p.get("samples", [])) for p in all_prompts)
    academic = sum(1 for p in all_prompts if p.get("test_type") == "academic")
    gt = sum(1 for p in all_prompts if p.get("test_type") == "gt")
    t1 = sum(1 for p in all_prompts if p.get("task_type") == 1)
    t2 = sum(1 for p in all_prompts if p.get("task_type") == 2)
    with_samples = sum(1 for p in all_prompts if p.get("samples"))

    print(f"\nTOTAL: {len(all_prompts)} prompts, {total_samples} samples")
    print(f"  Academic: {academic}, GT: {gt}")
    print(f"  Task 1: {t1}, Task 2: {t2}")
    print(f"  With samples: {with_samples}")

    print("\n--- Per-prompt ---")
    for i, p in enumerate(all_prompts):
        n = len(p.get("samples", []))
        s = f"{n} samples" if n else "no samples"
        print(f"  {i+1:2d}. [{p.get('test_type','?'):8s} T{p.get('task_type','?')}] {p.get('source','?'):35s} | {s:12s} | {p.get('topic','')[:45]}")

    # Deduplicate by (source, task_type)
    seen = set()
    deduped = []
    for p in all_prompts:
        key = (p.get("source"), p.get("task_type"))
        if key in seen:
            print(f"  DEDUP: dropping duplicate {key}")
            continue
        seen.add(key)
        deduped.append(p)

    if len(deduped) < len(all_prompts):
        print(f"\nDeduplicated: {len(all_prompts)} → {len(deduped)}")
        all_prompts = deduped

    OUT.write_text(json.dumps(all_prompts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
