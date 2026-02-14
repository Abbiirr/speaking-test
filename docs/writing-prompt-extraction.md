# Writing Prompt Extraction Process

Pipeline for extracting IELTS writing prompts from Cambridge practice test PDFs into the database.

## Overview

```
PDF files ──┬── pymupdf (text-native) ──→ writing_pages.json
            └── Ollama VLM (scanned) ──→ writing_pages.json
                                              ↓
                                    build_seed_json.py (Ollama text LLM)
                                    [per-page chunking, ~1-3k chars each]
                                              ↓
                                    seed_writing_prompts.json
                                              ↓
                                    Claude Code supervises & reviews
                                              ↓
                                    seed_writing_prompts.py → database
```

## Models

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| **Vision/OCR** | `qwen3-vl:30b-a3b-instruct-q4_K_M` | ~12GB | Scanned PDF pages → text |
| **Text/JSON** | `dengcao/Qwen3-30B-A3B-Instruct-2507` | ~16GB | Extracted text → structured JSON |
| **Supervisor** | Claude Code | — | Reviews output, checks counts, approves seeding |

Both are MoE 30B models (only ~3B active params) — fast and fit 16GB VRAM.

Pull commands:
```bash
ollama pull qwen3-vl:30b-a3b-instruct-q4_K_M
ollama pull dengcao/Qwen3-30B-A3B-Instruct-2507
```

## Step 1: Extract text from PDFs

**Script:** `scripts/extract_text_writing.py`
**Run:** `.venv/Scripts/python.exe scripts/extract_text_writing.py`

### Hybrid extraction strategy

The script uses a two-tier approach:

1. **pymupdf first** — fast, accurate for text-native PDFs. Extracts text and renders page images.
2. **Ollama VLM fallback** — for scanned PDFs where pymupdf gets empty/garbled text. Sends page images to `qwen3-vl:30b-a3b-instruct` for OCR.

Detection logic: if pymupdf extracts <50 meaningful characters from a page, treat it as scanned and use VLM.

For each PDF, saves to `pdf_extracted/{slug}/`:
- `writing_pages.md` — human-readable markdown
- `writing_pages.json` — machine-readable JSON
- `images/page_NNNN.png` — page images (used by VLM and for Task 1 charts)

### PDF inventory

| Slug | Book | Type | Pages | Prompts | Answers |
|------|------|------|-------|---------|---------|
| cambridge-ielts-3 | Cambridge IELTS 3 | text-native | 179 | 12 (8 academic + 4 GT) | 12 |
| cambridge-ielts-18 | Cambridge IELTS 18 | text-native | 147 | 8 (8 academic) | 8 |
| ielts1 | IELTS 1 | text-native | 162 | 10 (8 academic + 2 GT) | varies |
| ielts2 | IELTS 2 | text-native | 80 | 12 (8 academic + 4 GT) | 12 |
| cambridge-ielts4-01 | Cambridge IELTS 4 (Test 1) | text-native | 29 | 2 (2 academic) | 0 |
| cambridge-ielts-10 | Cambridge IELTS 10 | scanned | ~170 | TBD | TBD |
| cambridge-ielts-11 | Cambridge IELTS 11 | scanned | ~170 | TBD | TBD |
| cambridge-ielts-12 | Cambridge IELTS 12 | scanned | ~170 | TBD | TBD |
| cambridge-4-ans-key | Cambridge IELTS 4 (answer key) | scanned | ~30 | 0 | TBD |

## Step 2: Ollama structures the extracted text

**Script:** `scripts/build_seed_json.py`
**Model:** `dengcao/Qwen3-30B-A3B-Instruct-2507`

```bash
.venv/Scripts/python.exe scripts/build_seed_json.py --model dengcao/Qwen3-30B-A3B-Instruct-2507
.venv/Scripts/python.exe scripts/build_seed_json.py --model dengcao/Qwen3-30B-A3B-Instruct-2507 --only ielts1
```

### Per-page chunking approach

Each Ollama call handles exactly ONE page (~1-3k chars). No long-text issues.

1. **Regex pre-classification** (no LLM) — each page tagged as `prompt`, `answer`, or `skip`
2. **Phase 1** — send each `prompt` page to Ollama → structured prompt JSON
3. **Phase 2** — send each `answer` page to Ollama → structured answer JSON
4. **Phase 3** — match answers to prompts by source + task_type, deduplicate

**Output:** `scripts/seed_writing_prompts.json`

Each entry:
```json
{
  "test_type": "academic|gt",
  "task_type": 1|2,
  "topic": "Short label",
  "prompt_text": "Full prompt text...",
  "source": "Cambridge IELTS 3, Test 1",
  "samples": [{
    "band": 5.0,
    "essay_text": "...",
    "examiner_notes": "...",
    "type": "sample|model"
  }]
}
```

## Step 3: Claude Code supervises

Claude Code acts as supervisor — does NOT read the full extracted text. Instead:

1. **Runs the scripts** and reads their summary output (counts, per-prompt table)
2. **Spot-checks** a few entries from seed JSON (short reads, not full file)
3. **Verifies counts** against expected totals
4. **Flags issues** — missing prompts, wrong test_type, unmatched answers
5. **Approves** or requests re-run of specific books

Quick verification:
```bash
.venv/Scripts/python.exe -c "
import json
data = json.load(open('scripts/seed_writing_prompts.json'))
print(f'Total: {len(data)}')
print(f'Academic: {sum(1 for p in data if p[\"test_type\"]==\"academic\")}')
print(f'GT: {sum(1 for p in data if p[\"test_type\"]==\"gt\")}')
print(f'Task 1: {sum(1 for p in data if p[\"task_type\"]==1)}')
print(f'Task 2: {sum(1 for p in data if p[\"task_type\"]==2)}')
print(f'With samples: {sum(1 for p in data if p.get(\"samples\"))}')
dupes = len(data) - len(set((p[\"source\"],p[\"task_type\"]) for p in data))
print(f'Duplicates: {dupes}')
"
```

## Step 4: Pre-seed database check

Before seeding, check if data already exists (may be partial/corrupted):

```bash
.venv/Scripts/python.exe scripts/verify_writing_prompts.py
```

If the database already has writing prompts:
- **Counts match expected** → skip seeding, data is good
- **Partial or wrong counts** → clear and re-seed:
  ```bash
  .venv/Scripts/python.exe -c "
  import sys; sys.path.insert(0, 'src')
  from speaking_test.database import get_db
  db = get_db()
  db.execute('DELETE FROM writing_samples')
  db.execute('DELETE FROM writing_prompts')
  db.commit()
  print('Cleared writing_prompts and writing_samples tables')
  "
  ```
- **Empty tables** → proceed to seeding

## Step 5: Seed the database

**Script:** `scripts/seed_writing_prompts.py`
**Run:** `.venv/Scripts/python.exe scripts/seed_writing_prompts.py`

Loads `seed_writing_prompts.json` into `writing_prompts` and `writing_samples` tables. Deduplicates by prompt_text + task_type.

## Step 6: Post-seed verification

**Script:** `scripts/verify_writing_prompts.py`
**Run:** `.venv/Scripts/python.exe scripts/verify_writing_prompts.py`

Checks: counts by type, no duplicates, no garbled characters, sample counts.

## Step 7: Smoke test in app

```bash
.venv/Scripts/python.exe -m streamlit run src/speaking_test/app.py
```

Verify Writing mode shows prompts for all 4 combinations: Academic Task 1, Academic Task 2, GT Task 1, GT Task 2.

## Adding new PDFs

1. Drop the PDF into `pdf/`
2. Add its entry to `PDF_FILES` in `scripts/extract_text_writing.py`
3. Run extraction: `.venv/Scripts/python.exe scripts/extract_text_writing.py`
4. Add its slug to `BOOKS` in `scripts/build_seed_json.py`
5. Run: `.venv/Scripts/python.exe scripts/build_seed_json.py --only <new-slug>`
6. Claude Code reviews the summary output
7. Seed + verify (Steps 4-7 above)
