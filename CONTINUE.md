# Continue: Writing Prompt Extraction & Seeding

This file contains everything a new Claude Code session needs to pick up the writing prompt extraction pipeline. Read this fully before doing anything.

## Project Overview

This is an IELTS practice test app (Streamlit). It already has **speaking tests working** (1980 questions in DB). We are adding **writing test support** by extracting prompts and sample answers from Cambridge IELTS practice test PDFs, then seeding them into the database.

## Current State (2026-02-14)

| Step | Status | Notes |
|------|--------|-------|
| 1. Extract text from PDFs | DONE | 5 text-native PDFs extracted via pymupdf |
| 2. Structure text with Ollama | NOT STARTED | `build_seed_json.py` exists but hasn't been run with the correct model yet |
| 3. Claude Code reviews output | TODO | Check counts, spot-check entries |
| 4. Seed database | TODO | `seed_writing_prompts.py` ready to use |
| 5. Verify database | TODO | `verify_writing_prompts.py` ready to use |
| 6. Smoke test app | TODO | Run Streamlit, check Writing mode |

### What exists already
- `pdf_extracted/*/writing_pages.json` — extracted text from 5 PDFs (Step 1 output)
- `scripts/build_seed_json.py` — per-page chunked Ollama extraction script (ready to run)
- `scripts/seed_writing_prompts.json` — **STALE** (42 prompts from old 8B model, has known issues — must be regenerated)
- `pdf_extracted/*/ollama_response.txt` — old Ollama responses from previous attempts (can be ignored)

### Database state
- `writing_prompts` table: **0 rows** (empty)
- `writing_samples` table: **0 rows** (empty)
- Speaking data: **1980 questions** (intact, do NOT touch)
- DB path: `data/history.db`

---

## Step-by-Step Instructions

### Step 2: Run build_seed_json.py with Ollama

#### Prerequisites
The Ollama model `dengcao/Qwen3-30B-A3B-Instruct-2507` must be pulled and available. Check:
```bash
ollama list | grep Qwen3-30B
```

If NOT available, pull it (this is a ~16GB MoE model, fits 16GB VRAM):
```bash
ollama pull dengcao/Qwen3-30B-A3B-Instruct-2507
```

If pull fails or is too slow, fallback options:
```bash
# Option A: Already-installed 8B model (lower quality but works)
.venv/Scripts/python.exe scripts/build_seed_json.py --model qwen3:8b

# Option B: Any other Ollama model the user has
ollama list   # see what's available
```

#### Test run (smallest book first)
```bash
.venv/Scripts/python.exe scripts/build_seed_json.py --only cambridge-ielts4-01
```
Expected: 2 prompts (academic Task 1 + Task 2), 0 samples (this book has no answer key).

If this works, proceed to full run.

#### Full run (all 5 books)
```bash
.venv/Scripts/python.exe scripts/build_seed_json.py
```

This sends each page individually to Ollama. Expect ~60-90 pages total, each taking 5-30 seconds. Total time: 15-45 minutes depending on model speed.

Output: overwrites `scripts/seed_writing_prompts.json`

#### How build_seed_json.py works
1. **Regex pre-classification** (no LLM) — scans each page for "WRITING TASK 1/2" or "MODEL ANSWER" patterns, tags as `prompt`, `answer`, or `skip`
2. **Phase 1** — sends each `prompt` page to Ollama one-at-a-time, gets back structured JSON (test_type, task_type, topic, prompt_text)
3. **Phase 2** — sends each `answer` page to Ollama one-at-a-time, gets back structured JSON (band, essay_text, examiner_notes)
4. **Phase 3** — matches answers to prompts by (source, task_type), deduplicates by (source, task_type) key

#### Selective re-run
If a specific book's output looks wrong, re-run just that book:
```bash
.venv/Scripts/python.exe scripts/build_seed_json.py --only ielts1
```
This preserves results from other books and only regenerates the specified slug(s).

#### CLI flags
- `--model MODEL_NAME` — override the default Ollama model
- `--only slug1 slug2` — process only these books (preserves others in output JSON)

---

### Step 3: Claude Code reviews output

**Do NOT read the full seed JSON file** (it's large and wastes tokens). Instead:

#### 3a. Check summary counts
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
dupes = len(data) - len(set((p['source'],p['task_type']) for p in data))
print(f'Duplicates: {dupes}')
for p in data:
    n = len(p.get('samples', []))
    s = f'{n} samples' if n else 'NO SAMPLES'
    print(f'  [{p.get(\"test_type\",\"?\"):8s} T{p.get(\"task_type\",\"?\")}] {p.get(\"source\",\"?\"):35s} | {s:12s} | {p.get(\"topic\",\"\")[:45]}')
"
```

#### 3b. Compare against expected counts

| Book | Slug | Expected Prompts | Expected Answers |
|------|------|------------------|------------------|
| Cambridge IELTS 3 | cambridge-ielts-3 | 12 (8 academic + 4 GT) | 12 |
| Cambridge IELTS 18 | cambridge-ielts-18 | 8 (8 academic, no GT) | 8 |
| IELTS 1 | ielts1 | 10 (8 academic + 2 GT) | varies (~8-10) |
| IELTS 2 | ielts2 | 12 (8 academic + 4 GT) | 12 |
| Cambridge IELTS 4 | cambridge-ielts4-01 | 2 (2 academic, Test 1 only) | 0 |
| **TOTAL** | | **44** (34 academic + 10 GT) | **~40** |

Breakdown: 22 Task 1 + 22 Task 2

#### 3c. What to flag
- **Total != 44** — missing or extra prompts
- **Duplicates > 0** — deduplication should have caught these
- **Prompts with "NO SAMPLES" that should have samples** — answer matching failed
- **Wrong test_type** — academic prompts labeled as GT or vice versa (common with IELTS 1 book)
- **Topic says "skip" or is empty** — Ollama returned skip for a real prompt page

#### 3d. Spot-check (optional, only if counts look off)
```bash
.venv/Scripts/python.exe -c "
import json
data = json.load(open('scripts/seed_writing_prompts.json'))
# Check a specific entry by index
i = 0  # change this
p = data[i]
print(f'Source: {p[\"source\"]}')
print(f'Type: {p[\"test_type\"]} Task {p[\"task_type\"]}')
print(f'Topic: {p[\"topic\"]}')
print(f'Prompt (first 200 chars): {p[\"prompt_text\"][:200]}')
print(f'Samples: {len(p.get(\"samples\", []))}')
for j, s in enumerate(p.get('samples', [])):
    print(f'  Sample {j+1}: band={s[\"band\"]}, type={s[\"type\"]}, length={len(s[\"essay_text\"])} chars')
"
```

---

### Step 4: Pre-seed database check

```bash
.venv/Scripts/python.exe scripts/verify_writing_prompts.py
```

**If writing_prompts is empty (0 rows):** proceed to Step 5.

**If partial/wrong data exists:** clear and re-seed:
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

**WARNING:** Do NOT drop tables or touch any other tables. Speaking data (1980 questions) must remain intact.

---

### Step 5: Seed database

```bash
.venv/Scripts/python.exe scripts/seed_writing_prompts.py
```

Expected output: `Done: 44 prompts inserted, 0 skipped (duplicates), ~40 samples inserted`

The script deduplicates by (prompt_text, task_type), so running it twice is safe.

---

### Step 6: Post-seed verification

```bash
.venv/Scripts/python.exe scripts/verify_writing_prompts.py
```

Check that:
- Total prompts = 44
- Academic = 34, GT = 10
- Task 1 = 22, Task 2 = 22
- No duplicates
- No garbled characters
- ~40 samples total
- Prompts with samples >= 40 (all except Cambridge IELTS 4's 2 prompts)

---

### Step 7: Smoke test app

```bash
.venv/Scripts/python.exe -m streamlit run src/speaking_test/app.py
```

Verify Writing mode shows prompts for all 4 combinations:
- Academic Task 1 (describe data/chart/diagram)
- Academic Task 2 (essay)
- GT Task 1 (letter writing)
- GT Task 2 (essay)

---

## Architecture Reference

### Pipeline diagram
```
PDF files (in pdf/)
    ↓
extract_text_writing.py (pymupdf for text-native, VLM for scanned)
    ↓
pdf_extracted/{slug}/writing_pages.json  (raw text per page)
    ↓
build_seed_json.py (Ollama, per-page chunking)
    ↓
scripts/seed_writing_prompts.json  (structured JSON)
    ↓
Claude Code reviews counts + spot-checks
    ↓
seed_writing_prompts.py → database (writing_prompts + writing_samples tables)
```

### Database schema (relevant tables)

```sql
CREATE TABLE writing_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_type TEXT NOT NULL,          -- "academic" or "gt"
    task_type INTEGER NOT NULL,       -- 1 or 2
    topic TEXT DEFAULT '',            -- short 5-10 word label
    prompt_text TEXT NOT NULL,        -- full prompt text
    source_doc_id INTEGER,            -- FK to documents (unused for now)
    source_page_no INTEGER,           -- (unused for now)
    chart_asset_id INTEGER,           -- FK to document_assets (unused for now)
    task1_data_json TEXT DEFAULT '',   -- (unused for now)
    created_at TEXT NOT NULL
);

CREATE TABLE writing_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id INTEGER NOT NULL,       -- FK to writing_prompts
    band REAL NOT NULL,               -- band score (e.g. 5.0, 7.0, 8.0)
    essay_text TEXT NOT NULL,         -- the full essay
    examiner_notes TEXT DEFAULT '',   -- examiner's commentary
    source TEXT DEFAULT '',           -- e.g. "Cambridge IELTS 3, Test 1"
    created_at TEXT NOT NULL
);
```

### seed_writing_prompts.json format
```json
[
  {
    "test_type": "academic",
    "task_type": 1,
    "topic": "Short label",
    "prompt_text": "Full prompt text...",
    "source": "Cambridge IELTS 3, Test 1",
    "samples": [
      {
        "band": 8.0,
        "essay_text": "The full essay text...",
        "examiner_notes": "Examiner's comment...",
        "type": "model"
      }
    ]
  }
]
```

### Key files
| File | Purpose |
|------|---------|
| `scripts/build_seed_json.py` | Per-page chunked Ollama extraction (Step 2) |
| `scripts/seed_writing_prompts.json` | Intermediate output — structured prompts+samples |
| `scripts/seed_writing_prompts.py` | Loads JSON into database (Step 5) |
| `scripts/verify_writing_prompts.py` | Checks database counts and quality (Steps 4, 6) |
| `scripts/extract_text_writing.py` | PDF → text extraction (Step 1, already done) |
| `pdf_extracted/*/writing_pages.json` | Raw extracted text per page (Step 1 output) |
| `docs/writing-prompt-extraction.md` | Full process documentation |
| `src/speaking_test/database.py` | DB connection, schema, get_db() |
| `src/speaking_test/app.py` | Streamlit app, render_writing_mode() at ~line 897 |
| `data/history.db` | SQLite database file |
| `.env` | App config: PROVIDER=ollama, OLLAMA_MODEL=sam860/deepseek-r1-0528-qwen3:8b |

### Ollama models
| Role | Model | Size | Status |
|------|-------|------|--------|
| **Text/JSON extraction** | `dengcao/Qwen3-30B-A3B-Instruct-2507` | ~16GB | Needs `ollama pull` |
| **App evaluator** | `sam860/deepseek-r1-0528-qwen3:8b` | ~5GB | Already installed, used by app |
| **Vision/OCR (future)** | `qwen3-vl:30b-a3b-instruct-q4_K_M` | ~12GB | For scanned PDFs later |

### Book slugs → labels mapping
```python
BOOKS = {
    "cambridge-ielts-3":    "Cambridge IELTS 3",
    "cambridge-ielts-18":   "Cambridge IELTS 18",
    "ielts1":               "IELTS 1",
    "ielts2":               "IELTS 2",
    "cambridge-ielts4-01":  "Cambridge IELTS 4",
}
```

---

## Known Issues from Previous Attempts

These were problems with the old 8B model (`sam860/deepseek-r1-0528-qwen3:8b`). The new 30B model may or may not have them:

1. **IELTS 1 Practice Test 4 misclassified as GT** — the 8B model confused academic with GT for this test
2. **IELTS 1 only produced 8/10 prompts** — missing Practice Test 4 Task 1 and GT Task 2
3. **Duplicate topics across books** — e.g. "Computers Dependency" appeared in multiple entries with same prompt_text for different tests
4. **Long Ollama responses produced malformed JSON** — this was solved by the per-page chunking approach
5. **Answer pages not detected** — regex `ANSWER_RE` in build_seed_json.py was expanded to catch "Sample Writing answers" and other variants

If any of these recur with the new model, re-run the specific book with `--only slug` after fixing.

---

## Future Work (not blocking)

- **Scanned PDFs**: Cambridge IELTS 10, 11, 12 are scanned. Need `qwen3-vl:30b-a3b-instruct-q4_K_M` for VLM-based OCR. This is a separate step to tackle later.
- **Chart images for Task 1**: Academic Task 1 prompts reference charts/diagrams. The images are saved in `pdf_extracted/*/images/` but not yet linked to DB entries via `chart_asset_id`.
- **Cambridge IELTS 4 answer key**: Separate scanned PDF exists (`cambridge-4-ans-key`), could provide the 2 missing samples.
