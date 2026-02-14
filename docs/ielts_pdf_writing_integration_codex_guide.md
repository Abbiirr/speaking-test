# IELTS Prep Helper — PDF Ingestion + Writing Module Integration (Codex Guide)

This guide explains how to:
1) ingest IELTS PDFs once,  
2) index/search them efficiently,  
3) store prompts, assets, and “examiner calibration” data in SQLite, and  
4) add IELTS **Writing (Task 1/Task 2)** into your existing Speaking-focused system.

It assumes your current app architecture described in `project-overview.md` and `architecture.md`. fileciteturn1file0 fileciteturn1file1

---

## 0) Ground rules (exam-accurate evaluation)

### Official criteria (use these as your “source of truth”)
- IELTS Writing is scored on **Task Achievement/Task Response**, **Coherence & Cohesion**, **Lexical Resource**, **Grammatical Range & Accuracy**. Task 2 carries more weight than Task 1. citeturn1search1turn1search5turn1search3
- The official public **Writing band descriptors** PDF can be embedded into your prompts and stored locally for consistent scoring. citeturn1search46turn1search2

### “User-first” workflow (prevents memorisation)
1) user writes (or speaks) first  
2) system marks using band descriptors  
3) system gives targeted improvement actions  
4) only then show models / snippets for learning (optional)

---

## 1) PDF ingestion: parse once, store forever

### 1.1 Ingestion goals
From each PDF (e.g., Cambridge IELTS books):
- **page text** (for search + prompt extraction)
- **images** (Task 1 charts / diagrams)
- **document metadata** (hash, file name, page count)
- **extracted prompts** (Task 1/2 question text)
- optional: “calibration artifacts” (sample answers, examiner notes, etc.)

### 1.2 Recommended extractor: PyMuPDF (fast, handles images)
PyMuPDF can extract:
- page text blocks/layout
- images via `page.get_images()` or `page.get_text("dict")` image blocks citeturn2search5turn2search1turn2search0

Alternative: `pypdf` for lightweight text/image extraction; but be mindful of memory spikes on large streams. citeturn2search8turn2search6  
Alternative: `pdfminer.six` for detailed text layout; slower but flexible. citeturn2search3

### 1.3 Parsing once: content-hash gating
Compute a SHA256 hash of the PDF bytes; only parse if unseen.

**Rule:** `doc_hash` + `parser_version` controls re-ingestion.

---

## 2) SQLite schema for PDFs + search

Your current DB is speaking-centric (`sessions`, `attempts`, `questions`). fileciteturn1file1  
Add a **documents** subsystem + an FTS index.

### 2.1 Core tables

```sql
-- 1) Documents tracked by hash so parsing is done once
CREATE TABLE IF NOT EXISTS documents (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_hash      TEXT NOT NULL UNIQUE,
  file_name     TEXT NOT NULL,
  doc_type      TEXT NOT NULL,            -- "cambridge_book" | "official_rubric" | etc.
  page_count    INTEGER NOT NULL,
  parser        TEXT NOT NULL,            -- "pymupdf"
  parser_version TEXT NOT NULL,
  ingested_at   TEXT NOT NULL             -- ISO8601
);

-- 2) Per-page text extracted once
CREATE TABLE IF NOT EXISTS document_pages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id     INTEGER NOT NULL,
  page_no    INTEGER NOT NULL,            -- 0-based
  text       TEXT NOT NULL,
  FOREIGN KEY (doc_id) REFERENCES documents(id),
  UNIQUE (doc_id, page_no)
);

-- 3) Extracted assets (images) for Task 1 charts etc.
CREATE TABLE IF NOT EXISTS document_assets (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id      INTEGER NOT NULL,
  page_no     INTEGER NOT NULL,
  asset_type  TEXT NOT NULL,              -- "image"
  file_path   TEXT NOT NULL,              -- local path, e.g. data/assets/<hash>.png
  width       INTEGER,
  height      INTEGER,
  meta_json   TEXT DEFAULT '',
  FOREIGN KEY (doc_id) REFERENCES documents(id)
);
```

### 2.2 Full-text search index (FTS5)
Create an FTS5 virtual table for fast search over page text. SQLite FTS5 supports `snippet()` and `highlight()` for UI previews. citeturn1search0turn1search4

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS document_pages_fts
USING fts5(text, doc_id UNINDEXED, page_no UNINDEXED, content='document_pages', content_rowid='id');

-- keep in sync after inserts:
INSERT INTO document_pages_fts(rowid, text, doc_id, page_no)
VALUES (?, ?, ?, ?);
```

**Search query pattern:**
```sql
SELECT
  doc_id,
  page_no,
  snippet(document_pages_fts, -1, '[', ']', '…', 32) AS preview
FROM document_pages_fts
WHERE document_pages_fts MATCH ?
LIMIT 20;
```

---

## 3) Writing question bank tables (separate from speaking)

You currently seed Speaking questions from CSV into `questions`. fileciteturn1file0  
Writing needs different fields: task type, prompt text, optional chart image, and optional extracted chart data.

```sql
CREATE TABLE IF NOT EXISTS writing_prompts (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  test_type        TEXT NOT NULL,     -- "academic" | "gt"
  task_type        INTEGER NOT NULL,  -- 1 or 2
  topic            TEXT DEFAULT '',
  prompt_text      TEXT NOT NULL,
  source_doc_id    INTEGER,
  source_page_no   INTEGER,
  chart_asset_id   INTEGER,           -- for Task 1
  task1_data_json  TEXT DEFAULT '',   -- optional structured data extracted once
  created_at       TEXT NOT NULL,
  FOREIGN KEY (source_doc_id) REFERENCES documents(id),
  FOREIGN KEY (chart_asset_id) REFERENCES document_assets(id)
);

-- Optional: multiple band-level samples per prompt for calibration (private use)
CREATE TABLE IF NOT EXISTS writing_samples (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt_id        INTEGER NOT NULL,
  band            REAL NOT NULL,      -- e.g. 6.5, 7.0, 8.0, 9.0
  essay_text       TEXT NOT NULL,
  examiner_notes   TEXT DEFAULT '',
  source           TEXT DEFAULT '',   -- "cambridge", "human_marker", "user_approved"
  created_at       TEXT NOT NULL,
  FOREIGN KEY (prompt_id) REFERENCES writing_prompts(id)
);
```

---

## 4) Writing attempts: store user essays + marking output

Do NOT overload your existing `attempts` table with writing columns; it will become messy. Add `writing_attempts`.

```sql
CREATE TABLE IF NOT EXISTS writing_attempts (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id         INTEGER NOT NULL,
  timestamp          TEXT NOT NULL,
  prompt_id          INTEGER NOT NULL,
  task_type          INTEGER NOT NULL,   -- 1 or 2
  essay_text         TEXT NOT NULL,
  word_count         INTEGER NOT NULL,

  -- criterion scores (0-9)
  task_score         REAL NOT NULL,
  coherence_score    REAL NOT NULL,
  lexical_score      REAL NOT NULL,
  grammar_score      REAL NOT NULL,
  overall_band       REAL NOT NULL,

  -- structured feedback
  examiner_feedback  TEXT DEFAULT '',
  paragraph_feedback TEXT DEFAULT '',    -- JSON
  grammar_corrections TEXT DEFAULT '',   -- JSON list
  vocabulary_upgrades TEXT DEFAULT '',   -- JSON list
  improvement_tips   TEXT DEFAULT '',    -- JSON list

  provider           TEXT NOT NULL,      -- "gemini" | "ollama" | ...
  raw_json           TEXT DEFAULT '',

  FOREIGN KEY (session_id) REFERENCES sessions(id),
  FOREIGN KEY (prompt_id) REFERENCES writing_prompts(id)
);
```

---

## 5) Extracting writing prompts from PDFs

### 5.1 Strategy options
- **Manual tagging (fastest MVP):** search the PDF via FTS for “Writing Task 1/Task 2” headers, then manually select prompt text and (Task 1) the chart image.
- **Semi-automated extraction:** heuristics detect writing sections + task boundaries.

Given “minimal time MVP”, do manual tagging first, then automate later.

### 5.2 Ingestion script skeleton (`scripts/ingest_pdf.py`)
Core steps:
1) read PDF bytes
2) compute SHA256
3) if exists in `documents` with same parser_version → skip
4) else:
   - insert document
   - for each page:
     - extract text
     - insert into `document_pages` and `document_pages_fts`
     - extract images → store PNGs → insert into `document_assets`

PyMuPDF examples for text + images:
- `page.get_text()` for text
- `page.get_images()` or `page.get_text("dict")` for images citeturn2search5turn2search1turn2search0

---

## 6) “Search through PDFs” UX inside Streamlit

Add an internal admin page: **PDF Search**
- text input `query`
- results table: doc, page, preview (`snippet()`)
- click result → show full page text + extracted images on that page

Why: it turns the PDF into a searchable corpus and makes prompt extraction a UI workflow, not a coding workflow.

FTS5 `snippet()`/`highlight()` are designed for this exact usage. citeturn1search0turn1search4

---

## 7) Writing evaluation pipeline (AI examiner)

### 7.1 Evaluation criteria + rules (hard constraints)
Use official rules in the scorer:
- Task 1 minimum 150 words; Task 2 minimum 250 words. citeturn1search5turn1search3
- Penalize underlength in Task score + overall.

### 7.2 Pydantic schema for writing (mirror your speaking pattern)
You already use structured output for Speaking via Pydantic (`ContentEvaluation`, `EnhancedReview`). fileciteturn1file1  
Replicate for Writing:

- `WritingEvaluation` (criterion scores + feedback)
- `WritingEnhancedReview` (adds paragraph analysis, corrections, upgrades)

### 7.3 Prompt design: “examiner mode”
Hard requirement: the model must:
- score each criterion
- cite evidence from the essay (quote small phrases)
- list actionable improvements
- not “compare to band 9 answer”; compare to band descriptors

Use official writing band descriptors as an embedded rubric reference. citeturn1search46turn1search1turn1search2

### 7.4 “Hints from real-life evaluations”
Three practical ways (in descending reliability):

1) **Official band descriptors (highest reliability)**  
   Use the official PDFs as the rubric basis. citeturn1search46turn1search2turn1search1

2) **Human-marked practice products (ground truth)**
   - IELTS Progress Check: practice tests assessed by official markers, feedback derived from band descriptors. citeturn3search1  
   Your system can mimic the style: “criterion band + descriptor-like comments + next steps.”

3) **Commercial AI feedback patterns (style inspiration only)**
   - British Council IELTS Ready Premium mentions AI feedback on writing. citeturn3search6turn3search2  
   - Cambridge Write & Improve provides an IELTS band estimate and feedback workflow. citeturn3search3  
   Use these to shape UX (instant score + iterative rewrite), not as a scoring authority.

---

## 8) Task 1 charts: making evaluation reliable

### Problem
To evaluate Task 1 Task Achievement properly, the examiner checks if the candidate summarised key features accurately. Your system needs access to chart information.

### MVP options
**Option A (recommended): pre-extract Task 1 “chart summary” once**
1) extract the chart image from PDF
2) run a one-time “chart-to-structured-data” step:
   - Gemini Vision (or any vision-capable API) reads the chart
   - store `task1_data_json` + `key_features` in DB
3) During marking, provide `task1_data_json` to the examiner prompt.

This converts a multimodal problem into a text problem at runtime.

**Option B (lower accuracy): no chart data**
Mark only on structure and language, and give a disclaimer that Task Achievement is approximate.

---

## 9) Integrating writing into existing codebase

Your current flow is audio → Whisper → evaluator → database → review renderer. fileciteturn1file1  
Writing is simpler: text → evaluator → database → review renderer.

### 9.1 New modules (minimal)
- `writing_questions.py`  
  - `get_random_writing_prompt(test_type, task_type, topic=None)`
- `writing_models.py` (or extend `models.py`)  
  - `WritingEvaluation`, `WritingEnhancedReview`
- `writing_evaluator.py` (or extend provider files)  
  - `evaluate_writing()`, `evaluate_writing_enhanced()`
- `writing_review.py` (or extend `review.py`)  
  - `render_writing_review()`
- `database.py`
  - migrations for `writing_prompts`, `writing_attempts`, `documents`, `document_pages`, `document_assets`, `document_pages_fts`

### 9.2 Facade changes (`evaluator.py`)
Add two new functions mirroring Speaking:
- `evaluate_writing(prompt, essay_text, task_type, task1_data_json=None)`
- `evaluate_writing_enhanced(...)`

This keeps the provider switch (Gemini vs Ollama) consistent with your existing design. fileciteturn1file1

### 9.3 UI changes (`app.py`)
Add mode: **Writing**
- choose test type: Academic/GT
- choose task: Task 1 / Task 2
- show prompt text + chart image (if Task 1)
- `st.text_area` for essay
- word count + timer (20 min for Task 1, 40 min for Task 2) citeturn3search0turn1search5turn1search3
- submit → run evaluation → save → render

---

## 10) “Parse PDF once” operational checklist

### Filesystem layout
```
data/
  history.db
  assets/
    pdf_images/
      <doc_hash>/
        p0003_img01.png
scripts/
  ingest_pdf.py
  extract_writing_prompts.py   # optional
src/
  speaking_test/
  writing_test/
```

### Ingestion idempotency
- `doc_hash` unique
- `parser_version` stored
- re-ingest only when:
  - file changes OR
  - parser logic changes (parser_version bump)

---

## 11) Quality gates (so scoring feels examiner-like)

### Writing checks before LLM call
- enforce word count minimums (soft penalty)
- detect off-topic via simple keyword overlap with prompt
- detect “memorised/template” via similarity to known samples

### LLM output requirements
- strict JSON schema (Pydantic)
- criterion score 0–9
- evidence quotes for each criterion
- 3 “highest-impact” fixes
- 1 rewrite plan (what to change in next draft)

### Post-processing
- clamp bands to 0.0–9.0
- round to nearest 0.5
- compute overall: (Task + CC + LR + GRA)/4, then combine Task1/Task2 with Task2 weighted more. citeturn1search1turn1search5turn1search3

---

## 12) Implementation checklist for Codex (copy/paste)

### Phase 1 — PDF ingestion + search
- [ ] add tables: `documents`, `document_pages`, `document_assets`
- [ ] add FTS: `document_pages_fts`
- [ ] write `scripts/ingest_pdf.py` using PyMuPDF
- [ ] add Streamlit “PDF Search” admin page using FTS `snippet()`

### Phase 2 — Writing prompts
- [ ] add tables: `writing_prompts`
- [ ] build UI to create prompts from a searched page (copy selected text; attach chart image)
- [ ] seed a small set of prompts (10 Task 1 + 10 Task 2)

### Phase 3 — Writing evaluation
- [ ] add Pydantic models for Writing
- [ ] add provider prompts: `WRITING_SYSTEM_PROMPT`, `WRITING_ENHANCED_SYSTEM_PROMPT`
  - embed official criteria and band descriptor snippets citeturn1search46turn1search1turn1search2
- [ ] implement `evaluate_writing(_enhanced)` in `evaluator.py`

### Phase 4 — Writing attempts + history
- [ ] add table: `writing_attempts`
- [ ] implement `save_writing_attempt()`
- [ ] add History view aggregation for writing bands over time

### Phase 5 — Task 1 chart data extraction (optional but high value)
- [ ] one-time chart interpretation (Gemini Vision) → store `task1_data_json`
- [ ] use stored `task1_data_json` during marking

---

## 13) Legal note (important)
Cambridge books are copyrighted. For a private, local MVP, ingestion is typically fine. Do not redistribute extracted prompts/answers publicly without permission; for public release, rely on official free materials from IELTS websites or obtain licensing.

---

## Appendix A — Minimal PyMuPDF extraction snippet

```python
import hashlib
import pymupdf  # PyMuPDF

pdf_path = "Cambridge IELTS 19 Academic.pdf"
pdf_bytes = open(pdf_path, "rb").read()
doc_hash = hashlib.sha256(pdf_bytes).hexdigest()

doc = pymupdf.open(pdf_path)

for page_no in range(len(doc)):
    page = doc[page_no]
    text = page.get_text()  # fast plain text

    # images (simple)
    for idx, img in enumerate(page.get_images(), start=1):
        xref = img[0]
        pix = pymupdf.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        out = f"data/assets/pdf_images/{doc_hash}/p{page_no:04d}_img{idx:02d}.png"
        pix.save(out)
        pix = None
```

PyMuPDF image extraction patterns and the `get_text("dict")` approach are documented in their recipes. citeturn2search5turn2search1turn2search0

---

## Appendix B — FTS5 query examples

```sql
-- Find candidate pages and show match previews
SELECT doc_id, page_no,
       snippet(document_pages_fts, -1, '[', ']', '…', 32) AS preview
FROM document_pages_fts
WHERE document_pages_fts MATCH 'writing NEAR/5 task'
LIMIT 30;

-- Highlight full text
SELECT highlight(document_pages_fts, 0, '<b>', '</b>')
FROM document_pages_fts
WHERE document_pages_fts MATCH 'task 2'
LIMIT 5;
```

FTS5 `snippet()` and `highlight()` are part of core SQLite documentation. citeturn1search0turn1search4
