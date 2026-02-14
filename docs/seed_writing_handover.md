# Handover: Seed IELTS Writing Prompts

## Status
- Scripts written, ready to run
- Seed JSON (`scripts/seed_writing_prompts.json`) NOT yet created — build it in next session
- PDF text extraction already done and verified — data inventory below

## Scripts created

| Script | Purpose | Run command |
|---|---|---|
| `scripts/extract_text_writing.py` | Dump writing pages from text-native PDFs | `uv run --extra pdf python scripts/extract_text_writing.py` |
| `scripts/seed_writing_prompts.py` | Load seed JSON into database | `uv run python scripts/seed_writing_prompts.py` |
| `scripts/ocr_extract_writing.py` | OCR scanned-image PDFs (Phase 2) | `uv run --extra pdf python scripts/ocr_extract_writing.py "IELTS 10"` |
| `scripts/verify_writing_prompts.py` | Verify database was seeded correctly | `uv run python scripts/verify_writing_prompts.py` |

## Next session: Step-by-step

### Phase 1: Build seed JSON from text-native PDFs

1. **Extract raw text** (already done once, but rerun if needed):
   ```bash
   uv run --extra pdf python scripts/extract_text_writing.py > /tmp/writing_pages.txt
   ```

2. **Build `scripts/seed_writing_prompts.json`** — use the inventory below. The JSON format:
   ```json
   [
     {
       "test_type": "academic",
       "task_type": 1,
       "topic": "Short descriptive label",
       "prompt_text": "Full prompt text including instructions...",
       "source": "Cambridge IELTS 3, Test 1",
       "samples": [
         {
           "band": 5.0,
           "essay_text": "Full essay text...",
           "examiner_notes": "Examiner comment...",
           "type": "sample"
         }
       ]
     }
   ]
   ```

3. **Run seed loader**:
   ```bash
   uv run python scripts/seed_writing_prompts.py
   ```

4. **Verify**:
   ```bash
   uv run python scripts/verify_writing_prompts.py
   ```

5. **Test in app**:
   ```bash
   uv run streamlit run src/speaking_test/app.py
   ```
   Navigate to Writing mode → select Academic/Task 2 → click "Get Prompt"

### Phase 2: OCR scanned PDFs (later)

1. Install Tesseract: `winget install UB-Mannheim.TesseractOCR`
2. Run one book at a time:
   ```bash
   uv run --extra pdf python scripts/ocr_extract_writing.py "Cambridge IELTS 10" > /tmp/ocr_ielts10.txt
   ```
3. Read output, extract prompts, append to seed JSON
4. Re-run seed loader (it deduplicates)

---

## Complete inventory of extracted prompts

### Cambridge IELTS 3 (12 prompts, all have answers)

| # | Test | Task | Type | Topic | Answer | Band |
|---|---|---|---|---|---|---|
| 1 | Test 1 | Task 1 | academic | Japanese tourists abroad 1985-1995 | Sample | 5 |
| 2 | Test 1 | Task 2 | academic | Football World Cup / sporting events / tensions | Sample | 4 |
| 3 | Test 2 | Task 1 | academic | Consumer goods spending in 4 European countries | Model | 9 |
| 4 | Test 2 | Task 2 | academic | Technology vs traditional skills and ways of life | Model | 9 |
| 5 | Test 3 | Task 1 | academic | Education/science participation developing vs industrialised | Model | 9 |
| 6 | Test 3 | Task 2 | academic | Children in paid work — wrong or valuable? | Model | 9 |
| 7 | Test 4 | Task 1 | academic | US and Japan unemployment rates 1993-1999 | Sample | 7 |
| 8 | Test 4 | Task 2 | academic | Health/education/trade — richer nations helping poorer | Sample | 6 |
| 9 | GT Test A | Task 1 | gt | Letter to new employer about delay starting job | Model | 9 |
| 10 | GT Test A | Task 2 | gt | Children's leisure must be educational — agree/disagree | Model | 9 |
| 11 | GT Test B | Task 1 | gt | Letter to newspaper about airport expansion | Sample | 7 |
| 12 | GT Test B | Task 2 | gt | Families not as close as before — reasons and solutions | Sample | 6 |

### Cambridge IELTS 18 (8 prompts, all have answers)

| # | Test | Task | Type | Topic | Answer | Band |
|---|---|---|---|---|---|---|
| 1 | Test 1 | Task 1 | academic | Population in Asian cities 1970-2040 | Sample | 6 |
| 2 | Test 1 | Task 2 | academic | Science should improve people's lives | Model | 9 |
| 3 | Test 2 | Task 1 | academic | US households by annual income 2007-2015 | Sample | 5.5 |
| 4 | Test 2 | Task 2 | academic | University students — study other subjects or focus | Model | 9 |
| 5 | Test 3 | Task 1 | academic | Library floor plan — 20 years ago vs now | Model | 9 |
| 6 | Test 3 | Task 2 | academic | Rural people moving to cities — positive or negative | Model | 9 |
| 7 | Test 4 | Task 1 | academic | Monthly price changes copper/nickel/zinc 2014 | Model | 9 |
| 8 | Test 4 | Task 2 | academic | Ageing population — advantages vs disadvantages | Model | 9 |

### IELTS 1 (12 prompts, all have answers)

| # | Test | Task | Type | Topic | Answer | Band |
|---|---|---|---|---|---|---|
| 1 | Test 1 | Task 1 | academic | Consumer durables in Britain 1972-1983 | Model | 9 |
| 2 | Test 1 | Task 2 | academic | Fatherhood emphasised as much as motherhood | Model | 9 |
| 3 | Test 2 | Task 1 | academic | Leisure time by gender and employment status | Model | 9 |
| 4 | Test 2 | Task 2 | academic | Prevention vs cure — health budget allocation | Model | 9 |
| 5 | Test 3 | Task 1 | academic | UK residents visits abroad 1994-98 purpose & destination | Sample | 7 |
| 6 | Test 3 | Task 2 | academic | Capital punishment — essential to control violence? | Sample | 8 |
| 7 | Test 4 | Task 1 | academic | Imprisonment figures in 5 countries 1930-1980 | Model | 9 |
| 8 | Test 4 | Task 2 | academic | Women working — cause of juvenile delinquency? | Model | 9 |
| 9 | GT Test A | Task 1 | gt | Letter to library about overdue books | Model | 9 |
| 10 | GT Test A | Task 2 | gt | Government controlling number of children via taxes | Model | 9 |
| 11 | GT Test B | Task 1 | gt | Letter to airline about lost suitcase | Sample | 7 |
| 12 | GT Test B | Task 2 | gt | Why study in English / why English important | Sample | 6 |

### IELTS 2 / Cambridge IELTS 2 (10 prompts, 4 have answers)

| # | Test | Task | Type | Topic | Answer | Band |
|---|---|---|---|---|---|---|
| 1 | Test 1 | Task 1 | academic | Adult education survey — reasons & cost sharing | None | — |
| 2 | Test 1 | Task 2 | academic | Why do we need music? Traditional vs international | None | — |
| 3 | Test 2 | Task 1 | academic | Australian Bureau of Meteorology weather forecasts | None | — |
| 4 | Test 2 | Task 2 | academic | Wealthy nations sharing wealth with poorer nations | None | — |
| 5 | Test 3 | Task 1 | academic | Fast foods spending & consumption trends in Britain | Model | 9 |
| 6 | Test 3 | Task 2 | academic | News editors — what influences decisions, bad news | Model | 9 |
| 7 | Test 4 | Task 1 | academic | Chorleywood village development map | None | — |
| 8 | Test 4 | Task 2 | academic | Single career old fashioned — multiple careers/lifelong education | None | — |
| 9 | GT | Task 1 | gt | Letter to bank about incorrect overdrawn charge | Model | 9 |
| 10 | GT | Task 2 | gt | Dependence on computers — good thing or suspicious? | Model | 9 |

### Cambridge IELTS 4 Test 1 (2 prompts, no answers)

| # | Test | Task | Type | Topic | Answer | Band |
|---|---|---|---|---|---|---|
| 1 | Test 1 | Task 1 | academic | Families living in poverty in Australia 1999 | None | — |
| 2 | Test 1 | Task 2 | academic | Media for communicating — comics/books/radio/TV/film/theatre | None | — |

---

## Totals

- **44 prompts** total across 5 PDFs
- **28 with answers** (model or sample)
- Academic: 34 prompts, GT: 10 prompts
- Task 1: 22 prompts, Task 2: 22 prompts

## Key extraction notes

- **PDF text is clean** — minimal garbled characters in these text-native PDFs
- **Replace** `'` with `'` if found (PDF encoding artifact)
- **Task 1 prompts reference charts/graphs** that exist only as images in the PDF. The prompt text still makes sense without seeing the chart, but the chart data isn't available as text. This is fine — users will practice writing based on the prompt description.
- **For model answers** use `band: 9.0` and `type: "model"`
- **For sample answers** the band score and examiner comment are stated in the text
- **Examiner notes**: For model answers, use "This model has been prepared by an examiner as an example of a very good answer." For sample answers, use the actual examiner comment text.

## Database tables (reference)

```sql
-- Prompts
INSERT INTO writing_prompts (test_type, task_type, topic, prompt_text, created_at)
VALUES ('academic', 2, 'Topic label', 'Full prompt text...', '2026-02-14T...');

-- Samples
INSERT INTO writing_samples (prompt_id, band, essay_text, examiner_notes, source, created_at)
VALUES (1, 5.0, 'Essay text...', 'Examiner comment...', 'Cambridge IELTS 3, Test 1', '2026-02-14T...');
```

## How writing_questions.py loads prompts

`load_writing_prompts(test_type, task_type)` queries `writing_prompts` table and returns `WritingPrompt` objects. The app calls `get_random_writing_prompt()` to pick one. If the table is empty, the app shows "No writing prompts in the database yet."
