# IELTS Practice

A local IELTS speaking and writing preparation assistant with full mock tests, AI-powered deep reviews, 550+ questions, PDF ingestion, and progress tracking.

Built with Streamlit, Whisper (faster-whisper), and Google Gemini or Ollama. Runs on GPU (NVIDIA CUDA).

## Features

| Mode | Description |
|------|-------------|
| **Interview** | Random questions from a 550+ question bank across all 3 parts. Deep Review mode gives grammar corrections, vocabulary upgrades, strengths, and priorities. |
| **Mock Test** | Full Part 1 → 2 → 3 simulation with realistic timing (1-min prep for Part 2). Per-question enhanced reviews and overall results. |
| **Practice** | Paste a reference script, read it aloud, get WER + fluency + pronunciation feedback. |
| **Transcribe** | Record speech and get a transcript with pitch visualization. |
| **Writing** | IELTS Writing Task 1 & 2 practice with AI evaluation on all 4 official criteria. Supports prompts from Cambridge IELTS PDFs or custom prompts. |
| **PDF Library** | Full-text search over ingested IELTS PDFs with page image preview. |
| **History** | Band score trends, per-criterion breakdown charts, weak area detection, session history, and writing analytics. |

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (tested on RTX 4060 Ti)
- A microphone
- Gemini API key **or** a running Ollama instance (for Interview and Mock Test modes)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already, then:

```bash
uv sync
```

To also install PDF ingestion support (optional — only needed if you want to extract prompts from Cambridge IELTS PDFs):

```bash
uv sync --extra pdf
```

Create a `.env` file. Choose one AI provider:

**Option A — Gemini (recommended):**

```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash-lite   # optional, this is the default
```

**Option B — Ollama (local, no API key needed):**

```
PROVIDER=ollama
OLLAMA_MODEL=deepseek-r1:8b          # or any model you have pulled
OLLAMA_BASE_URL=http://localhost:11434  # optional, this is the default
```

Make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull deepseek-r1:8b`).

## Usage

```bash
uv run streamlit run src/speaking_test/app.py
```

Opens at `http://localhost:8501`.

On first run, the Whisper `base.en` model (~150 MB) is downloaded automatically. After that, transcription runs offline. Gemini evaluation requires an internet connection; Ollama runs fully locally.

### Interview mode

1. Select **Interview** in the sidebar.
2. Filter by part (optional) and toggle **Deep Review** on/off.
3. Click **New Question** to get a random question from 550+ options.
4. Record your answer and click **Evaluate**.
5. Get band scores, examiner feedback, grammar corrections, vocabulary upgrades, and actionable improvement tips.

### Mock Test mode

1. Select **Mock Test** and click **Start Mock Test**.
2. Answer Part 1 questions (8-10 questions).
3. Part 2: read the cue card, use the 1-minute prep timer, then record your 2-minute answer.
4. Answer Part 3 discussion questions (4-5 questions, linked to the Part 2 topic).
5. View overall results with per-question expandable reviews.

### History mode

- **Band Trend** — line chart of your overall band score over time.
- **Criterion Breakdown** — four lines (Fluency, Lexical, Grammar, Pronunciation) showing per-criterion trends + weak area callout.
- **Sessions** — expandable list of past sessions with full attempt details.

### Practice mode

1. Paste a reference script.
2. Record yourself reading it aloud.
3. Get WER, fluency, speech rate, pronunciation, and band score feedback.

### Transcribe mode

1. Record your speech.
2. Get a text transcript and pitch contour chart.

### Writing mode

1. Select **Writing** in the sidebar.
2. Choose test type (Academic / General Training) and task (Task 1 / Task 2).
3. Click **Get Prompt** for a question from the database, or paste a custom prompt.
4. Write your essay in the text area (live word count shown).
5. Click **Submit Essay** for AI evaluation on all 4 IELTS Writing criteria.

### PDF Library mode

1. Search ingested documents with full-text search.
2. Browse results with page text snippets and full-page image previews.

### PDF Ingestion (one-time setup)

To populate the writing question bank from Cambridge IELTS PDFs:

```bash
# Step 1: Ingest PDFs (extracts text + page images)
uv run --extra pdf python scripts/ingest_pdf.py pdf/

# Step 2: Extract writing prompts from ingested pages
uv run python scripts/extract_writing_prompts.py
```

Extracted content is saved to `pdf_extracted/` as markdown, JSON, and PNG images.

## How Scoring Works

Interview and Mock Test modes blend Gemini content evaluation with audio analysis:

- **Fluency & Coherence (25%)** — 50% audio fluency (speech rate + pause ratio) + 50% Gemini coherence score
- **Lexical Resource (25%)** — 100% Gemini
- **Grammatical Range (25%)** — 100% Gemini
- **Pronunciation (25%)** — Whisper per-word recognition confidence

Overall band is the average, rounded to nearest 0.5, clamped to 4.0–9.0.

### Deep Review (Enhanced — Speaking)

When Deep Review is enabled, the AI examiner also returns:
- Specific grammar errors with corrections and rule explanations
- Vocabulary upgrades (basic word → advanced alternatives + example)
- 2-3 genuine strengths
- 2-3 specific, actionable improvement priorities

### Writing Scoring

Writing evaluation uses the official 4 IELTS Writing criteria (each 25%):

- **Task Achievement** — Does the essay address the prompt? (Task 1: data description; Task 2: argument development)
- **Coherence & Cohesion** — Paragraphing, linking, logical flow
- **Lexical Resource** — Vocabulary range and precision
- **Grammatical Range & Accuracy** — Sentence variety and error-free usage

Word count penalty: essays below the minimum (150 for Task 1, 250 for Task 2) are capped at Band 5 for Task Achievement. The enhanced review also provides paragraph-by-paragraph analysis, grammar corrections, and vocabulary upgrades.

## Question Bank

553 unique questions sourced from 3 files:
- `docs/ielts_questions.md` — original test questions with band-9 answers
- `docs/ielts_speaking_question_bank.md` — 40 Part 1 topics, 65 cue cards, 15 Part 3 themes
- `docs/ielts_speaking_master_pack_v2.md` — 26 Part 1 topics, 52 cue cards, 13 Part 3 themes + band-9 samples

A full export of all questions with their matched band-9 answers is available at [`questions_answers.csv`](questions_answers.csv) in the project root (109 questions have band-9 reference answers).

Mock tests automatically link Part 2 cue cards to related Part 3 discussion themes.

## Project Structure

```
src/speaking_test/
    app.py                Streamlit UI — sidebar nav, all 7 modes
    models.py             Shared data models (Speaking + Writing)
    questions.py          Speaking question loading + mock test assembly
    writing_questions.py  Writing prompt loading from DB
    evaluator.py          Provider-agnostic facade (Gemini or Ollama)
    gemini_evaluator.py   Gemini API — speaking + writing evaluation
    ollama_evaluator.py   Ollama API — with response normalization
    scorer.py             Audio analysis — speech rate, pauses, pronunciation
    review.py             Progressive disclosure review renderer (speaking + writing)
    database.py           SQLite persistence — sessions, attempts, writing, documents
scripts/
    ingest_pdf.py             Standalone PDF ingestion (text + page images)
    extract_writing_prompts.py  Writing prompt extraction from ingested pages
pdf_extracted/                Auto-generated: markdown, JSON, page images
data/
    history.db            SQLite database (auto-created, gitignored)
```

## Tips

- Use a quiet environment for best transcription accuracy.
- Aim for 120–160 WPM with minimal pauses for the best fluency score.
- Check the History tab regularly to track progress and identify weak areas.
- Use Deep Review mode to get specific grammar and vocabulary feedback.
