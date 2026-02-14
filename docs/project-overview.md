# Project Overview

## 1. Introduction

IELTS Speaking Practice is a local, offline-capable application that helps candidates prepare for the IELTS Speaking exam. It combines real-time speech transcription (via Whisper), audio delivery analysis (speech rate, pauses, pronunciation confidence), and AI-powered content evaluation (via Gemini or Ollama) to provide band scores and detailed examiner-style feedback on spoken answers.

The core value proposition is **immediate, private, detailed feedback** — candidates can practice unlimited questions with AI evaluation that scores against official IELTS band descriptors, without needing a human examiner or sending audio to a third-party service (transcription runs locally on GPU).

## 2. Features and Modes

| Mode | Input | Output | Use Case |
|------|-------|--------|----------|
| **Interview** | Random IELTS question + recorded audio | 4-criterion band score, examiner feedback, grammar corrections, vocabulary upgrades, pronunciation warnings, strengths, improvement priorities | Practice individual questions with full AI evaluation. Supports deep review toggle and part filtering. |
| **Mock Test** | Full 3-part IELTS simulation (8–10 Part 1 + 1 Part 2 with prep timer + 4–5 Part 3) | Per-question reviews + overall averages across all criteria | Simulate a complete IELTS Speaking exam. Progress bar tracks advancement through parts. |
| **Practice** | Reference script (pasted text) + recorded audio of reading it aloud | WER score, estimated band, speech rate, pause ratio, pronunciation confidence, pitch chart | Read-aloud practice for pronunciation and fluency. No AI evaluation — scores based on word accuracy and audio metrics. |
| **Transcribe** | Recorded audio | Text transcription + pitch contour chart | Pure speech-to-text tool for checking pronunciation and intonation patterns. |
| **History** | *(reads from database)* | Band trend chart, per-criterion breakdown, session drill-down with full reviews, weakness analysis (recurring grammar errors, basic words, improvement tips, criterion trends) | Track progress over time and identify persistent weak areas. |

## 3. Tech Stack

| Category | Dependencies | Purpose |
|----------|-------------|---------|
| **Framework** | `streamlit >= 1.42.0` | Web UI with audio input, metrics, charts, expanders, session state |
| **Speech Processing** | `faster-whisper >= 1.1.0`, `librosa >= 0.10.2`, `soundfile >= 0.12.1`, `numpy >= 1.26.0` | Local GPU transcription (CUDA float16), audio analysis (pitch, VAD, speech rate), WAV file handling |
| **Text Comparison** | `jiwer >= 3.1.0` | Word Error Rate computation for Practice mode |
| **AI Evaluation** | `google-genai >= 1.0.0`, `httpx >= 0.27.0`, `pydantic >= 2.0.0` | Gemini structured output, Ollama HTTP API, schema validation for AI responses |
| **Configuration** | `python-dotenv >= 1.0.0` | `.env` file loading for API keys and settings |
| **Data Storage** | `sqlite3` (stdlib), `csv` (stdlib) | Persistent history database, question bank CSV seeding |
| **Runtime** | Python >= 3.11, CUDA-capable GPU (RTX 4060 Ti recommended) | Whisper runs on GPU with float16; CPU fallback possible but slower |

## 4. How to Run

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
#    Create a .env file in the project root:
#    GEMINI_API_KEY=your-key-here       (required for Gemini provider)
#    PROVIDER=gemini                     (or "ollama")
#    GEMINI_MODEL=gemini-2.5-flash-lite  (optional, default shown)
#    OLLAMA_BASE_URL=http://localhost:11434  (optional, default shown)
#    OLLAMA_MODEL=deepseek-r1:8b         (optional, default shown)

# 3. Run the app
uv run streamlit run src/speaking_test/app.py
```

On first run, the application seeds the SQLite database (`data/history.db`) from `questions_answers_updated.csv`. This happens once automatically.

## 5. Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(none)* | Google Gemini API key. Required when `PROVIDER=gemini`. |
| `PROVIDER` | `"gemini"` | AI evaluation provider: `"gemini"` or `"ollama"`. |
| `GEMINI_MODEL` | `"gemini-2.5-flash-lite"` | Gemini model ID for evaluation calls. |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Ollama server URL. |
| `OLLAMA_MODEL` | `"deepseek-r1:8b"` | Ollama model name for evaluation calls. |

### File-Based Configuration

| File | Purpose |
|------|---------|
| `.env` | Environment variables (loaded via `python-dotenv`) |
| `questions_answers_updated.csv` | Question bank with band 9 answers and variants |
| `data/history.db` | SQLite database (auto-created on first run) |
| `logs/app.log` | Application log file (DEBUG level, UTF-8) |

## 6. Question Bank

The question bank contains **691 unique questions** across **1,980 rows** (multiple answer variants per question).

### Breakdown by Part

| Part | Unique Questions | Total Rows | Description |
|------|----------------:|----------:|-------------|
| Part 1 | ~371 | 1,113 | Introduction & Interview — short answers |
| Part 2 | ~62 | 186 | Long Turn — cue card with bullet points |
| Part 3 | ~258 | 681 | Discussion — deeper analytical questions |

### Sources

| Source | Rows | Description |
|--------|-----:|-------------|
| `question_bank` | 835 | Curated IELTS question bank |
| `master_pack` | 682 | IELTS Speaking master pack v2 |
| `web_research_2026-02-13` | 399 | Web-researched questions |
| `ielts_questions` | 64 | Core IELTS question set |

### CSV Format

Columns: `part`, `topic`, `question`, `cue_card`, `source`, `band9_answer`, `answer_variant`

- Each question has 2–3 answer variants (A, B, C) with slightly different wording to prevent memorization
- Part 2 questions include `cue_card` text with bullet points for the candidate's preparation
- `band9_answer` provides a model answer used as reference scope (not for direct comparison in scoring)

## 7. Current Limitations

### Speaking-Specific Gaps

- **No real-time feedback** — evaluation happens after recording, not during
- **Whisper accuracy** — transcription errors can affect AI evaluation quality; no mechanism to correct transcripts before evaluation
- **Pronunciation scoring is indirect** — based on Whisper word-level confidence probabilities, not phoneme-level analysis
- **No speaker verification** — anyone can use the app; no enrollment or voice profile

### Architecture Limitations

- **Single-file UI** — all 789 lines of UI logic in `app.py` will grow unwieldy as modes are added
- **Speaking-only evaluation prompts** — system prompts in `gemini_evaluator.py` and `ollama_evaluator.py` are hardcoded for speaking evaluation; writing requires separate prompts
- **No test-type abstraction** — the data flow assumes audio input → transcription → evaluation; writing would skip transcription entirely
- **Database schema is speaking-centric** — columns like `speech_rate`, `pause_ratio`, `pronunciation_confidence` are irrelevant for writing; no `word_count` or `task_type` fields

### Missing Features That Motivate the Writing Test Addition

- **No writing practice** — the most commonly paired IELTS module (many candidates take both Speaking and Writing)
- **No Task 1/Task 2 differentiation** — writing has fundamentally different task types (report/letter vs essay)
- **No word count tracking** — critical for IELTS Writing (150 words for Task 1, 250 words for Task 2)
- **No essay structure analysis** — paragraph organization, introduction/conclusion detection, thesis statement evaluation
- **No integrated study plan** — combining speaking and writing progress for a holistic IELTS preparation experience

## 8. Reusable Patterns for Writing Test

| Existing Pattern | Where It Lives | How It Reuses for Writing |
|-----------------|----------------|--------------------------|
| **Provider facade** | `evaluator.py` dispatches to Gemini/Ollama based on env var | Add `evaluate_writing()` and `evaluate_writing_enhanced()` following the same dispatch pattern |
| **Pydantic structured output** | `ContentEvaluation`, `EnhancedReview` in `models.py`; Gemini `response_schema` parameter | Define `WritingEvaluation` with writing-specific criteria (Task Achievement, Coherence & Cohesion, Lexical Resource, Grammatical Range & Accuracy); use the same `CriterionScore` sub-model |
| **System prompts** | `SYSTEM_PROMPT` and `ENHANCED_SYSTEM_PROMPT` in `gemini_evaluator.py` | Add `WRITING_SYSTEM_PROMPT` with IELTS Writing band descriptors, word count expectations, and Task 1 vs Task 2 requirements |
| **Ollama normalization** | `_normalize_evaluation()` with alias maps in `ollama_evaluator.py` | Extend alias maps for writing-specific keys or reuse existing ones (writing shares Lexical Resource and Grammatical Range criteria) |
| **Progressive disclosure UI** | `render_review()` in `review.py` — always-visible scores + expandable details | Add `render_writing_review()` with the same pattern: band scores visible, then expandable grammar corrections, vocabulary upgrades, paragraph analysis |
| **Session tracking** | `create_session()`, `save_attempt()`, `_update_session_stats()` in `database.py` | Reuse `sessions` table with mode `"writing"`; add `writing_attempts` table or extend `attempts` with a `test_type` discriminator |
| **History & weakness aggregation** | `get_detailed_weaknesses()`, `get_criterion_trends()` in `database.py` | Extend aggregation queries to include writing attempts; grammar and vocabulary weakness tracking works identically for writing |
| **Question loading** | `questions.py` loads from DB, picks random variants, assembles mock tests | Add `load_writing_prompts()` with similar logic; writing prompts are simpler (no cue cards, no 3-part structure) |
| **Cached data loading** | `@st.cache_data` on `_load_all()` in `app.py` | Apply same caching pattern to writing prompts |
