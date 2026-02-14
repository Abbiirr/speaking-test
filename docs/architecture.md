# Architecture

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UI Layer (Streamlit)                      │
│                                                                  │
│  app.py (789 lines)                                              │
│  ┌──────────┬───────────┬──────────┬────────────┬──────────────┐ │
│  │Interview │ Mock Test │ Practice │ Transcribe │   History    │ │
│  └────┬─────┴─────┬─────┴────┬─────┴──────┬─────┴──────┬───────┘ │
│       │           │          │            │            │          │
│       │           │          │            │    review.py (209)    │
│       │           │          │            │    render_review()    │
│       │           │          │            │    render_review_     │
│       │           │          │            │      from_dict()      │
└───────┼───────────┼──────────┼────────────┼────────────┼─────────┘
        │           │          │            │            │
┌───────┼───────────┼──────────┼────────────┼────────────┼─────────┐
│       ▼           ▼          │            │            │          │
│             Evaluation Layer │            │            │          │
│                              │            │            │          │
│  evaluator.py (84) ─── facade/dispatcher  │            │          │
│       │           │                       │            │          │
│  ┌────▼────┐ ┌────▼──────┐  ┌─────────┐  │            │          │
│  │ Gemini  │ │  Ollama   │  │scorer.py│  │            │          │
│  │  (313)  │ │  (269)    │  │  (138)  │◄─┘            │          │
│  └─────────┘ └───────────┘  └─────────┘               │          │
│                                                        │          │
│  models.py (175) ─── Pydantic schemas + dataclasses    │          │
└────────────────────────────────────────────────────────┼──────────┘
                                                         │
┌────────────────────────────────────────────────────────┼──────────┐
│                        Data Layer                      │          │
│                                                        │          │
│  database.py (439) ─── SQLite persistence ◄────────────┘          │
│       │                                                           │
│  questions.py (145) ─── question loading + mock test assembly     │
│       │                                                           │
│  questions_answers_updated.csv ─── 691 questions, 1980 rows       │
│       │                                                           │
│  data/history.db ─── sessions + attempts tables                   │
└───────────────────────────────────────────────────────────────────┘
```

**Flow summary:** The UI layer captures audio, delegates transcription to Whisper, sends transcripts to the evaluation layer (AI content scoring + audio analysis), and persists results through the data layer. The `review.py` module renders evaluation results back to the user with progressive disclosure.

## 2. Module Responsibility Map

| # | File | Lines | Responsibility | Key Public APIs |
|---|------|------:|----------------|-----------------|
| 1 | `app.py` | 789 | UI orchestration — sidebar nav, 5 mode renderers, audio capture, transcription, evaluation dispatch, result display | `render_interview_mode()`, `render_mock_test_mode()`, `render_practice_mode()`, `render_transcribe_mode()`, `render_history_mode()`, `transcribe_audio()`, `save_attempt_from_eval()` |
| 2 | `models.py` | 175 | All data models — Pydantic schemas for AI evaluation, dataclasses for questions and DB records | `Question`, `QuestionWithAnswer`, `CriterionScore`, `ContentEvaluation`, `EnhancedReview`, `GrammarCorrection`, `VocabularyUpgrade`, `PronunciationWarning`, `MockTestPlan`, `MockTestResponse`, `MockTestState`, `SessionRecord`, `AttemptRecord` |
| 3 | `database.py` | 439 | SQLite persistence — schema init, CSV seeding, CRUD for sessions/attempts, trend and weakness aggregation | `get_db()`, `create_session()`, `save_attempt()`, `get_band_trend()`, `get_criterion_trends()`, `get_weak_areas()`, `get_recent_sessions()`, `get_attempts_for_session()`, `get_detailed_weaknesses()`, `get_all_questions_from_db()` |
| 4 | `evaluator.py` | 84 | Provider-agnostic facade — dispatches evaluation calls to Gemini or Ollama based on `PROVIDER` env var | `get_provider()`, `is_provider_configured()`, `evaluate_answer()`, `evaluate_answer_enhanced()`, `compute_combined_band()`, `detect_fillers()` |
| 5 | `gemini_evaluator.py` | 313 | Gemini integration — system prompts, structured output via Pydantic schemas, combined band computation, filler detection | `create_gemini_client()`, `get_model_name()`, `evaluate_answer()`, `evaluate_answer_enhanced()`, `compute_combined_band()`, `detect_fillers()` |
| 6 | `ollama_evaluator.py` | 269 | Ollama integration — chat API, think-tag stripping, JSON extraction, key normalization layer | `is_available()`, `evaluate_answer()`, `evaluate_answer_enhanced()`, `_normalize_evaluation()`, `_chat()` |
| 7 | `scorer.py` | 138 | Audio analysis — speech rate, pause ratio, pronunciation confidence, WER-based band estimation, feedback generation | `analyze_audio()`, `estimate_band()`, `generate_feedback()` |
| 8 | `review.py` | 209 | Review renderer — progressive disclosure UI for evaluation results, both live and from DB history | `render_review()`, `render_review_from_dict()` |
| 9 | `questions.py` | 145 | Question loading — DB queries, random variant selection, mock test plan assembly with topic matching | `load_all_questions()`, `get_random_question()`, `get_all_topics()`, `assemble_mock_test()` |
| 10 | `__init__.py` | 34 | Package init — centralized logging setup (file handler DEBUG + console handler ERROR) | *(logging configuration only)* |

## 3. Data Flow Diagrams

### 3a. Interview / Mock Test (full AI evaluation pipeline)

```
User records audio
       │
       ▼
┌─────────────────┐
│ st.audio_input() │
└────────┬────────┘
         │  .wav bytes
         ▼
┌─────────────────────────────┐
│ tempfile → tmp_path          │
│ WhisperModel.transcribe()    │
│   → transcript, words        │
└────────┬────────────────────┘
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
┌──────────────┐    ┌──────────────────────────┐
│ scorer.py    │    │ evaluator.py (facade)     │
│ analyze_audio│    │   → gemini_evaluator.py   │
│              │    │   OR ollama_evaluator.py  │
│ Returns:     │    │                           │
│  duration    │    │ Returns:                  │
│  speech_rate │    │  ContentEvaluation        │
│  pause_ratio │    │  OR EnhancedReview        │
│  pron_conf   │    │  (4 criterion scores +    │
│  long_pauses │    │   feedback + corrections) │
└──────┬───────┘    └────────────┬──────────────┘
       │                         │
       └──────────┬──────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │ compute_combined_band()   │
    │                           │
    │ Blends AI scores + audio: │
    │  fluency = 50% audio +    │
    │           50% coherence   │
    │  lexical = 100% AI        │
    │  grammar = 100% AI        │
    │  pronunciation = 100%     │
    │    audio (Whisper conf)   │
    │  overall = avg of 4 / 25% │
    └────────────┬──────────────┘
                 │
    ┌────────────┴──────────────┐
    │                           │
    ▼                           ▼
┌────────────┐     ┌────────────────────┐
│ review.py  │     │ database.py        │
│ render_    │     │ save_attempt()     │
│ review()   │     │ _update_session_   │
│            │     │   stats()          │
└────────────┘     └────────────────────┘
```

### 3b. Practice Mode (audio-only scoring via WER)

```
User pastes reference script + records audio
       │
       ▼
┌──────────────────┐     ┌──────────────────────┐
│ st.text_area()    │     │ st.audio_input()      │
│ → reference text  │     │ → .wav bytes          │
└────────┬──────────┘     └────────┬──────────────┘
         │                         │
         │                         ▼
         │              ┌──────────────────────┐
         │              │ transcribe_audio()    │
         │              │ → transcript, words   │
         │              └────────┬──────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│ normalize_text() on both reference and  │
│ transcript (lowercase, expand           │
│ contractions, strip punctuation)        │
│                                         │
│ jiwer.wer(ref_norm, hyp_norm)           │
│ → wer_score                             │
└──────────────────┬──────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
┌──────────────┐  ┌────────────────────┐
│ scorer.py    │  │ scorer.py          │
│ analyze_     │  │ estimate_band()    │
│ audio()      │  │ WER × 0.3 +       │
│              │  │ fluency × 0.4 +    │
│              │  │ pronunciation × 0.3│
└──────┬───────┘  └────────┬───────────┘
       │                   │
       └────────┬──────────┘
                │
                ▼
    ┌──────────────────────┐
    │ scorer.py            │
    │ generate_feedback()  │
    │ → markdown string    │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ st.metric() columns  │
    │ st.markdown(feedback)│
    │ show_pitch_chart()   │
    └──────────────────────┘
```

### 3c. Question Loading (CSV → SQLite → session cache)

```
App startup
     │
     ▼
┌──────────────────────────┐
│ database.get_db()         │
│   → _get_connection()     │
│   → _init_db()            │  Creates tables if
│   → _seed_questions()     │  not exist, imports
│                           │  CSV once if table
│  questions_answers_       │  is empty
│  updated.csv ──────►     │
│  1980 rows → questions    │
│  table (part, topic,      │
│  question_text, cue_card, │
│  source, band9_answer,    │
│  answer_variant)          │
└────────────┬──────────────┘
             │
             ▼
┌──────────────────────────┐
│ questions.py              │
│ load_all_questions()      │
│   → get_all_questions_    │
│     from_db()             │
│                           │
│  Groups by (part, text),  │
│  picks 1 random variant   │
│  per question             │
│  → 691 QuestionWithAnswer │
└────────────┬──────────────┘
             │
             ▼
┌──────────────────────────┐
│ app.py                    │
│ @st.cache_data            │
│ _load_all()               │
│                           │
│ Random variant choice is  │
│ stable per session via    │
│ Streamlit cache           │
└───────────────────────────┘
```

## 4. Database Schema

The database is stored at `data/history.db` (SQLite, WAL journal mode).

```sql
-- Tracks practice sessions (one per Interview question or Mock Test run)
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,          -- ISO 8601 UTC
    mode            TEXT NOT NULL,          -- "interview" | "mock_test" | "practice"
    overall_band    REAL DEFAULT 0.0,       -- Average band across attempts (auto-updated)
    attempt_count   INTEGER DEFAULT 0       -- Number of attempts in session (auto-updated)
);

-- Individual question attempts with full scoring data
CREATE TABLE IF NOT EXISTS attempts (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id                INTEGER NOT NULL,         -- FK → sessions.id
    timestamp                 TEXT NOT NULL,             -- ISO 8601 UTC
    part                      INTEGER DEFAULT 0,        -- IELTS part: 1, 2, or 3
    topic                     TEXT DEFAULT '',           -- e.g. "Flowers and plants"
    question_text             TEXT DEFAULT '',           -- The question asked
    transcript                TEXT DEFAULT '',           -- Whisper transcription
    duration                  REAL DEFAULT 0.0,          -- Audio duration in seconds
    overall_band              REAL DEFAULT 0.0,          -- Combined band score
    fluency_coherence         REAL DEFAULT 0.0,          -- 50% audio + 50% AI coherence
    lexical_resource          REAL DEFAULT 0.0,          -- 100% AI
    grammatical_range         REAL DEFAULT 0.0,          -- 100% AI
    pronunciation             REAL DEFAULT 0.0,          -- 100% audio (Whisper confidence)
    speech_rate               REAL DEFAULT 0.0,          -- Words per minute
    pause_ratio               REAL DEFAULT 0.0,          -- Fraction of silence
    pronunciation_confidence  REAL DEFAULT 0.0,          -- Whisper word probability mean
    examiner_feedback         TEXT DEFAULT '',            -- AI overall_feedback string
    grammar_corrections       TEXT DEFAULT '',            -- JSON: [{original, corrected, explanation}]
    vocabulary_upgrades       TEXT DEFAULT '',            -- JSON: [{basic_word, alternatives, example}]
    improvement_tips          TEXT DEFAULT '',            -- JSON: [string]
    band9_answer              TEXT DEFAULT '',            -- Reference answer shown to user
    strengths                 TEXT DEFAULT '',            -- JSON: [string]
    pronunciation_warnings    TEXT DEFAULT '',            -- JSON: [{word, phonetic, tip}]
    source                    TEXT DEFAULT '',            -- Question source identifier
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Question bank seeded from CSV at first startup
CREATE TABLE IF NOT EXISTS questions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    part            INTEGER NOT NULL,       -- 1, 2, or 3
    topic           TEXT DEFAULT '',         -- Topic grouping
    question_text   TEXT NOT NULL,           -- The question
    cue_card        TEXT DEFAULT '',         -- Part 2 cue card bullet points
    source          TEXT DEFAULT '',         -- "question_bank" | "master_pack" | etc.
    band9_answer    TEXT DEFAULT '',         -- Model answer
    answer_variant  TEXT DEFAULT ''          -- "A", "B", or "C"
);
```

**Notes:**
- `sessions.overall_band` and `attempt_count` are auto-updated by `_update_session_stats()` after each `save_attempt()` call
- JSON columns (`grammar_corrections`, `vocabulary_upgrades`, etc.) store serialized lists; parsed back via `json.loads()` in `get_attempts_for_session()`
- The `questions` table is seeded once from `questions_answers_updated.csv` via `_seed_questions()`; subsequent runs skip seeding if the table has rows
- Schema migration for new columns uses `ALTER TABLE ... ADD COLUMN` with `try/except` to handle already-existing columns

## 5. Scoring Methodology

### Interview / Mock Test: Combined Band (AI + Audio)

The overall band is the equally weighted average (25% each) of four IELTS criteria:

| Criterion | Source | Formula |
|-----------|--------|---------|
| Fluency & Coherence | 50% audio + 50% AI | `0.5 * audio_fluency + 0.5 * coherence.score` |
| Lexical Resource | 100% AI | `lexical_resource.score` (from LLM evaluation) |
| Grammatical Range | 100% AI | `grammatical_range.score` (from LLM evaluation) |
| Pronunciation | 100% audio | `min(9.0, max(4.0, whisper_confidence * 10.0))` |

**Overall band** = `round((F&C + LR + GR + P) / 4 * 2) / 2` (nearest 0.5, clamped 4.0–9.0)

#### Audio Sub-Scoring Thresholds

**Speech Rate (WPM):**

| WPM Range | Score |
|-----------|-------|
| 120–160 | 9.0 |
| 100–119 or 161–180 | 7.0 |
| 80–99 or 181–200 | 5.5 |
| < 80 or > 200 | 4.0 |

**Pause Ratio (fraction of silence):**

| Pause Ratio | Score |
|-------------|-------|
| < 0.15 | 9.0 |
| 0.15–0.24 | 7.0 |
| 0.25–0.39 | 5.5 |
| >= 0.40 | 4.0 |

**Audio Fluency** = `(rate_score + pause_score) / 2`

**Pronunciation Band** = `min(9.0, max(4.0, whisper_word_probability_mean * 10.0))`

### Practice Mode: WER-Based Estimation

Practice mode uses `scorer.estimate_band()` with different weights (no AI involved):

| Component | Weight | Source |
|-----------|--------|--------|
| Accuracy | 30% | `(1.0 - wer_score) * 9.0` |
| Fluency | 40% | Same speech rate + pause ratio thresholds as above |
| Pronunciation | 30% | Same Whisper confidence formula |

**Band** = `round(weighted_average * 2) / 2` (nearest 0.5, clamped 4.0–9.0)

## 6. Provider Abstraction

### Facade Pattern (`evaluator.py`)

```
evaluator.py
├── get_provider()           →  reads PROVIDER env var ("gemini" | "ollama")
├── is_provider_configured() →  checks API key or Ollama reachability
├── evaluate_answer()        →  dispatches to provider, returns ContentEvaluation
├── evaluate_answer_enhanced()→  dispatches to provider, returns EnhancedReview
├── compute_combined_band()  →  re-exported from gemini_evaluator
└── detect_fillers()         →  re-exported from gemini_evaluator (regex-based, no API)
```

### Gemini Provider (`gemini_evaluator.py`)

- Uses `google-genai` SDK with `response_mime_type="application/json"` and `response_schema=<PydanticModel>`
- Structured output: Gemini returns JSON that directly conforms to the Pydantic schema
- System prompts define IELTS examiner role with band descriptors for each criterion
- Two prompt variants: `SYSTEM_PROMPT` (standard) and `ENHANCED_SYSTEM_PROMPT` (adds grammar corrections, vocabulary upgrades, pronunciation warnings, strengths, improvement priorities)
- Temperature: 0.3
- Default model: `gemini-2.5-flash-lite` (configurable via `GEMINI_MODEL` env var)

### Ollama Provider (`ollama_evaluator.py`)

- Uses Ollama's `/api/chat` endpoint via `httpx`
- Default model: `deepseek-r1:8b` (configurable via `OLLAMA_MODEL` env var)
- Requests `"format": "json"` for JSON mode
- Post-processing pipeline:
  1. `_strip_think_tags()` — removes `<think>...</think>` blocks from DeepSeek-R1
  2. `_extract_json()` — handles markdown code fences
  3. `_normalize_evaluation()` — maps varied key formats to canonical Pydantic schema
- Normalization handles three response formats:
  - **Flat keys**: `{"coherence_score": 7, "coherence_feedback": "..."}`
  - **Nested objects**: `{"coherence": {"score": 7, "feedback": "..."}}`
  - **Bare values**: `{"coherence": 7}`
- Extensive alias maps (`_SCORE_ALIASES`, `_FEEDBACK_ALIASES`, `_NESTED_ALIASES`) handle alternative key names from smaller models (e.g., `"grammar_score"` → `"grammatical_range_score"`)
- Reuses the same system prompts as Gemini, with appended flat-key schema instructions

## 7. Key Design Decisions

### Pydantic vs Dataclass Split

- **Pydantic `BaseModel`** is used for AI evaluation schemas (`ContentEvaluation`, `EnhancedReview`, `CriterionScore`, `GrammarCorrection`, `VocabularyUpgrade`, `PronunciationWarning`) — these benefit from Pydantic's JSON validation, `model_validate_json()`, and structured output support with Gemini
- **`@dataclass`** is used for internal data structures (`Question`, `QuestionWithAnswer`, `MockTestPlan`, `MockTestState`, `MockTestResponse`, `SessionRecord`, `AttemptRecord`) — simpler, mutable, no validation overhead needed

### Single-File UI (`app.py`)

All 5 modes are rendered from `app.py` with `st.radio()` sidebar navigation. Each mode is a standalone function (`render_*_mode()`). This keeps the Streamlit entry point simple and avoids multi-page complexity for what is currently a single-purpose app.

### Session State for Multi-Step Flows

The Mock Test mode uses `st.session_state["mock_test"]` to store a `MockTestState` dataclass across Streamlit reruns. This tracks progress through Parts 1→2→3, the prep timer state, and accumulated responses. Interview mode similarly stores the current question and session ID in session state.

### JSON Columns in SQLite

Rich evaluation data (grammar corrections, vocabulary upgrades, strengths, improvement tips, pronunciation warnings) is stored as JSON strings in TEXT columns rather than normalized tables. This avoids schema complexity for what is essentially append-only display data. Parsed back to Python lists/dicts via `json.loads()` when reading from the database.

### CSV Seeding with Answer Variants

The question bank CSV contains multiple answer variants per question (A, B, C) to provide variety. At load time, `get_all_questions_from_db()` groups by `(part, question_text)` and picks one random variant per question. The `@st.cache_data` decorator on `_load_all()` ensures the random choice is stable within a session.

### Centralized Logging

The `__init__.py` sets up a package-level logger with dual handlers: file (DEBUG level, UTF-8) and console (ERROR only). All modules use `logging.getLogger(__name__)` to inherit this configuration.

## 8. Extension Points for Writing Test

### Models (`models.py`)

- Add new Pydantic schemas for writing evaluation (e.g., `WritingEvaluation`, `WritingEnhancedReview`) with IELTS Writing criteria: Task Achievement, Coherence & Cohesion, Lexical Resource, Grammatical Range & Accuracy
- Add dataclasses for writing-specific state (e.g., `WritingTestState`, `WritingAttemptRecord`)
- The existing `CriterionScore` model can be reused as-is for writing criteria scoring

### Evaluator Facade (`evaluator.py`)

- Add `evaluate_writing()` and `evaluate_writing_enhanced()` dispatch functions following the same pattern as `evaluate_answer()` / `evaluate_answer_enhanced()`
- Add `compute_writing_band()` — writing has no audio component, so this is simpler (pure AI scores)

### System Prompts (`gemini_evaluator.py`, `ollama_evaluator.py`)

- Add `WRITING_SYSTEM_PROMPT` and `WRITING_ENHANCED_SYSTEM_PROMPT` with IELTS Writing band descriptors
- Writing prompts would reference Task 1 (report/letter) vs Task 2 (essay) requirements, word count expectations, and different scoring criteria than speaking

### Database Schema (`database.py`)

- Add a `writing_attempts` table (or extend `attempts` with a `test_type` column) with writing-specific fields: `essay_text`, `word_count`, `task_type` (1 or 2), and writing-specific criterion scores
- The existing `sessions` table can be reused with a new mode value (e.g., `"writing"`)

### UI Modes (`app.py`)

- Add `"Writing"` to the sidebar radio options
- Add `render_writing_mode()` function with:
  - Task prompt display (Task 1 or Task 2)
  - `st.text_area()` for essay input (replaces `st.audio_input()`)
  - Word count indicator
  - Timer (optional, 20 min for Task 1, 40 min for Task 2)
- No audio processing needed — the entire Whisper/scorer pipeline is bypassed

### Review Renderer (`review.py`)

- Add `render_writing_review()` — similar structure to `render_review()` but without delivery metrics (no speech rate, pause ratio, pitch chart)
- Can reuse the grammar corrections, vocabulary upgrades, and improvement priorities sections
- Add writing-specific sections: paragraph structure analysis, word count feedback, task achievement details

### Question Bank

- Add writing prompts to the CSV or a separate `writing_questions.csv` with columns: `task_type`, `topic`, `prompt_text`, `band9_essay`
- Extend `questions.py` with `load_writing_questions()` and writing-specific assembly logic
