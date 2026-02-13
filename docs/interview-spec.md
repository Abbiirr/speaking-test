# Interview Mode — Feature Specification

## Overview

Interview mode simulates an IELTS speaking exam. The app presents random IELTS questions from a curated question bank, records the user's spontaneous spoken answer, transcribes it with Whisper, and evaluates it using both audio analysis and Gemini AI — producing a combined IELTS band score across all four official criteria.

## User Flow

1. User selects **Interview** mode from the top radio buttons
2. Optionally filters by IELTS part (Part 1 / Part 2 / Part 3)
3. Clicks **"New Question"** — a random question appears (with cue card for Part 2)
4. Records their spoken answer using the audio widget
5. Clicks **"Evaluate"** — the system:
   - Transcribes speech with Whisper (local, on GPU)
   - Computes audio delivery metrics (speech rate, pause ratio, pronunciation confidence)
   - Sends transcript to Gemini for content evaluation against IELTS criteria
   - Blends both into a combined band score
6. Results displayed: overall band, 4 criterion scores, delivery metrics, examiner feedback, pitch chart, transcript, and band-9 reference answer

## Technical Architecture

```
Audio Recording
      │
      ▼
┌─────────────┐     ┌──────────────────┐
│   Whisper    │────►│  Audio Metrics   │
│ (local GPU)  │     │  (scorer.py)     │
└──────┬──────┘     └────────┬─────────┘
       │ transcript          │ speech_rate, pause_ratio,
       │                     │ pronunciation_confidence
       ▼                     │
┌──────────────┐             │
│  Gemini API  │             │
│  (evaluator) │             │
└──────┬───────┘             │
       │ content scores      │
       ▼                     ▼
┌──────────────────────────────┐
│   Combined Band Calculator   │
│  (compute_combined_band)     │
└──────────────┬───────────────┘
               │
               ▼
         Final Scores
```

### Key Modules

| Module | Responsibility |
|--------|---------------|
| `questions.py` | Parse `ielts_questions.md` and `ielts_band9_answers.md` into structured Question/Answer pairs |
| `gemini_evaluator.py` | Gemini API client, IELTS examiner prompt, structured JSON scoring, combined band calculation |
| `scorer.py` | Audio-based metrics: speech rate, pause ratio, pronunciation confidence (reused from Practice mode) |
| `app.py` | Streamlit UI orchestration for all three modes |

## Scoring Methodology

### IELTS Band Criteria (equal 25% weight each)

| Criterion | Source | Blend |
|-----------|--------|-------|
| **Fluency & Coherence** | Audio fluency (speech rate + pause ratio) + Gemini coherence score | 50% audio / 50% Gemini |
| **Lexical Resource** | Gemini evaluation only | 100% Gemini |
| **Grammatical Range & Accuracy** | Gemini evaluation only | 100% Gemini |
| **Pronunciation** | Whisper word-level confidence scores | 100% audio |

### Audio Fluency Scoring

Speech rate (WPM):
- 120–160 WPM → 9.0
- 100–120 or 160–180 → 7.0
- 80–100 or 180–200 → 5.5
- Outside range → 4.0

Pause ratio:
- < 15% → 9.0
- 15–25% → 7.0
- 25–40% → 5.5
- > 40% → 4.0

Audio fluency = average of speech rate score and pause score.

### Pronunciation Scoring

Based on Whisper's word-level probability scores, mapped to a 4.0–9.0 band range via `min(9.0, max(4.0, confidence * 10.0))`.

### Overall Band

Average of all four criteria, rounded to nearest 0.5, clamped to 4.0–9.0.

## Gemini Prompt Design

The system prompt instructs Gemini to:
- Act as an experienced IELTS speaking examiner
- Evaluate **content only** (vocabulary, grammar, coherence, task response)
- **Not** assess pronunciation or fluency (handled by audio analysis)
- Use the band-9 reference answer for calibration, not as the only valid answer
- Return structured JSON matching the `GeminiEvaluation` Pydantic schema
- Use 0.5-increment scoring on the 0–9 band scale

Temperature is set to 0.3 for consistent, repeatable evaluations.

## Data Flow

1. **Question bank** (`docs/ielts_questions.md`) → parsed at app start, cached
2. **Band-9 answers** (`docs/ielts_band9_answers.md`) → paired with questions via fuzzy text matching
3. **User audio** → temp WAV file → Whisper transcription + audio metrics
4. **Transcript + question + band-9 answer** → Gemini API → structured evaluation
5. **Gemini scores + audio metrics** → combined band calculator → final display

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model to use |

Set in `.env` file in the project root.

## Error Handling

- **Missing API key**: Clear error message shown at the top of Interview mode with setup instructions
- **Gemini API failure**: Graceful fallback — audio-only metrics are still displayed with a warning
- **No speech detected**: Error message prompting re-recording
- **Empty question bank**: Falls back to full question list if part filter yields no results

## Future Considerations

- Session history: track scores across multiple questions in a session
- Timed responses: enforce Part 1 (30s), Part 2 (2min), Part 3 (1min) time limits
- Follow-up questions: simulate examiner follow-ups based on the user's answer
- Custom question bank: allow users to add their own questions
- Export results: PDF/CSV report of practice session scores
