# Speaking Test

A local, offline IELTS speaking practice app with two modes:

- **Transcribe** — record speech and get a text transcript with pitch visualization.
- **Practice** — paste a reference script, record yourself reading it, and get feedback on accuracy, fluency, speech rate, and pronunciation with an estimated IELTS band score.

Built with Streamlit and OpenAI's Whisper (via faster-whisper). Runs on GPU (NVIDIA CUDA).

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (tested on RTX 4060 Ti)
- A microphone

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already, then:

```bash
uv sync
```

That's it — `uv sync` creates the virtual environment and installs all dependencies from the lockfile.

## Usage

```bash
uv run streamlit run src/speaking_test/app.py
```

This opens the app in your browser at `http://localhost:8501`.

### Transcribe mode

1. Select **Transcribe**.
2. Click the microphone to record your speech.
3. Click **Transcribe** to get the text and a pitch contour chart.

### Practice mode

1. Select **Practice**.
2. Paste a reference script into the text area.
3. Click the microphone to record yourself reading the script aloud.
4. Click **Analyze** to get your results.

On first run, the Whisper `base.en` model (~150 MB) is downloaded automatically from Hugging Face. After that, everything runs offline.

## What You Get

| Metric | Description |
|--------|-------------|
| **Band Score** | Estimated IELTS band (4.0-9.0) based on weighted accuracy, fluency, and pronunciation |
| **Word Error Rate** | How closely your speech matched the reference script |
| **Speech Rate** | Words per minute (ideal range: 120-160 WPM) |
| **Pause Ratio** | Percentage of silence in your recording |
| **Pronunciation** | Whisper's confidence in recognizing your words |
| **Pitch Contour** | Visual graph of your speech pitch over time |

The app also shows written feedback for each metric and lets you compare your transcription against the reference script.

## How Scoring Works

- **Accuracy (30%)** — derived from word error rate (WER). Both your speech and the reference are normalized (lowercased, punctuation stripped, contractions expanded) so natural variations like "don't" vs "do not" aren't penalized.
- **Fluency (40%)** — based on speech rate and pause ratio. Natural pacing (120-160 WPM) with minimal pauses scores highest.
- **Pronunciation (30%)** — based on Whisper's per-word recognition confidence.

The weighted score is clamped to the 4.0-9.0 IELTS range and rounded to the nearest 0.5.

## Project Structure

```
src/speaking_test/
    __init__.py
    app.py          Streamlit UI — recording, transcription, display
    scorer.py       Analysis logic — metrics, band estimation, feedback
docs/
    spec.md         Original specification
    deep-research-report.md
pyproject.toml      Project metadata and dependencies
uv.lock             Locked dependency versions
```

## Tips for Practice

- Start with short paragraphs and work up to longer passages.
- Record in a quiet environment for best transcription accuracy.
- If your WER is high but you read correctly, try speaking more clearly — Whisper may have misheard you.
- Aim for 120-160 WPM with minimal pauses for the best fluency score.
