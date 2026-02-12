```markdown
# IELTS Speaking MVP (Script + Voice Comparison)

## Full Roadmap (Local, Fast to Build, Free, Personal Use Only)

---

# 1. Goal

Build a **local app** where:

1. You paste your **written script** (reference answer).
2. You record yourself speaking it.
3. The app:
   - Transcribes your speech
   - Compares speech vs script
   - Measures fluency
   - Estimates pronunciation (proxy)
   - Outputs IELTS-style band feedback

This is a **practice evaluator**, not an official scoring system.

---

# 2. Fastest Tech Stack (Minimal Setup)

Use:

- **Python**
- **Streamlit** (simple UI)
- **faster-whisper** (local speech recognition)
- **jiwer** (Word Error Rate comparison)
- **librosa** (audio analysis)
- **numpy + pandas**

Everything runs locally. No paid APIs.

---

# 3. Architecture (Simple)
```

User Script (Text)

- User Audio Recording
  ↓
  Speech-to-Text (Whisper)
  ↓
  Transcript
  ↓
  Analysis Engine:

* WER (accuracy vs script)
* Speech rate
* Pause ratio
* Word confidence
  ↓
  Band Estimation + Feedback

```

---

# 4. Project Structure

```

ielts_speaking_mvp/
│
├── app.py
├── scorer.py
├── requirements.txt
└── models/ (optional if you store models locally)

````

---

# 5. Installation

## Step 1 — Create Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
````

## Step 2 — Install Packages

```bash
pip install streamlit faster-whisper torch jiwer librosa soundfile numpy pandas
```

---

# 6. Scoring Logic Design

We approximate the IELTS 4 criteria:

| IELTS Criterion  | How We Estimate             |
| ---------------- | --------------------------- |
| Fluency          | Speech rate + pause ratio   |
| Coherence        | Script coverage (WER)       |
| Lexical Accuracy | Missing/changed words       |
| Pronunciation    | Word-level confidence score |

---

# 7. Metrics to Calculate

## 7.1 Word Error Rate (Accuracy vs Script)

```python
WER = jiwer.wer(reference_script, spoken_transcript)
```

Interpretation:

| WER       | Meaning       |
| --------- | ------------- |
| < 0.10    | Excellent     |
| 0.10–0.20 | Good          |
| 0.20–0.35 | Moderate      |
| > 0.35    | Weak accuracy |

---

## 7.2 Speech Rate

```
speech_rate = total_words / duration_minutes
```

IELTS natural speaking range:

- 120–160 WPM = ideal
- <100 = slow
- > 180 = too fast

---

## 7.3 Pause Ratio

Use librosa to detect silence:

```
silence_time / total_audio_time
```

Healthy range:

- 15%–30% = natural
- > 40% = hesitant

---

## 7.4 Pronunciation Proxy (Whisper Confidence)

From faster-whisper:

- Each word has `probability`
- Compute average word confidence

```
avg_confidence = mean(word.probability)
```

| Confidence | Interpretation        |
| ---------- | --------------------- |
| >0.90      | Very clear            |
| 0.80–0.90  | Good                  |
| 0.70–0.80  | Needs improvement     |
| <0.70      | Unclear pronunciation |

---

# 8. Band Estimation Logic (Simple Heuristic)

Example mapping:

```python
if WER < 0.10 and confidence > 0.90 and 120 <= speech_rate <= 160:
    band = 8.5
elif WER < 0.20 and confidence > 0.85:
    band = 7.5
elif WER < 0.30:
    band = 6.5
else:
    band = 5.5
```

You can refine later.

---

# 9. Core Implementation

---

## app.py

```python
import streamlit as st
import tempfile
import numpy as np
import librosa
from faster_whisper import WhisperModel
from jiwer import wer
from scorer import analyze_audio, estimate_band

st.title("IELTS Speaking Practice MVP")

script = st.text_area("Paste your reference script here:")

audio_file = st.audio_input("Record your answer")

if audio_file and script:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True)

    transcript = ""
    words = []

    for segment in segments:
        transcript += segment.text
        for word in segment.words:
            words.append(word)

    st.subheader("Transcription")
    st.write(transcript)

    WER = wer(script.lower(), transcript.lower())

    metrics = analyze_audio(audio_path, transcript, words)
    band = estimate_band(WER, metrics)

    st.subheader("Results")
    st.write(f"Word Error Rate: {WER:.2f}")
    st.write(f"Speech Rate: {metrics['speech_rate']:.1f} WPM")
    st.write(f"Pause Ratio: {metrics['pause_ratio']:.2f}")
    st.write(f"Pronunciation Confidence: {metrics['confidence']:.2f}")
    st.write(f"Estimated Band: {band}")
```

---

## scorer.py

```python
import numpy as np
import librosa

def analyze_audio(audio_path, transcript, words):

    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    word_count = len(transcript.split())
    speech_rate = word_count / (duration / 60)

    intervals = librosa.effects.split(y, top_db=30)
    speech_time = sum((end - start) for start, end in intervals) / sr
    pause_ratio = 1 - (speech_time / duration)

    confidences = [w.probability for w in words if w.probability is not None]
    confidence = np.mean(confidences) if confidences else 0.0

    return {
        "speech_rate": speech_rate,
        "pause_ratio": pause_ratio,
        "confidence": confidence
    }

def estimate_band(WER, metrics):

    if WER < 0.10 and metrics["confidence"] > 0.90:
        return 8.5
    elif WER < 0.20 and metrics["confidence"] > 0.85:
        return 7.5
    elif WER < 0.30:
        return 6.5
    else:
        return 5.5
```

---

# 10. Run the App

```bash
streamlit run app.py
```

Open browser → test yourself.

---

# 11. What This MVP Gives You

✔ Measures how accurately you follow your script
✔ Detects hesitation
✔ Measures speaking speed
✔ Estimates clarity
✔ Gives rough IELTS-aligned band

All offline. Free.

---

# 12. Limitations (Important)

- Not true IELTS scoring.
- Whisper confidence ≠ real pronunciation score.
- Does not measure coherence beyond script accuracy.

---

# 13. Phase 2 Improvements (Optional Later)

- Highlight mispronounced words
- Show deleted / inserted words visually
- Add grammar detection
- Add Part 2 timer
- Track progress history
- Add real pronunciation scoring (MFA + GOP)

---

# 14. Expected Build Time

If you know basic Python:

- Setup: 30 min
- Implementation: 2–3 hours
- Testing/refinement: 1–2 hours

Total: ~1 day to working MVP.

---

If you want, next I can generate:

- A cleaner scoring algorithm closer to real IELTS descriptors
- Or a Codex-optimized prompt that will build this entire project automatically

```

```
