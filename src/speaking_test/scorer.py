import librosa
import numpy as np


def analyze_audio(audio_path: str, transcript: str, words: list) -> dict:
    """Compute speech metrics from audio and Whisper output."""
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr if sr > 0 else 0.0

    if duration == 0:
        return {
            "duration": 0.0,
            "speech_rate": 0.0,
            "pause_ratio": 1.0,
            "pronunciation_confidence": 0.0,
        }

    # Speech rate (words per minute)
    word_count = len(transcript.split())
    speech_rate = word_count / (duration / 60) if duration > 0 else 0.0

    # Pause ratio via voice activity detection
    intervals = librosa.effects.split(y, top_db=30)
    speech_time = sum((end - start) for start, end in intervals) / sr
    pause_ratio = 1.0 - (speech_time / duration) if duration > 0 else 1.0
    pause_ratio = max(0.0, min(1.0, pause_ratio))

    # Pronunciation confidence from Whisper word probabilities
    if words:
        pronunciation_confidence = float(np.mean([w.probability for w in words]))
    else:
        pronunciation_confidence = 0.0

    return {
        "duration": round(duration, 2),
        "speech_rate": round(speech_rate, 1),
        "pause_ratio": round(pause_ratio, 3),
        "pronunciation_confidence": round(pronunciation_confidence, 3),
    }


def estimate_band(wer_score: float, metrics: dict) -> float:
    """Estimate an IELTS band score from WER and speech metrics."""
    # Accuracy score from WER (0 = perfect, 1 = all wrong)
    accuracy = max(0.0, 1.0 - wer_score) * 9.0

    # Fluency score from speech rate and pause ratio
    wpm = metrics["speech_rate"]
    if 120 <= wpm <= 160:
        rate_score = 9.0
    elif 100 <= wpm < 120 or 160 < wpm <= 180:
        rate_score = 7.0
    elif 80 <= wpm < 100 or 180 < wpm <= 200:
        rate_score = 5.5
    else:
        rate_score = 4.0

    pause = metrics["pause_ratio"]
    if pause < 0.15:
        pause_score = 9.0
    elif pause < 0.25:
        pause_score = 7.0
    elif pause < 0.40:
        pause_score = 5.5
    else:
        pause_score = 4.0

    fluency = (rate_score + pause_score) / 2

    # Pronunciation score from confidence
    conf = metrics["pronunciation_confidence"]
    pronunciation = min(9.0, max(4.0, conf * 10.0))

    # Weighted average
    raw = accuracy * 0.3 + fluency * 0.4 + pronunciation * 0.3

    # Clamp to IELTS range and round to nearest 0.5
    clamped = max(4.0, min(9.0, raw))
    return round(clamped * 2) / 2


def generate_feedback(wer_score: float, metrics: dict, band: float) -> str:
    """Generate human-readable feedback from metrics."""
    lines = [f"**Estimated Band: {band}**\n"]

    # Accuracy feedback
    wer_pct = round(wer_score * 100, 1)
    if wer_score < 0.05:
        lines.append(f"- **Accuracy:** Excellent — only {wer_pct}% word error rate.")
    elif wer_score < 0.15:
        lines.append(f"- **Accuracy:** Good — {wer_pct}% word error rate. Minor deviations from the script.")
    elif wer_score < 0.30:
        lines.append(f"- **Accuracy:** Fair — {wer_pct}% word error rate. Several words differ from the script.")
    else:
        lines.append(f"- **Accuracy:** Needs work — {wer_pct}% word error rate. Many words differ from the script.")

    # Speech rate feedback
    wpm = metrics["speech_rate"]
    if 120 <= wpm <= 160:
        lines.append(f"- **Speech Rate:** Natural pace at {wpm} WPM.")
    elif wpm < 120:
        lines.append(f"- **Speech Rate:** Slow at {wpm} WPM. Try to speak a bit faster (aim for 120–160 WPM).")
    else:
        lines.append(f"- **Speech Rate:** Fast at {wpm} WPM. Try slowing down for clarity (aim for 120–160 WPM).")

    # Pause ratio feedback
    pause = metrics["pause_ratio"]
    if pause < 0.15:
        lines.append(f"- **Fluency:** Smooth delivery with minimal pauses ({pause:.0%} silence).")
    elif pause < 0.25:
        lines.append(f"- **Fluency:** Some pauses detected ({pause:.0%} silence). Generally fluent.")
    elif pause < 0.40:
        lines.append(f"- **Fluency:** Noticeable pauses ({pause:.0%} silence). Practice reading more continuously.")
    else:
        lines.append(f"- **Fluency:** Frequent pauses ({pause:.0%} silence). Work on reducing hesitations.")

    # Pronunciation feedback
    conf = metrics["pronunciation_confidence"]
    if conf >= 0.85:
        lines.append(f"- **Pronunciation:** Clear and confident (confidence: {conf:.0%}).")
    elif conf >= 0.70:
        lines.append(f"- **Pronunciation:** Generally clear (confidence: {conf:.0%}). Some words could be sharper.")
    elif conf >= 0.50:
        lines.append(f"- **Pronunciation:** Some unclear words (confidence: {conf:.0%}). Focus on enunciation.")
    else:
        lines.append(f"- **Pronunciation:** Needs improvement (confidence: {conf:.0%}). Practice individual word clarity.")

    return "\n".join(lines)
