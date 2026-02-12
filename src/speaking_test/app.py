import os
import re
import tempfile

import librosa
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from jiwer import wer

from speaking_test.scorer import analyze_audio, estimate_band, generate_feedback

st.set_page_config(page_title="IELTS Speaking Practice", layout="centered")
st.title("IELTS Speaking Practice")


@st.cache_resource
def load_model():
    """Load Whisper model on GPU (RTX 4060 Ti)."""
    return WhisperModel("base.en", device="cuda", compute_type="float16")


def normalize_text(text: str) -> str:
    """Normalize text for fairer WER comparison.

    Lowercases, expands common contractions, and strips punctuation so that
    natural speech variations (e.g. 'don't' vs 'do not') aren't penalized.
    """
    text = text.lower().strip()
    # Expand contractions
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
        "shouldn't": "should not", "isn't": "is not", "aren't": "are not",
        "wasn't": "was not", "weren't": "were not", "haven't": "have not",
        "hasn't": "has not", "hadn't": "had not", "can't": "cannot",
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "you're": "you are", "you've": "you have", "you'll": "you will",
        "he's": "he is", "she's": "she is", "it's": "it is",
        "we're": "we are", "we've": "we have", "we'll": "we will",
        "they're": "they are", "they've": "they have", "they'll": "they will",
        "that's": "that is", "there's": "there is", "here's": "here is",
        "what's": "what is", "who's": "who is", "let's": "let us",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    # Strip punctuation and collapse whitespace
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def show_pitch_chart(audio_path: str):
    """Display a pitch contour chart for the recorded audio."""
    y, sr = librosa.load(audio_path, sr=None)
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    # Keep only voiced frames so the chart isn't cluttered with zeros
    mask = voiced & ~np.isnan(f0)
    st.line_chart(
        {"Pitch (Hz)": f0[mask]},
        x_label="Time (frames)",
        y_label="Hz",
    )


model = load_model()

mode = st.radio("Mode", ["Transcribe", "Practice"], horizontal=True)

if mode == "Transcribe":
    audio = st.audio_input("Record your speech")

    if st.button("Transcribe", type="primary"):
        if audio is None:
            st.warning("Please record your speech first.")
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(audio.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Transcribing audio..."):
                    segments, _info = model.transcribe(
                        tmp_path, language="en", word_timestamps=True
                    )
                    segment_list = list(segments)

                transcript = " ".join(seg.text.strip() for seg in segment_list)

                if not transcript.strip():
                    st.error("No speech detected. Please try recording again.")
                else:
                    st.text_area("Transcript", transcript, height=200)
                    show_pitch_chart(tmp_path)

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

else:
    reference = st.text_area(
        "Paste your reference script",
        height=200,
        placeholder="Paste the IELTS script you want to practice reading aloud...",
    )

    audio = st.audio_input("Record yourself reading the script")

    if st.button("Analyze", type="primary"):
        if not reference.strip():
            st.warning("Please paste a reference script first.")
        elif audio is None:
            st.warning("Please record yourself reading the script.")
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(audio.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Transcribing audio..."):
                    segments, _info = model.transcribe(
                        tmp_path, language="en", word_timestamps=True
                    )
                    segment_list = list(segments)

                transcript = " ".join(seg.text.strip() for seg in segment_list)
                words = [w for seg in segment_list for w in (seg.words or [])]

                if not transcript.strip():
                    st.error("No speech detected. Please try recording again.")
                else:
                    # Normalize both texts for fair comparison
                    ref_norm = normalize_text(reference)
                    hyp_norm = normalize_text(transcript)
                    wer_score = wer(ref_norm, hyp_norm)

                    with st.spinner("Analyzing speech..."):
                        metrics = analyze_audio(tmp_path, transcript, words)

                    band = estimate_band(wer_score, metrics)
                    feedback = generate_feedback(wer_score, metrics, band)

                    st.divider()

                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Band Score", f"{band}")
                    col2.metric("Word Error Rate", f"{wer_score:.0%}")
                    col3.metric("Duration", f"{metrics['duration']}s")

                    col4, col5, col6 = st.columns(3)
                    col4.metric("Speech Rate", f"{metrics['speech_rate']} WPM")
                    col5.metric("Pause Ratio", f"{metrics['pause_ratio']:.0%}")
                    col6.metric("Pronunciation", f"{metrics['pronunciation_confidence']:.0%}")

                    st.divider()
                    st.markdown(feedback)

                    show_pitch_chart(tmp_path)

                    # Show transcription comparison
                    with st.expander("View transcription"):
                        st.markdown("**Your speech:**")
                        st.write(transcript)
                        st.markdown("**Reference script:**")
                        st.write(reference)

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
