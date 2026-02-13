import json
import os
import re
import tempfile
import time

import librosa
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from jiwer import wer

from speaking_test.database import (
    create_session,
    get_attempts_for_session,
    get_band_trend,
    get_criterion_trends,
    get_recent_sessions,
    get_weak_areas,
    save_attempt,
)
from speaking_test.evaluator import (
    compute_combined_band,
    detect_fillers,
    evaluate_answer,
    evaluate_answer_enhanced,
    get_provider,
    is_provider_configured,
)
from speaking_test.models import (
    AttemptRecord,
    EnhancedReview,
    MockTestResponse,
    MockTestState,
)
from speaking_test.questions import (
    assemble_mock_test,
    get_random_question,
    load_all_questions,
    load_questions,
)
from speaking_test.review import render_review, render_review_from_dict
from speaking_test.scorer import analyze_audio, estimate_band, generate_feedback

load_dotenv()

st.set_page_config(page_title="IELTS Speaking Practice", layout="centered")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load Whisper model on GPU (RTX 4060 Ti)."""
    return WhisperModel("base.en", device="cuda", compute_type="float16")


@st.cache_data
def _load_all():
    return load_all_questions()


def normalize_text(text: str) -> str:
    """Normalize text for fairer WER comparison."""
    text = text.lower().strip()
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
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def show_pitch_chart(audio_path: str):
    """Display a pitch contour chart for the recorded audio."""
    y, sr = librosa.load(audio_path, sr=None)
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    mask = voiced & ~np.isnan(f0)
    st.line_chart(
        {"Pitch (Hz)": f0[mask]},
        x_label="Time (frames)",
        y_label="Hz",
    )


def transcribe_audio(model, tmp_path):
    """Transcribe audio and return (transcript, words, segment_list)."""
    segments, _info = model.transcribe(
        tmp_path, language="en", word_timestamps=True
    )
    segment_list = list(segments)
    transcript = " ".join(seg.text.strip() for seg in segment_list)
    words = [w for seg in segment_list for w in (seg.words or [])]
    return transcript, words, segment_list


def save_attempt_from_eval(
    session_id: int,
    question_text: str,
    part: int,
    topic: str,
    transcript: str,
    metrics: dict,
    combined: dict | None,
    evaluation=None,
):
    """Save an attempt record to the database."""
    record = AttemptRecord(
        session_id=session_id,
        part=part,
        topic=topic,
        question_text=question_text,
        transcript=transcript,
        duration=metrics.get("duration", 0),
        overall_band=combined.get("overall_band", 0) if combined else 0,
        fluency_coherence=combined.get("fluency_coherence", 0) if combined else 0,
        lexical_resource=combined.get("lexical_resource", 0) if combined else 0,
        grammatical_range=combined.get("grammatical_range", 0) if combined else 0,
        pronunciation=combined.get("pronunciation", 0) if combined else 0,
        speech_rate=metrics.get("speech_rate", 0),
        pause_ratio=metrics.get("pause_ratio", 0),
        pronunciation_confidence=metrics.get("pronunciation_confidence", 0),
        examiner_feedback=evaluation.overall_feedback if evaluation else "",
    )

    if isinstance(evaluation, EnhancedReview):
        record.grammar_corrections = json.dumps(
            [gc.model_dump() for gc in evaluation.grammar_corrections]
        )
        record.vocabulary_upgrades = json.dumps(
            [vu.model_dump() for vu in evaluation.vocabulary_upgrades]
        )
        record.improvement_tips = json.dumps(evaluation.improvement_priorities)

    return save_attempt(record)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

model = load_model()

with st.sidebar:
    st.title("IELTS Speaking")
    mode = st.radio(
        "Mode",
        ["Interview", "Mock Test", "Practice", "Transcribe", "History"],
        label_visibility="collapsed",
    )

    weak = get_weak_areas()
    if weak:
        weakest = min(weak, key=weak.get)
        st.caption(f"Focus area: **{weakest}** ({weak[weakest]})")


# ---------------------------------------------------------------------------
# Transcribe mode
# ---------------------------------------------------------------------------

def render_transcribe_mode():
    st.header("Transcribe")
    audio = st.audio_input("Record your speech")

    if st.button("Transcribe", type="primary"):
        if audio is None:
            st.warning("Please record your speech first.")
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Transcribing audio..."):
                    transcript, words, _ = transcribe_audio(model, tmp_path)

                if not transcript.strip():
                    st.error("No speech detected. Please try recording again.")
                else:
                    st.text_area("Transcript", transcript, height=200)
                    show_pitch_chart(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Practice mode
# ---------------------------------------------------------------------------

def render_practice_mode():
    st.header("Practice")
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
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Transcribing audio..."):
                    transcript, words, _ = transcribe_audio(model, tmp_path)

                if not transcript.strip():
                    st.error("No speech detected. Please try recording again.")
                else:
                    ref_norm = normalize_text(reference)
                    hyp_norm = normalize_text(transcript)
                    wer_score = wer(ref_norm, hyp_norm)

                    with st.spinner("Analyzing speech..."):
                        metrics = analyze_audio(tmp_path, transcript, words)

                    band = estimate_band(wer_score, metrics)
                    feedback = generate_feedback(wer_score, metrics, band)

                    st.divider()
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

                    with st.expander("View transcription"):
                        st.markdown("**Your speech:**")
                        st.write(transcript)
                        st.markdown("**Reference script:**")
                        st.write(reference)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Interview mode
# ---------------------------------------------------------------------------

def render_interview_mode():
    st.header("Interview")

    provider = get_provider()
    provider_ready = is_provider_configured()
    if not provider_ready:
        if provider == "ollama":
            st.error(
                "**Ollama is not reachable.** Make sure it's running:\n\n"
                "```\nollama serve\n```"
            )
        else:
            st.error(
                "**GEMINI_API_KEY not found.** Add it to your `.env` file:\n\n"
                "```\nGEMINI_API_KEY=your-key-here\n```"
            )

    all_questions = _load_all()

    col_filter, col_deep = st.columns([2, 1])
    with col_filter:
        part_option = st.selectbox(
            "Filter by part",
            ["Any", "Part 1", "Part 2", "Part 3"],
        )
    with col_deep:
        deep_review = st.checkbox("Deep Review", value=True)

    part_filter = None
    if part_option != "Any":
        part_filter = int(part_option.split()[-1])

    if st.button("New Question", type="primary"):
        qwa = get_random_question(all_questions, part=part_filter)
        st.session_state["interview_question"] = qwa
        # Create a new session for this interview question
        st.session_state["interview_session_id"] = create_session("interview")

    qwa = st.session_state.get("interview_question")
    if qwa is None:
        st.info('Click **"New Question"** to get started.')
        return

    q = qwa.question
    st.subheader(f"Part {q.part} — {q.topic}")
    st.markdown(f"**{q.text}**")
    if q.cue_card:
        st.info(q.cue_card)

    audio = st.audio_input("Record your answer")

    if st.button("Evaluate", type="secondary"):
        if audio is None:
            st.warning("Please record your answer first.")
        elif not provider_ready:
            st.error("Cannot evaluate — AI provider is not configured.")
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Transcribing your answer..."):
                    transcript, words, _ = transcribe_audio(model, tmp_path)

                if not transcript.strip():
                    st.error("No speech detected. Please try recording again.")
                else:
                    with st.spinner("Analyzing speech delivery..."):
                        metrics = analyze_audio(tmp_path, transcript, words)

                    content_eval = None
                    combined = None
                    try:
                        if deep_review:
                            with st.spinner("Running deep AI review..."):
                                content_eval = evaluate_answer_enhanced(
                                    question=q.text,
                                    part=q.part,
                                    transcript=transcript,
                                    band9_answer=qwa.band9_answer,
                                )
                        else:
                            with st.spinner("Evaluating content with AI examiner..."):
                                content_eval = evaluate_answer(
                                    question=q.text,
                                    part=q.part,
                                    transcript=transcript,
                                    band9_answer=qwa.band9_answer,
                                )
                        combined = compute_combined_band(content_eval, metrics)
                    except Exception as e:
                        st.warning(
                            f"AI evaluation failed ({provider}): {e}\n\n"
                            "Showing audio-only metrics below."
                        )

                    st.divider()

                    if combined:
                        render_review(
                            combined=combined,
                            evaluation=content_eval,
                            metrics=metrics,
                            transcript=transcript,
                            band9_answer=qwa.band9_answer,
                            pitch_chart_fn=show_pitch_chart,
                            audio_path=tmp_path,
                        )

                        # Auto-save to database
                        session_id = st.session_state.get("interview_session_id")
                        if not session_id:
                            session_id = create_session("interview")
                            st.session_state["interview_session_id"] = session_id
                        save_attempt_from_eval(
                            session_id=session_id,
                            question_text=q.text,
                            part=q.part,
                            topic=q.topic,
                            transcript=transcript,
                            metrics=metrics,
                            combined=combined,
                            evaluation=content_eval,
                        )
                    else:
                        # Fallback: show basic metrics
                        st.markdown("**Delivery Metrics**")
                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("Duration", f"{metrics['duration']}s")
                        d2.metric("Speech Rate", f"{metrics['speech_rate']} WPM")
                        d3.metric("Pause Ratio", f"{metrics['pause_ratio']:.0%}")
                        d4.metric("Pronunciation", f"{metrics['pronunciation_confidence']:.0%}")
                        show_pitch_chart(tmp_path)

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Mock Test mode
# ---------------------------------------------------------------------------

def render_mock_test_mode():
    st.header("Mock Test")

    provider = get_provider()
    provider_ready = is_provider_configured()
    if not provider_ready:
        if provider == "ollama":
            st.error(
                "**Ollama is not reachable.** Make sure it's running:\n\n"
                "```\nollama serve\n```"
            )
        else:
            st.error(
                "**GEMINI_API_KEY not found.** Add it to your `.env` file:\n\n"
                "```\nGEMINI_API_KEY=your-key-here\n```"
            )

    state: MockTestState | None = st.session_state.get("mock_test")

    # Start screen
    if state is None or state.completed:
        st.markdown(
            "A full IELTS Speaking simulation:\n"
            "- **Part 1** — Introduction & Interview (8-10 questions)\n"
            "- **Part 2** — Long Turn (1-min prep + 2-min answer)\n"
            "- **Part 3** — Discussion (4-5 questions)\n"
        )
        if st.button("Start Mock Test", type="primary"):
            all_questions = _load_all()
            plan = assemble_mock_test(all_questions)
            st.session_state["mock_test"] = MockTestState(
                plan=plan, started=True
            )
            st.session_state["mock_test_session_id"] = create_session("mock_test")
            st.rerun()
        return

    plan = state.plan
    total_questions = (
        len(plan.part1_questions)
        + (1 if plan.part2_cue_card else 0)
        + len(plan.part3_questions)
    )
    answered = len(state.responses)

    # Progress bar
    st.progress(answered / total_questions if total_questions > 0 else 0)
    st.caption(f"Question {answered + 1} of {total_questions}")

    # Determine current question
    p1_count = len(plan.part1_questions)
    p2_count = 1 if plan.part2_cue_card else 0
    p3_count = len(plan.part3_questions)

    if answered < p1_count:
        # Part 1
        current_part = 1
        current_qwa = plan.part1_questions[answered]
        st.subheader(f"Part 1 — {current_qwa.question.topic}")
        st.markdown(f"**{current_qwa.question.text}**")
    elif answered < p1_count + p2_count:
        # Part 2
        current_part = 2
        current_qwa = plan.part2_cue_card
        st.subheader(f"Part 2 — Long Turn")
        st.markdown(f"**{current_qwa.question.text}**")
        if current_qwa.question.cue_card:
            st.info(current_qwa.question.cue_card)

        # Prep timer
        prep_key = "mock_test_prep_done"
        if not st.session_state.get(prep_key):
            st.warning("You have **1 minute** to prepare. Click below when ready.")
            if st.button("Start Preparation Timer"):
                st.session_state["mock_test_prep_start"] = time.time()

            prep_start = st.session_state.get("mock_test_prep_start")
            if prep_start:
                elapsed = time.time() - prep_start
                remaining = max(0, 60 - elapsed)
                if remaining > 0:
                    st.info(f"Preparation time remaining: **{int(remaining)}s**")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state[prep_key] = True
                    st.rerun()
            return
    elif answered < p1_count + p2_count + p3_count:
        # Part 3
        current_part = 3
        idx = answered - p1_count - p2_count
        current_qwa = plan.part3_questions[idx]
        st.subheader(f"Part 3 — {current_qwa.question.topic}")
        st.markdown(f"**{current_qwa.question.text}**")
    else:
        # All done — show results
        _render_mock_test_results(state)
        return

    # Recording
    audio = st.audio_input("Record your answer", key=f"mock_audio_{answered}")

    if st.button("Submit & Next", type="primary", key=f"mock_submit_{answered}"):
        if audio is None:
            st.warning("Please record your answer first.")
        elif not provider_ready:
            st.error("Cannot evaluate — AI provider is not configured.")
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Transcribing..."):
                    transcript, words, _ = transcribe_audio(model, tmp_path)

                if not transcript.strip():
                    st.error("No speech detected. Please try again.")
                    return

                with st.spinner("Analyzing..."):
                    metrics = analyze_audio(tmp_path, transcript, words)

                content_eval = None
                combined = None
                try:
                    with st.spinner("Evaluating..."):
                        content_eval = evaluate_answer_enhanced(
                            question=current_qwa.question.text,
                            part=current_part,
                            transcript=transcript,
                            band9_answer=current_qwa.band9_answer,
                        )
                    combined = compute_combined_band(content_eval, metrics)
                except Exception as e:
                    st.warning(f"Evaluation failed ({provider}): {e}")

                # Save to database
                session_id = st.session_state.get("mock_test_session_id", 0)
                if combined:
                    save_attempt_from_eval(
                        session_id=session_id,
                        question_text=current_qwa.question.text,
                        part=current_part,
                        topic=current_qwa.question.topic,
                        transcript=transcript,
                        metrics=metrics,
                        combined=combined,
                        evaluation=content_eval,
                    )

                # Record response in state
                state.responses.append(MockTestResponse(
                    question=current_qwa,
                    transcript=transcript,
                    audio_metrics=metrics,
                    evaluation=content_eval,
                    combined_band=combined or {},
                ))

                # Advance
                if answered + 1 >= total_questions:
                    state.completed = True
                    # Clean up prep timer state
                    st.session_state.pop("mock_test_prep_done", None)
                    st.session_state.pop("mock_test_prep_start", None)

                st.rerun()

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


def _render_mock_test_results(state: MockTestState):
    """Render the results screen after completing a mock test."""
    st.subheader("Mock Test Complete")

    responses = state.responses
    if not responses:
        st.info("No responses recorded.")
        if st.button("Start New Mock Test"):
            st.session_state["mock_test"] = None
            st.rerun()
        return

    # Overall averages
    bands = [r.combined_band.get("overall_band", 0) for r in responses if r.combined_band]
    if bands:
        avg_band = round(sum(bands) / len(bands) * 2) / 2
        st.metric("Overall Band (average)", f"{avg_band}")

    # Per-criterion averages
    criteria = ["fluency_coherence", "lexical_resource", "grammatical_range", "pronunciation"]
    labels = ["Fluency & Coherence", "Lexical Resource", "Grammar", "Pronunciation"]
    avgs = {}
    for key in criteria:
        vals = [r.combined_band.get(key, 0) for r in responses if r.combined_band]
        avgs[key] = round(sum(vals) / len(vals) * 2) / 2 if vals else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(labels[0], f"{avgs[criteria[0]]}")
    c2.metric(labels[1], f"{avgs[criteria[1]]}")
    c3.metric(labels[2], f"{avgs[criteria[2]]}")
    c4.metric(labels[3], f"{avgs[criteria[3]]}")

    st.divider()

    # Per-question expandable reviews
    for i, resp in enumerate(responses):
        q = resp.question.question
        band_str = f" — Band {resp.combined_band.get('overall_band', '?')}" if resp.combined_band else ""
        with st.expander(f"Q{i+1}: Part {q.part} — {q.text[:60]}...{band_str}"):
            if resp.combined_band and resp.evaluation:
                render_review(
                    combined=resp.combined_band,
                    evaluation=resp.evaluation,
                    metrics=resp.audio_metrics,
                    transcript=resp.transcript,
                    band9_answer=resp.question.band9_answer,
                )
            elif resp.transcript:
                st.write(resp.transcript)

    st.divider()
    if st.button("Start New Mock Test"):
        st.session_state["mock_test"] = None
        st.rerun()


# ---------------------------------------------------------------------------
# History mode
# ---------------------------------------------------------------------------

def render_history_mode():
    st.header("History")

    tab1, tab2, tab3 = st.tabs(["Band Trend", "Criterion Breakdown", "Sessions"])

    with tab1:
        data = get_band_trend(limit=50)
        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.line_chart(df, x="timestamp", y="overall_band", y_label="Band Score")
        else:
            st.info("No data yet. Complete some practice sessions to see trends.")

    with tab2:
        data = get_criterion_trends(limit=50)
        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.rename(columns={
                "fluency_coherence": "Fluency & Coherence",
                "lexical_resource": "Lexical Resource",
                "grammatical_range": "Grammar",
                "pronunciation": "Pronunciation",
            })
            st.line_chart(
                df,
                x="timestamp",
                y=["Fluency & Coherence", "Lexical Resource", "Grammar", "Pronunciation"],
                y_label="Band Score",
            )

            weak = get_weak_areas()
            if weak:
                weakest = min(weak, key=weak.get)
                st.warning(
                    f"**Focus area:** {weakest} (avg {weak[weakest]}) — "
                    "This is your lowest-scoring criterion over recent attempts."
                )
        else:
            st.info("No data yet.")

    with tab3:
        sessions = get_recent_sessions(limit=20)
        if not sessions:
            st.info("No sessions recorded yet.")
            return

        for sess in sessions:
            label = (
                f"{sess['mode'].title()} — Band {sess['overall_band']} "
                f"({sess['attempt_count']} questions) — {sess['timestamp'][:16]}"
            )
            with st.expander(label):
                attempts = get_attempts_for_session(sess["id"])
                for att in attempts:
                    st.markdown(
                        f"**Part {att['part']}** — {att['question_text'][:80]} "
                        f"— Band **{att['overall_band']}**"
                    )
                    render_review_from_dict(att)
                    st.divider()


# ---------------------------------------------------------------------------
# Main routing
# ---------------------------------------------------------------------------

if mode == "Transcribe":
    render_transcribe_mode()
elif mode == "Practice":
    render_practice_mode()
elif mode == "Interview":
    render_interview_mode()
elif mode == "Mock Test":
    render_mock_test_mode()
elif mode == "History":
    render_history_mode()
