"""Enhanced review renderer with progressive disclosure."""

from __future__ import annotations

import streamlit as st

from speaking_test.models import EnhancedReview, ContentEvaluation


def render_review(
    combined: dict,
    evaluation: ContentEvaluation | EnhancedReview | None,
    metrics: dict,
    transcript: str = "",
    band9_answer: str = "",
    pitch_chart_fn=None,
    audio_path: str = "",
) -> None:
    """Render a full review with progressive disclosure.

    Always visible:
      1. Overall band + 4 criterion scores
      2. Examiner summary
      3. Strengths (if enhanced)
      4. Improvement priorities (if enhanced)

    Expandable:
      5. Grammar Corrections
      6. Vocabulary Upgrades
      7. Detailed Criterion Breakdown
      8. Transcript
      9. Band 9 Reference Answer
      10. Delivery Metrics + Pitch Chart
    """
    # 1. Band scores — compact metric row
    st.metric("Overall Band Score", f"{combined['overall_band']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fluency & Coherence", f"{combined['fluency_coherence']}")
    c2.metric("Lexical Resource", f"{combined['lexical_resource']}")
    c3.metric("Grammar", f"{combined['grammatical_range']}")
    c4.metric("Pronunciation", f"{combined['pronunciation']}")

    if not evaluation:
        return

    # 2. Examiner summary
    st.markdown(f"*{evaluation.overall_feedback}*")

    # 3 & 4. Strengths + Improvement priorities (enhanced only)
    is_enhanced = isinstance(evaluation, EnhancedReview)

    if is_enhanced and evaluation.strengths:
        st.markdown("**Strengths**")
        for s in evaluation.strengths:
            st.markdown(f"- {s}")

    if is_enhanced and evaluation.improvement_priorities:
        st.markdown("**Priorities**")
        for tip in evaluation.improvement_priorities:
            st.markdown(f"- {tip}")

    # 5. Grammar Corrections (expandable)
    if is_enhanced and evaluation.grammar_corrections:
        with st.expander(f"Grammar Corrections ({len(evaluation.grammar_corrections)})"):
            for gc in evaluation.grammar_corrections:
                st.markdown(
                    f"~~{gc.original}~~ &rarr; **{gc.corrected}**\n\n"
                    f"*{gc.explanation}*"
                )
                st.divider()

    # 6. Vocabulary Upgrades (expandable)
    if is_enhanced and evaluation.vocabulary_upgrades:
        with st.expander(f"Vocabulary Upgrades ({len(evaluation.vocabulary_upgrades)})"):
            for vu in evaluation.vocabulary_upgrades:
                alts = ", ".join(f"**{a}**" for a in vu.alternatives)
                st.markdown(
                    f"*{vu.basic_word}* &rarr; {alts}\n\n"
                    f"Example: *{vu.example}*"
                )
                st.divider()

    # 7. Detailed Criterion Breakdown
    with st.expander("Detailed Criterion Breakdown"):
        st.markdown(
            f"**Coherence ({evaluation.coherence.score}):** "
            f"{evaluation.coherence.feedback}"
        )
        st.markdown(
            f"**Lexical Resource ({evaluation.lexical_resource.score}):** "
            f"{evaluation.lexical_resource.feedback}"
        )
        st.markdown(
            f"**Grammar ({evaluation.grammatical_range.score}):** "
            f"{evaluation.grammatical_range.feedback}"
        )
        st.markdown(
            f"**Task Response ({evaluation.task_response.score}):** "
            f"{evaluation.task_response.feedback}"
        )

    # 8. Transcript
    if transcript:
        with st.expander("Transcript"):
            st.write(transcript)

    # 9. Band 9 Reference Answer
    if band9_answer:
        with st.expander("Reference Answer"):
            st.markdown(band9_answer)

    # 10. Delivery Metrics + Pitch Chart
    with st.expander("Delivery Metrics"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Duration", f"{metrics.get('duration', 0)}s")
        d2.metric("Speech Rate", f"{metrics.get('speech_rate', 0)} WPM")
        d3.metric("Pause Ratio", f"{metrics.get('pause_ratio', 0):.0%}")
        d4.metric("Pronunciation Conf.", f"{metrics.get('pronunciation_confidence', 0):.0%}")

        long_pauses = metrics.get("long_pauses", 0)
        if long_pauses > 0:
            st.caption(f"Long pauses (>2s): {long_pauses}")

        if pitch_chart_fn and audio_path:
            pitch_chart_fn(audio_path)


def render_review_from_dict(attempt: dict) -> None:
    """Render a review from a database attempt dict (for History mode)."""
    # Band scores
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fluency", f"{attempt.get('fluency_coherence', 0)}")
    c2.metric("Lexical", f"{attempt.get('lexical_resource', 0)}")
    c3.metric("Grammar", f"{attempt.get('grammatical_range', 0)}")
    c4.metric("Pronunciation", f"{attempt.get('pronunciation', 0)}")

    if attempt.get("examiner_feedback"):
        st.markdown(f"*{attempt['examiner_feedback']}*")

    # Grammar corrections from JSON
    gc = attempt.get("grammar_corrections")
    if isinstance(gc, list) and gc:
        with st.expander(f"Grammar Corrections ({len(gc)})"):
            for item in gc:
                if isinstance(item, dict):
                    st.markdown(
                        f"~~{item.get('original', '')}~~ &rarr; "
                        f"**{item.get('corrected', '')}**\n\n"
                        f"*{item.get('explanation', '')}*"
                    )
                    st.divider()

    # Vocabulary upgrades from JSON
    vu = attempt.get("vocabulary_upgrades")
    if isinstance(vu, list) and vu:
        with st.expander(f"Vocabulary Upgrades ({len(vu)})"):
            for item in vu:
                if isinstance(item, dict):
                    alts = ", ".join(f"**{a}**" for a in item.get("alternatives", []))
                    st.markdown(
                        f"*{item.get('basic_word', '')}* &rarr; {alts}\n\n"
                        f"Example: *{item.get('example', '')}*"
                    )
                    st.divider()

    # Improvement tips from JSON
    tips = attempt.get("improvement_tips")
    if isinstance(tips, list) and tips:
        st.markdown("**Improvement Tips**")
        for tip in tips:
            st.markdown(f"- {tip}")

    if attempt.get("transcript"):
        with st.expander("Transcript"):
            st.write(attempt["transcript"])
