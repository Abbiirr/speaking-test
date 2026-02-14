"""Enhanced review renderer with progressive disclosure."""

from __future__ import annotations

import streamlit as st

from speaking_test.models import (
    EnhancedReview,
    ContentEvaluation,
    WritingEvaluation,
    WritingEnhancedReview,
)


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

    # 6b. Pronunciation Warnings (expandable)
    if is_enhanced and evaluation.pronunciation_warnings:
        with st.expander(f"Pronunciation Warnings ({len(evaluation.pronunciation_warnings)})"):
            for pw in evaluation.pronunciation_warnings:
                st.markdown(
                    f"**{pw.word}** — /{pw.phonetic}/\n\n"
                    f"*{pw.tip}*"
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

    # Strengths from JSON
    strengths = attempt.get("strengths")
    if isinstance(strengths, list) and strengths:
        st.markdown("**Strengths**")
        for s in strengths:
            st.markdown(f"- {s}")

    # Improvement tips from JSON
    tips = attempt.get("improvement_tips")
    if isinstance(tips, list) and tips:
        st.markdown("**Improvement Tips**")
        for tip in tips:
            st.markdown(f"- {tip}")

    # Pronunciation warnings from JSON
    pw = attempt.get("pronunciation_warnings")
    if isinstance(pw, list) and pw:
        with st.expander(f"Pronunciation Warnings ({len(pw)})"):
            for item in pw:
                if isinstance(item, dict):
                    st.markdown(
                        f"**{item.get('word', '')}** — /{item.get('phonetic', '')}/\n\n"
                        f"*{item.get('tip', '')}*"
                    )
                    st.divider()

    if attempt.get("transcript"):
        with st.expander("Transcript"):
            st.write(attempt["transcript"])

    if attempt.get("band9_answer"):
        with st.expander("Reference Answer"):
            st.markdown(attempt["band9_answer"])


# ---------------------------------------------------------------------------
# Writing review rendering
# ---------------------------------------------------------------------------

def render_writing_review(
    eval_result: WritingEvaluation | WritingEnhancedReview,
    word_count: int,
    task_type: int,
) -> None:
    """Render a writing evaluation with progressive disclosure."""
    from speaking_test.gemini_evaluator import compute_writing_band

    overall = compute_writing_band(eval_result)
    min_words = 150 if task_type == 1 else 250

    # 1. Band scores
    st.metric("Overall Band Score", f"{overall}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task Achievement", f"{eval_result.task_achievement.score}")
    c2.metric("Coherence", f"{eval_result.coherence.score}")
    c3.metric("Lexical Resource", f"{eval_result.lexical_resource.score}")
    c4.metric("Grammar", f"{eval_result.grammatical_range.score}")

    # 2. Word count indicator
    if word_count >= min_words:
        st.success(f"Word count: **{word_count}** (minimum: {min_words})")
    else:
        st.error(
            f"Word count: **{word_count}** — below minimum of {min_words}. "
            "This will cap Task Achievement at Band 5."
        )

    # 3. Examiner summary
    st.markdown(f"*{eval_result.overall_feedback}*")

    is_enhanced = isinstance(eval_result, WritingEnhancedReview)

    # 4. Strengths
    if is_enhanced and eval_result.strengths:
        st.markdown("**Strengths**")
        for s in eval_result.strengths:
            st.markdown(f"- {s}")

    # 5. Improvement priorities
    if is_enhanced and eval_result.improvement_priorities:
        st.markdown("**Priorities**")
        for tip in eval_result.improvement_priorities:
            st.markdown(f"- {tip}")

    # 6. Paragraph feedback
    if is_enhanced and eval_result.paragraph_feedback:
        with st.expander(f"Paragraph Analysis ({len(eval_result.paragraph_feedback)})"):
            for i, pf in enumerate(eval_result.paragraph_feedback, 1):
                st.markdown(f"**Paragraph {i}:** {pf}")

    # 7. Grammar corrections
    if is_enhanced and eval_result.grammar_corrections:
        with st.expander(f"Grammar Corrections ({len(eval_result.grammar_corrections)})"):
            for gc in eval_result.grammar_corrections:
                st.markdown(
                    f"~~{gc.original}~~ &rarr; **{gc.corrected}**\n\n"
                    f"*{gc.explanation}*"
                )
                st.divider()

    # 8. Vocabulary upgrades
    if is_enhanced and eval_result.vocabulary_upgrades:
        with st.expander(f"Vocabulary Upgrades ({len(eval_result.vocabulary_upgrades)})"):
            for vu in eval_result.vocabulary_upgrades:
                alts = ", ".join(f"**{a}**" for a in vu.alternatives)
                st.markdown(
                    f"*{vu.basic_word}* &rarr; {alts}\n\n"
                    f"Example: *{vu.example}*"
                )
                st.divider()

    # 9. Detailed criterion breakdown
    with st.expander("Detailed Criterion Breakdown"):
        st.markdown(
            f"**Task Achievement ({eval_result.task_achievement.score}):** "
            f"{eval_result.task_achievement.feedback}"
        )
        st.markdown(
            f"**Coherence & Cohesion ({eval_result.coherence.score}):** "
            f"{eval_result.coherence.feedback}"
        )
        st.markdown(
            f"**Lexical Resource ({eval_result.lexical_resource.score}):** "
            f"{eval_result.lexical_resource.feedback}"
        )
        st.markdown(
            f"**Grammar ({eval_result.grammatical_range.score}):** "
            f"{eval_result.grammatical_range.feedback}"
        )


def render_writing_review_from_dict(attempt: dict) -> None:
    """Render a writing review from a database attempt dict (History mode)."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task", f"{attempt.get('task_score', 0)}")
    c2.metric("Coherence", f"{attempt.get('coherence_score', 0)}")
    c3.metric("Lexical", f"{attempt.get('lexical_score', 0)}")
    c4.metric("Grammar", f"{attempt.get('grammar_score', 0)}")

    if attempt.get("examiner_feedback"):
        st.markdown(f"*{attempt['examiner_feedback']}*")

    # Paragraph feedback
    pf = attempt.get("paragraph_feedback")
    if isinstance(pf, list) and pf:
        with st.expander(f"Paragraph Analysis ({len(pf)})"):
            for i, p in enumerate(pf, 1):
                st.markdown(f"**Paragraph {i}:** {p}")

    # Grammar corrections
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

    # Vocabulary upgrades
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

    # Improvement tips
    tips = attempt.get("improvement_tips")
    if isinstance(tips, list) and tips:
        st.markdown("**Improvement Tips**")
        for tip in tips:
            st.markdown(f"- {tip}")
