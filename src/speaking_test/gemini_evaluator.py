"""Gemini provider — IELTS answer evaluation with structured scoring."""

from __future__ import annotations

import os
import re

from google import genai

from speaking_test.models import (
    CriterionScore,
    EnhancedReview,
    ContentEvaluation,
)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an experienced IELTS speaking examiner. Evaluate the candidate's spoken \
answer on its OWN merits — judge the ideas, vocabulary, grammar, and coherence \
that the candidate actually produced. Do NOT assess pronunciation or fluency — \
those are measured separately by audio analysis.

Score each criterion on the IELTS 0–9 band scale (use 0.5 increments). Be fair but \
rigorous. Base your scores entirely on what the candidate said, not on how closely \
it matches any reference answer. Any reference answer provided is ONLY for understanding \
the question's expected scope — ignore its wording, structure, and vocabulary when scoring.

Return your evaluation as a JSON object with this exact structure:
{
  "coherence": {"score": <float>, "feedback": "<string>"},
  "lexical_resource": {"score": <float>, "feedback": "<string>"},
  "grammatical_range": {"score": <float>, "feedback": "<string>"},
  "task_response": {"score": <float>, "feedback": "<string>"},
  "overall_feedback": "<string>"
}
"""

ENHANCED_SYSTEM_PROMPT = """\
You are an experienced IELTS speaking examiner giving a detailed review. Evaluate \
the candidate's spoken answer on its OWN merits — judge the ideas, vocabulary, \
grammar, and coherence that the candidate actually produced. Do NOT assess \
pronunciation or fluency — those are measured separately.

Score each criterion on the IELTS 0–9 band scale (use 0.5 increments). Be fair but \
rigorous. Base your scores entirely on what the candidate said. Any reference answer \
provided is ONLY for understanding the question's expected scope — ignore its wording, \
structure, and vocabulary when scoring.

In addition to scores, provide:
1. **Grammar corrections**: Find specific grammar errors in the candidate's transcript. \
For each, give the original phrase, the corrected version, and a brief rule explanation. \
Only cite errors the candidate actually made.
2. **Vocabulary upgrades**: Identify 2-4 basic/common words the candidate actually used \
and suggest advanced alternatives with an example sentence.
3. **Strengths**: List 2-3 specific things the candidate did well (not generic praise). \
Quote or reference specific parts of their answer.
4. **Improvement priorities**: List 2-3 specific, actionable tips based on weaknesses \
you observed in this answer (not generic advice like "practice more").

Return a JSON object matching the schema provided.
"""


def create_gemini_client() -> genai.Client:
    """Create a Gemini client using the API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment. "
            "Add it to your .env file: GEMINI_API_KEY=your-key-here"
        )
    return genai.Client(api_key=api_key)


def get_model_name() -> str:
    """Get the Gemini model name from environment or use default."""
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")


# ---------------------------------------------------------------------------
# Standard evaluation (unchanged)
# ---------------------------------------------------------------------------

def evaluate_answer(
    client: genai.Client,
    model: str,
    question: str,
    part: int,
    transcript: str,
    band9_answer: str = "",
) -> ContentEvaluation:
    """Send a candidate's transcript for IELTS content evaluation via Gemini."""
    user_prompt = f"""## IELTS Speaking Part {part}

**Question:** {question}

**Candidate's Answer (transcribed from speech):**
{transcript}
"""
    if band9_answer:
        user_prompt += f"""
**Reference Answer (for question scope only — do NOT compare or score against this):**
{band9_answer}
"""

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=ContentEvaluation,
        ),
    )

    return ContentEvaluation.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Enhanced evaluation (new)
# ---------------------------------------------------------------------------

def evaluate_answer_enhanced(
    client: genai.Client,
    model: str,
    question: str,
    part: int,
    transcript: str,
    band9_answer: str = "",
) -> EnhancedReview:
    """Evaluate with richer feedback: grammar corrections, vocab upgrades, etc."""
    user_prompt = f"""## IELTS Speaking Part {part}

**Question:** {question}

**Candidate's Answer (transcribed from speech):**
{transcript}
"""
    if band9_answer:
        user_prompt += f"""
**Reference Answer (for question scope only — do NOT compare or score against this):**
{band9_answer}
"""

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=ENHANCED_SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=EnhancedReview,
        ),
    )

    return EnhancedReview.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Combined band computation (unchanged)
# ---------------------------------------------------------------------------

def compute_combined_band(
    content_eval: ContentEvaluation | EnhancedReview,
    audio_metrics: dict,
) -> dict:
    """Blend AI content scores with audio delivery metrics.

    IELTS has 4 equally weighted criteria (25% each):
    - Fluency & Coherence: 50% audio fluency + 50% AI coherence
    - Lexical Resource: 100% AI content evaluation
    - Grammatical Range & Accuracy: 100% AI content evaluation
    - Pronunciation: 100% audio metrics (Whisper confidence)
    """
    # Audio-based fluency score
    wpm = audio_metrics.get("speech_rate", 0)
    if 120 <= wpm <= 160:
        rate_score = 9.0
    elif 100 <= wpm < 120 or 160 < wpm <= 180:
        rate_score = 7.0
    elif 80 <= wpm < 100 or 180 < wpm <= 200:
        rate_score = 5.5
    else:
        rate_score = 4.0

    pause = audio_metrics.get("pause_ratio", 1.0)
    if pause < 0.15:
        pause_score = 9.0
    elif pause < 0.25:
        pause_score = 7.0
    elif pause < 0.40:
        pause_score = 5.5
    else:
        pause_score = 4.0

    audio_fluency = (rate_score + pause_score) / 2

    # Pronunciation from Whisper confidence
    conf = audio_metrics.get("pronunciation_confidence", 0.0)
    pronunciation_band = min(9.0, max(4.0, conf * 10.0))

    # Combine criteria
    fluency_coherence = 0.5 * audio_fluency + 0.5 * content_eval.coherence.score
    lexical = content_eval.lexical_resource.score
    grammar = content_eval.grammatical_range.score
    pronunciation = pronunciation_band

    # Overall band: equal 25% weight
    raw = (fluency_coherence + lexical + grammar + pronunciation) / 4
    overall = round(raw * 2) / 2  # Round to nearest 0.5
    overall = max(4.0, min(9.0, overall))

    return {
        "overall_band": overall,
        "fluency_coherence": round(fluency_coherence * 2) / 2,
        "lexical_resource": round(lexical * 2) / 2,
        "grammatical_range": round(grammar * 2) / 2,
        "pronunciation": round(pronunciation * 2) / 2,
    }


# ---------------------------------------------------------------------------
# Filler detection (no API call)
# ---------------------------------------------------------------------------

FILLER_PATTERNS = [
    r"\bum+\b",
    r"\buh+\b",
    r"\berm+\b",
    r"\blike\b",
    r"\byou know\b",
    r"\bi mean\b",
    r"\bbasically\b",
    r"\bactually\b",
    r"\bliterally\b",
    r"\bso+\b(?=\s+(?:yeah|like|um|uh))",
    r"\bkind of\b",
    r"\bsort of\b",
]


def detect_fillers(transcript: str) -> dict[str, int]:
    """Count filler words/phrases in a transcript. No API call needed."""
    text = transcript.lower()
    counts: dict[str, int] = {}
    for pattern in FILLER_PATTERNS:
        label = pattern.replace(r"\b", "").replace("+", "").replace(r"\s+", " ")
        label = re.sub(r"[^a-z ]", "", label).strip()
        matches = re.findall(pattern, text)
        if matches:
            counts[label] = len(matches)
    return counts
