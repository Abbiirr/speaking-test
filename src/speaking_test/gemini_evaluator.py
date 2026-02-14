"""Gemini provider — IELTS answer evaluation with structured scoring."""

from __future__ import annotations

import logging
import os
import re

from google import genai

logger = logging.getLogger(__name__)

from speaking_test.models import (
    CriterionScore,
    EnhancedReview,
    ContentEvaluation,
    WritingEvaluation,
    WritingEnhancedReview,
)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an experienced IELTS speaking examiner. Evaluate the candidate's spoken \
answer on its OWN merits — judge the ideas, vocabulary, grammar, and coherence \
that the candidate actually produced. Do NOT assess pronunciation or fluency — \
those are measured separately by audio analysis.

The candidate's transcript is your PRIMARY input. Quote the candidate's actual words \
when giving feedback. Never invent errors — only cite what the candidate actually said.

Score against official IELTS Band 9 descriptors:

Coherence (Band 9): Speaks fluently with only rare repetition or self-correction. \
Hesitations are content-related, not language-searching. Topics are fully developed. \
Look for: signposting ("There are two reasons…", "On the other hand…"), logical flow.

Lexical Resource (Band 9): Flexible and precise vocabulary across topics. \
Idiomatic language is natural and accurate, not forced. \
Look for: collocations ("heavy traffic" not "big traffic"), precise word choice, \
topic-specific vocabulary. Flag basic/vague words the candidate should upgrade.

Grammatical Range (Band 9): Full range of structures used naturally. \
Consistent accuracy with only occasional slips. \
Look for: relative clauses, contrast (although/whereas), conditionals. \
Quote each grammar error from the transcript with the corrected version.

Task Response: Directly addresses the question with developed ideas and examples. \
Part 1: Should be 15-25s — direct answer + reason + micro-example. \
Part 3: Should be 35-60s — opinion → reasons → example → counterpoint → summary.

Score each criterion on the IELTS 0-9 band scale (use 0.5 increments). Be fair but \
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

The candidate's transcript is your PRIMARY input. Quote the candidate's actual words \
when giving feedback. Highlight specific errors with the exact phrase from the transcript. \
Never invent errors — only cite what the candidate actually said.

Score against official IELTS Band 9 descriptors:

Coherence (Band 9): Speaks fluently with only rare repetition or self-correction. \
Hesitations are content-related, not language-searching. Topics are fully developed. \
Look for: signposting ("There are two reasons…", "On the other hand…"), logical flow.

Lexical Resource (Band 9): Flexible and precise vocabulary across topics. \
Idiomatic language is natural and accurate, not forced. \
Look for: collocations ("heavy traffic" not "big traffic"), precise word choice, \
topic-specific vocabulary. Flag basic/vague words the candidate should upgrade.

Grammatical Range (Band 9): Full range of structures used naturally. \
Consistent accuracy with only occasional slips. \
Look for: relative clauses, contrast (although/whereas), conditionals. \
Quote each grammar error from the transcript with the corrected version.

Task Response: Directly addresses the question with developed ideas and examples. \
Part 1: Should be 15-25s — direct answer + reason + micro-example. \
Part 3: Should be 35-60s — opinion → reasons → example → counterpoint → summary.

Score each criterion on the IELTS 0-9 band scale (use 0.5 increments). Be fair but \
rigorous. Base your scores entirely on what the candidate said. Any reference answer \
provided is ONLY for understanding the question's expected scope — ignore its wording, \
structure, and vocabulary when scoring.

In addition to scores, provide:

1. **Grammar corrections**: Quote the EXACT phrase from the transcript that contains \
the error. Show the corrected version. Explain the rule briefly. \
Only cite errors the candidate actually made.

2. **Vocabulary upgrades**: Find basic/common words the candidate ACTUALLY USED in \
their transcript. Suggest 2-3 advanced alternatives with an example sentence.

3. **Pronunciation warnings**: List words from the transcript that are commonly \
mispronounced by non-native speakers, with phonetic guidance (simplified IPA or \
syllable stress). Include a tip on the common mistake.

4. **Strengths**: Quote specific phrases from the transcript that demonstrate \
good language use. Be specific, not generic.

5. **Improvement priorities**: Reference specific moments in the transcript where \
the candidate could improve. Give actionable rewrites.

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

    logger.info("Gemini evaluate_answer: part=%d, transcript_len=%d", part, len(transcript))
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

    logger.debug("Gemini raw response: %s", response.text[:500])
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

    logger.info("Gemini evaluate_answer_enhanced: part=%d, transcript_len=%d", part, len(transcript))
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

    logger.debug("Gemini enhanced raw response: %s", response.text[:500])
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


# ---------------------------------------------------------------------------
# Writing evaluation
# ---------------------------------------------------------------------------

WRITING_SYSTEM_PROMPT = """\
You are an experienced IELTS Writing examiner. Evaluate the candidate's essay \
against the official IELTS Writing band descriptors for the 4 criteria.

Score each criterion on the IELTS 0-9 band scale (use 0.5 increments). Be fair \
but rigorous. Quote specific phrases from the essay when giving feedback.

**Task Achievement / Task Response:**
- Task 1: Does the response describe the key features accurately? Is there an \
overview? Is data selected appropriately? Minimum 150 words.
- Task 2: Does the response address all parts of the task? Is the position clear \
throughout? Are ideas developed and supported with examples? Minimum 250 words.

**Coherence & Cohesion (Band 9):** Skilful paragraphing. A wide range of cohesive \
devices used with full flexibility. Information and ideas presented logically with \
clear progression throughout. Look for: topic sentences, linking words, reference \
chains, paragraph organisation.

**Lexical Resource (Band 9):** Full flexibility and precise use of vocabulary. \
Rare minor errors only as slips. Look for: collocations, academic vocabulary, \
precision, avoidance of repetition. Flag basic/overused words.

**Grammatical Range & Accuracy (Band 9):** Full range of structures used \
accurately and appropriately. Rare minor errors only as slips. Look for: \
complex sentences, passive voice, relative clauses, conditionals, articles, \
prepositions.

Word count penalties: Under minimum = max Band 5 for Task Achievement.
"""

WRITING_ENHANCED_SYSTEM_PROMPT = """\
You are an experienced IELTS Writing examiner giving a detailed review. Evaluate \
the candidate's essay against the official IELTS Writing band descriptors.

Score each criterion on the IELTS 0-9 band scale (use 0.5 increments). Quote \
specific phrases from the essay when giving feedback.

**Task Achievement / Task Response:**
- Task 1: Does the response describe key features? Is there an overview? \
Is data selected appropriately? Minimum 150 words.
- Task 2: Does the response address all parts of the task? Is the position \
clear? Are ideas developed with examples? Minimum 250 words.

**Coherence & Cohesion (Band 9):** Skilful paragraphing. Wide range of cohesive \
devices. Logical progression throughout.

**Lexical Resource (Band 9):** Full flexibility and precise vocabulary. Rare \
minor errors only as slips.

**Grammatical Range & Accuracy (Band 9):** Full range of structures used \
accurately. Rare minor errors only as slips.

In addition to scores, provide:

1. **Grammar corrections**: Quote the EXACT phrase from the essay that contains \
the error. Show the corrected version. Explain the rule briefly.

2. **Vocabulary upgrades**: Find basic/overused words the candidate ACTUALLY USED. \
Suggest 2-3 advanced alternatives with an example sentence.

3. **Paragraph feedback**: For each paragraph, give a 1-2 sentence analysis of \
its effectiveness: topic sentence, development, cohesion.

4. **Strengths**: Quote specific phrases that demonstrate good writing. Be specific.

5. **Improvement priorities**: Reference specific parts of the essay where the \
candidate could improve. Give actionable rewrites.

Word count penalties: Under minimum = max Band 5 for Task Achievement.
"""


def evaluate_writing(
    client: genai.Client,
    model: str,
    prompt_text: str,
    essay_text: str,
    task_type: int,
    task1_data_json: str | None = None,
) -> WritingEvaluation:
    """Evaluate a writing essay via Gemini."""
    task_label = "Task 1" if task_type == 1 else "Task 2"
    min_words = 150 if task_type == 1 else 250
    word_count = len(essay_text.split())

    user_prompt = f"""## IELTS Writing {task_label}

**Question/Prompt:**
{prompt_text}

**Candidate's Essay ({word_count} words, minimum {min_words}):**
{essay_text}
"""
    if task1_data_json:
        user_prompt += f"\n**Chart Data (JSON):**\n{task1_data_json}\n"

    logger.info("Gemini evaluate_writing: task=%d, word_count=%d", task_type, word_count)
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=WRITING_SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=WritingEvaluation,
        ),
    )
    logger.debug("Gemini writing raw response: %s", response.text[:500])
    return WritingEvaluation.model_validate_json(response.text)


def evaluate_writing_enhanced(
    client: genai.Client,
    model: str,
    prompt_text: str,
    essay_text: str,
    task_type: int,
    task1_data_json: str | None = None,
) -> WritingEnhancedReview:
    """Evaluate writing with richer feedback: corrections, upgrades, paragraph analysis."""
    task_label = "Task 1" if task_type == 1 else "Task 2"
    min_words = 150 if task_type == 1 else 250
    word_count = len(essay_text.split())

    user_prompt = f"""## IELTS Writing {task_label}

**Question/Prompt:**
{prompt_text}

**Candidate's Essay ({word_count} words, minimum {min_words}):**
{essay_text}
"""
    if task1_data_json:
        user_prompt += f"\n**Chart Data (JSON):**\n{task1_data_json}\n"

    logger.info("Gemini evaluate_writing_enhanced: task=%d, word_count=%d", task_type, word_count)
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=WRITING_ENHANCED_SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=WritingEnhancedReview,
        ),
    )
    logger.debug("Gemini writing enhanced raw response: %s", response.text[:500])
    return WritingEnhancedReview.model_validate_json(response.text)


def compute_writing_band(
    eval_result: WritingEvaluation | WritingEnhancedReview,
) -> float:
    """Average of 4 writing criteria, rounded to nearest 0.5."""
    raw = (
        eval_result.task_achievement.score
        + eval_result.coherence.score
        + eval_result.lexical_resource.score
        + eval_result.grammatical_range.score
    ) / 4
    return round(raw * 2) / 2


def writing_quality_checks(essay_text: str, task_type: int) -> dict:
    """Pre-LLM validation: word count, empty check."""
    word_count = len(essay_text.split())
    min_words = 150 if task_type == 1 else 250
    return {
        "word_count": word_count,
        "min_words": min_words,
        "meets_minimum": word_count >= min_words,
        "is_empty": word_count == 0,
    }
