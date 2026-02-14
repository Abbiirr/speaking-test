"""Ollama-based IELTS answer evaluation with structured scoring."""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

from speaking_test.gemini_evaluator import (
    ENHANCED_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    WRITING_SYSTEM_PROMPT,
    WRITING_ENHANCED_SYSTEM_PROMPT,
)
from speaking_test.models import (
    EnhancedReview,
    ContentEvaluation,
    WritingEvaluation,
    WritingEnhancedReview,
)

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")

# Flat JSON schemas — small models handle flat keys better than nested objects.
# No concrete example values to avoid the model copying them verbatim.
_EVALUATION_SCHEMA = """\
Return a JSON object with ALL of these keys. Use IELTS 0-9 band scores based on the ACTUAL transcript above.
Score the candidate's OWN answer — do NOT compare it to any reference answer.
Write feedback that references SPECIFIC words and phrases from the candidate's answer.

Required keys:
- "coherence_score": float (0-9, score for coherence and cohesion)
- "coherence_feedback": string (feedback referencing the candidate's actual answer)
- "lexical_resource_score": float (0-9, score for vocabulary range)
- "lexical_resource_feedback": string (feedback about vocabulary used in the answer)
- "grammatical_range_score": float (0-9, score for grammar accuracy)
- "grammatical_range_feedback": string (feedback about grammar in the answer)
- "task_response_score": float (0-9, score for relevance to the question)
- "task_response_feedback": string (feedback about how well the question was answered)
- "overall_feedback": string (2-3 sentence examiner summary of THIS specific answer)"""

_ENHANCED_SCHEMA = """\
Return a JSON object with ALL of these keys. Use IELTS 0-9 band scores based on the ACTUAL transcript above.
Score the candidate's OWN answer — do NOT compare it to any reference answer.
Write feedback that references SPECIFIC words and phrases from the candidate's answer.
Find REAL errors from the transcript — do NOT invent examples.

Required keys:
- "coherence_score": float (0-9)
- "coherence_feedback": string (feedback referencing the candidate's actual answer)
- "lexical_resource_score": float (0-9)
- "lexical_resource_feedback": string (feedback about vocabulary used in the answer)
- "grammatical_range_score": float (0-9)
- "grammatical_range_feedback": string (feedback about grammar in the answer)
- "task_response_score": float (0-9)
- "task_response_feedback": string (feedback about how well the question was answered)
- "overall_feedback": string (2-3 sentence examiner summary of THIS specific answer)
- "grammar_corrections": list of objects with keys "original", "corrected", "explanation" (find REAL errors from the transcript)
- "vocabulary_upgrades": list of objects with keys "basic_word", "alternatives" (list of strings), "example" (find basic words the candidate ACTUALLY used)
- "pronunciation_warnings": list of objects with keys "word", "phonetic", "tip" (words from the transcript commonly mispronounced by non-native speakers)
- "strengths": list of strings (specific things done well in THIS answer)
- "improvement_priorities": list of strings (specific actionable tips for THIS answer)"""


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from deepseek-r1 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> str:
    """Extract JSON from the response, handling markdown code fences."""
    # Try to extract from ```json ... ``` fences first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Otherwise return the stripped text as-is
    return text.strip()


_CRITERION_KEYS = ["coherence", "lexical_resource", "grammatical_range", "task_response"]

# Map common alternative key names produced by smaller models to canonical keys.
# Covers both flat (e.g. "grammar_score") and nested (e.g. "grammar") variants.
_SCORE_ALIASES: dict[str, str] = {
    # coherence
    "coherence_and_cohesion_score": "coherence_score",
    "fluency_and_coherence_score": "coherence_score",
    "fluency_coherence_score": "coherence_score",
    "fluency_score": "coherence_score",
    # lexical_resource
    "vocabulary_score": "lexical_resource_score",
    "lexical_score": "lexical_resource_score",
    "vocab_score": "lexical_resource_score",
    # grammatical_range
    "grammar_score": "grammatical_range_score",
    "grammatical_range_and_accuracy_score": "grammatical_range_score",
    "grammatical_accuracy_score": "grammatical_range_score",
    # task_response
    "task_achievement_score": "task_response_score",
    "relevance_score": "task_response_score",
}

_FEEDBACK_ALIASES: dict[str, str] = {
    "coherence_and_cohesion_feedback": "coherence_feedback",
    "fluency_and_coherence_feedback": "coherence_feedback",
    "vocabulary_feedback": "lexical_resource_feedback",
    "lexical_feedback": "lexical_resource_feedback",
    "grammar_feedback": "grammatical_range_feedback",
    "grammatical_accuracy_feedback": "grammatical_range_feedback",
    "task_achievement_feedback": "task_response_feedback",
    "relevance_feedback": "task_response_feedback",
}

_NESTED_ALIASES: dict[str, str] = {
    "coherence_and_cohesion": "coherence",
    "fluency_and_coherence": "coherence",
    "fluency_coherence": "coherence",
    "fluency": "coherence",
    "vocabulary": "lexical_resource",
    "lexical": "lexical_resource",
    "vocab": "lexical_resource",
    "grammar": "grammatical_range",
    "grammatical_range_and_accuracy": "grammatical_range",
    "grammar_range": "grammatical_range",
    "grammatical_accuracy": "grammatical_range",
    "task_achievement": "task_response",
    "task": "task_response",
    "relevance": "task_response",
    "response_relevance": "task_response",
    "feedback": "overall_feedback",
    "summary": "overall_feedback",
    "overall": "overall_feedback",
    "general_feedback": "overall_feedback",
    "examiner_feedback": "overall_feedback",
}


def _normalize_evaluation(raw: dict) -> dict:
    """Normalize Ollama output into the nested schema expected by Pydantic models.

    Handles three response formats from smaller models:
    1. Flat keys: ``{"coherence_score": 7, "coherence_feedback": "..."}``
       (our preferred prompt format)
    2. Nested objects: ``{"coherence": {"score": 7, "feedback": "..."}}``
    3. Bare values: ``{"coherence": 7}``
    4. Alternative key names: ``{"grammar": 7}`` instead of ``"grammatical_range"``
    """
    # 0. Remap aliased flat keys
    for alias, canonical in _SCORE_ALIASES.items():
        if alias in raw and canonical not in raw:
            raw[canonical] = raw.pop(alias)
    for alias, canonical in _FEEDBACK_ALIASES.items():
        if alias in raw and canonical not in raw:
            raw[canonical] = raw.pop(alias)

    # 1. Remap aliased nested keys
    for alias, canonical in _NESTED_ALIASES.items():
        if alias in raw and canonical not in raw:
            raw[canonical] = raw.pop(alias)

    # 2. Build nested CriterionScore dicts from whatever format we got
    for key in _CRITERION_KEYS:
        score_key = f"{key}_score"
        feedback_key = f"{key}_feedback"
        val = raw.get(key)

        if score_key in raw:
            # Flat format — assemble nested dict from flat keys
            raw[key] = {
                "score": raw.pop(score_key, 0),
                "feedback": raw.pop(feedback_key, ""),
            }
        elif isinstance(val, (int, float)):
            # Bare numeric
            raw[key] = {"score": val, "feedback": ""}
        elif isinstance(val, dict):
            # Already nested — ensure required fields
            if "score" not in val:
                val["score"] = 0
            if "feedback" not in val:
                val["feedback"] = ""
        else:
            raw[key] = {"score": 0, "feedback": ""}

    # 3. Ensure overall_feedback exists as a string
    if "overall_feedback" not in raw or not isinstance(raw.get("overall_feedback"), str):
        raw["overall_feedback"] = ""

    return raw


def _chat(system: str, user: str) -> str:
    """Send a chat request to Ollama and return the response text."""
    url = f"{_OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": _OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.3, "num_gpu": 999},
    }
    resp = httpx.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("message", {}).get("content", "")
    cleaned = _strip_think_tags(raw)
    return _extract_json(cleaned)


def _build_user_prompt(question: str, part: int, transcript: str, band9_answer: str) -> str:
    """Build the user prompt (shared between standard and enhanced evaluation)."""
    prompt = f"""## IELTS Speaking Part {part}

**Question:** {question}

**Candidate's Answer (transcribed from speech):**
{transcript}
"""
    if band9_answer:
        prompt += f"""
**Reference Answer (for question scope only — do NOT compare or score against this):**
{band9_answer}
"""
    return prompt


def is_available() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        resp = httpx.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def evaluate_answer(
    question: str,
    part: int,
    transcript: str,
    band9_answer: str = "",
) -> ContentEvaluation:
    """Send a candidate's transcript to Ollama for IELTS content evaluation."""
    system = SYSTEM_PROMPT + "\n\n" + _EVALUATION_SCHEMA
    user = _build_user_prompt(question, part, transcript, band9_answer)
    raw_json = _chat(system, user)
    logger.info("Ollama raw response: %s", raw_json)
    data = json.loads(raw_json)
    logger.info("Ollama parsed keys: %s", list(data.keys()))
    data = _normalize_evaluation(data)
    logger.info("Normalized data: %s", json.dumps(data, default=str)[:500])
    return ContentEvaluation.model_validate(data)


def evaluate_answer_enhanced(
    question: str,
    part: int,
    transcript: str,
    band9_answer: str = "",
) -> EnhancedReview:
    """Evaluate with richer feedback: grammar corrections, vocab upgrades, etc."""
    system = ENHANCED_SYSTEM_PROMPT + "\n\n" + _ENHANCED_SCHEMA
    user = _build_user_prompt(question, part, transcript, band9_answer)
    raw_json = _chat(system, user)
    logger.info("Ollama enhanced raw response: %s", raw_json)
    data = json.loads(raw_json)
    logger.info("Ollama enhanced parsed keys: %s", list(data.keys()))
    data = _normalize_evaluation(data)
    logger.info("Normalized enhanced data: %s", json.dumps(data, default=str)[:500])
    return EnhancedReview.model_validate(data)


# ---------------------------------------------------------------------------
# Writing evaluation — Ollama
# ---------------------------------------------------------------------------

_WRITING_CRITERION_KEYS = [
    "task_achievement", "coherence", "lexical_resource", "grammatical_range",
]

_WRITING_SCORE_ALIASES: dict[str, str] = {
    "task_achievement_score": "task_achievement_score",
    "task_response_score": "task_achievement_score",
    "task_score": "task_achievement_score",
    "coherence_score": "coherence_score",
    "coherence_and_cohesion_score": "coherence_score",
    "lexical_resource_score": "lexical_resource_score",
    "vocabulary_score": "lexical_resource_score",
    "lexical_score": "lexical_resource_score",
    "grammatical_range_score": "grammatical_range_score",
    "grammar_score": "grammatical_range_score",
    "grammatical_accuracy_score": "grammatical_range_score",
}

_WRITING_FEEDBACK_ALIASES: dict[str, str] = {
    "task_response_feedback": "task_achievement_feedback",
    "task_feedback": "task_achievement_feedback",
    "coherence_and_cohesion_feedback": "coherence_feedback",
    "vocabulary_feedback": "lexical_resource_feedback",
    "lexical_feedback": "lexical_resource_feedback",
    "grammar_feedback": "grammatical_range_feedback",
    "grammatical_accuracy_feedback": "grammatical_range_feedback",
}

_WRITING_NESTED_ALIASES: dict[str, str] = {
    "task_response": "task_achievement",
    "task": "task_achievement",
    "coherence_and_cohesion": "coherence",
    "vocabulary": "lexical_resource",
    "lexical": "lexical_resource",
    "grammar": "grammatical_range",
    "grammatical_accuracy": "grammatical_range",
}

_WRITING_EVALUATION_SCHEMA = """\
Return a JSON object with ALL of these keys. Use IELTS 0-9 band scores.
Score the candidate's OWN essay. Write feedback referencing SPECIFIC phrases.

Required keys:
- "task_achievement_score": float (0-9, Task Achievement / Task Response)
- "task_achievement_feedback": string
- "coherence_score": float (0-9, Coherence & Cohesion)
- "coherence_feedback": string
- "lexical_resource_score": float (0-9, Lexical Resource)
- "lexical_resource_feedback": string
- "grammatical_range_score": float (0-9, Grammatical Range & Accuracy)
- "grammatical_range_feedback": string
- "overall_feedback": string (2-3 sentence examiner summary)"""

_WRITING_ENHANCED_SCHEMA = """\
Return a JSON object with ALL of these keys. Use IELTS 0-9 band scores.
Score the candidate's OWN essay. Find REAL errors — do NOT invent examples.

Required keys:
- "task_achievement_score": float (0-9)
- "task_achievement_feedback": string
- "coherence_score": float (0-9)
- "coherence_feedback": string
- "lexical_resource_score": float (0-9)
- "lexical_resource_feedback": string
- "grammatical_range_score": float (0-9)
- "grammatical_range_feedback": string
- "overall_feedback": string (2-3 sentence examiner summary)
- "grammar_corrections": list of objects with keys "original", "corrected", "explanation"
- "vocabulary_upgrades": list of objects with keys "basic_word", "alternatives" (list), "example"
- "paragraph_feedback": list of strings (1-2 sentence analysis per paragraph)
- "strengths": list of strings
- "improvement_priorities": list of strings"""


def _normalize_writing_evaluation(raw: dict) -> dict:
    """Normalize Ollama output for writing evaluation models."""
    # Remap aliased flat keys
    for alias, canonical in _WRITING_SCORE_ALIASES.items():
        if alias in raw and canonical not in raw:
            raw[canonical] = raw.pop(alias)
    for alias, canonical in _WRITING_FEEDBACK_ALIASES.items():
        if alias in raw and canonical not in raw:
            raw[canonical] = raw.pop(alias)

    # Remap aliased nested keys
    for alias, canonical in _WRITING_NESTED_ALIASES.items():
        if alias in raw and canonical not in raw:
            raw[canonical] = raw.pop(alias)

    # Build nested CriterionScore dicts
    for key in _WRITING_CRITERION_KEYS:
        score_key = f"{key}_score"
        feedback_key = f"{key}_feedback"
        val = raw.get(key)

        if score_key in raw:
            raw[key] = {
                "score": raw.pop(score_key, 0),
                "feedback": raw.pop(feedback_key, ""),
            }
        elif isinstance(val, (int, float)):
            raw[key] = {"score": val, "feedback": ""}
        elif isinstance(val, dict):
            if "score" not in val:
                val["score"] = 0
            if "feedback" not in val:
                val["feedback"] = ""
        else:
            raw[key] = {"score": 0, "feedback": ""}

    if "overall_feedback" not in raw or not isinstance(raw.get("overall_feedback"), str):
        raw["overall_feedback"] = ""

    return raw


def _build_writing_user_prompt(
    prompt_text: str, essay_text: str, task_type: int,
    task1_data_json: str | None = None,
) -> str:
    """Build user prompt for writing evaluation."""
    task_label = "Task 1" if task_type == 1 else "Task 2"
    min_words = 150 if task_type == 1 else 250
    word_count = len(essay_text.split())

    prompt = f"""## IELTS Writing {task_label}

**Question/Prompt:**
{prompt_text}

**Candidate's Essay ({word_count} words, minimum {min_words}):**
{essay_text}
"""
    if task1_data_json:
        prompt += f"\n**Chart Data (JSON):**\n{task1_data_json}\n"
    return prompt


def evaluate_writing(
    prompt_text: str,
    essay_text: str,
    task_type: int,
    task1_data_json: str | None = None,
) -> WritingEvaluation:
    """Evaluate a writing essay via Ollama."""
    system = WRITING_SYSTEM_PROMPT + "\n\n" + _WRITING_EVALUATION_SCHEMA
    user = _build_writing_user_prompt(prompt_text, essay_text, task_type, task1_data_json)
    raw_json = _chat(system, user)
    logger.info("Ollama writing raw response: %s", raw_json)
    data = json.loads(raw_json)
    data = _normalize_writing_evaluation(data)
    logger.info("Normalized writing data: %s", json.dumps(data, default=str)[:500])
    return WritingEvaluation.model_validate(data)


def evaluate_writing_enhanced(
    prompt_text: str,
    essay_text: str,
    task_type: int,
    task1_data_json: str | None = None,
) -> WritingEnhancedReview:
    """Evaluate writing with richer feedback via Ollama."""
    system = WRITING_ENHANCED_SYSTEM_PROMPT + "\n\n" + _WRITING_ENHANCED_SCHEMA
    user = _build_writing_user_prompt(prompt_text, essay_text, task_type, task1_data_json)
    raw_json = _chat(system, user)
    logger.info("Ollama writing enhanced raw response: %s", raw_json)
    data = json.loads(raw_json)
    data = _normalize_writing_evaluation(data)
    logger.info("Normalized writing enhanced data: %s", json.dumps(data, default=str)[:500])
    return WritingEnhancedReview.model_validate(data)
