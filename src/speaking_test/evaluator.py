"""Provider-agnostic facade for IELTS answer evaluation."""

from __future__ import annotations

import os

from speaking_test.gemini_evaluator import compute_combined_band, detect_fillers
from speaking_test.models import EnhancedReview, ContentEvaluation

__all__ = [
    "get_provider",
    "is_provider_configured",
    "evaluate_answer",
    "evaluate_answer_enhanced",
    "compute_combined_band",
    "detect_fillers",
]


def get_provider() -> str:
    """Return the configured evaluation provider ('gemini' or 'ollama')."""
    return os.environ.get("PROVIDER", "gemini").lower()


def is_provider_configured() -> bool:
    """Check whether the active provider is ready to use."""
    provider = get_provider()
    if provider == "ollama":
        from speaking_test.ollama_evaluator import is_available

        return is_available()
    # Default: gemini
    return bool(os.environ.get("GEMINI_API_KEY"))


def evaluate_answer(
    question: str,
    part: int,
    transcript: str,
    band9_answer: str = "",
) -> ContentEvaluation:
    """Dispatch evaluation to the configured provider."""
    provider = get_provider()
    if provider == "ollama":
        from speaking_test import ollama_evaluator

        return ollama_evaluator.evaluate_answer(question, part, transcript, band9_answer)

    from speaking_test.gemini_evaluator import (
        create_gemini_client,
        evaluate_answer as gemini_evaluate,
        get_model_name,
    )

    client = create_gemini_client()
    model = get_model_name()
    return gemini_evaluate(client, model, question, part, transcript, band9_answer)


def evaluate_answer_enhanced(
    question: str,
    part: int,
    transcript: str,
    band9_answer: str = "",
) -> EnhancedReview:
    """Dispatch enhanced evaluation to the configured provider."""
    provider = get_provider()
    if provider == "ollama":
        from speaking_test import ollama_evaluator

        return ollama_evaluator.evaluate_answer_enhanced(
            question, part, transcript, band9_answer
        )

    from speaking_test.gemini_evaluator import (
        create_gemini_client,
        evaluate_answer_enhanced as gemini_evaluate_enhanced,
        get_model_name,
    )

    client = create_gemini_client()
    model = get_model_name()
    return gemini_evaluate_enhanced(client, model, question, part, transcript, band9_answer)
