"""Provider-agnostic facade for IELTS answer evaluation."""

from __future__ import annotations

import os
import time

from speaking_test.gemini_evaluator import (
    compute_combined_band,
    compute_writing_band,
    detect_fillers,
    writing_quality_checks,
)
from speaking_test.models import (
    EnhancedReview,
    ContentEvaluation,
    WritingEvaluation,
    WritingEnhancedReview,
)

__all__ = [
    "get_provider",
    "get_last_eval_meta",
    "is_provider_configured",
    "evaluate_answer",
    "evaluate_answer_enhanced",
    "compute_combined_band",
    "detect_fillers",
    "evaluate_writing",
    "evaluate_writing_enhanced",
    "compute_writing_band",
    "writing_quality_checks",
]

# Metadata from the most recent evaluation call (provider, model, timing).
_last_eval_meta: dict = {}


def get_last_eval_meta() -> dict:
    """Return metadata captured from the last evaluation call."""
    return dict(_last_eval_meta)


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
    global _last_eval_meta
    provider = get_provider()
    t0 = time.perf_counter()

    if provider == "ollama":
        from speaking_test import ollama_evaluator

        result = ollama_evaluator.evaluate_answer(question, part, transcript, band9_answer)
        _last_eval_meta = {
            "provider": "ollama",
            "model_name": ollama_evaluator._OLLAMA_MODEL,
            "response_time_ms": round((time.perf_counter() - t0) * 1000),
        }
        return result

    from speaking_test.gemini_evaluator import (
        create_gemini_client,
        evaluate_answer as gemini_evaluate,
        get_model_name,
    )

    client = create_gemini_client()
    model = get_model_name()
    result = gemini_evaluate(client, model, question, part, transcript, band9_answer)
    _last_eval_meta = {
        "provider": "gemini",
        "model_name": model,
        "response_time_ms": round((time.perf_counter() - t0) * 1000),
    }
    return result


def evaluate_answer_enhanced(
    question: str,
    part: int,
    transcript: str,
    band9_answer: str = "",
) -> EnhancedReview:
    """Dispatch enhanced evaluation to the configured provider."""
    global _last_eval_meta
    provider = get_provider()
    t0 = time.perf_counter()

    if provider == "ollama":
        from speaking_test import ollama_evaluator

        result = ollama_evaluator.evaluate_answer_enhanced(
            question, part, transcript, band9_answer
        )
        _last_eval_meta = {
            "provider": "ollama",
            "model_name": ollama_evaluator._OLLAMA_MODEL,
            "response_time_ms": round((time.perf_counter() - t0) * 1000),
        }
        return result

    from speaking_test.gemini_evaluator import (
        create_gemini_client,
        evaluate_answer_enhanced as gemini_evaluate_enhanced,
        get_model_name,
    )

    client = create_gemini_client()
    model = get_model_name()
    result = gemini_evaluate_enhanced(client, model, question, part, transcript, band9_answer)
    _last_eval_meta = {
        "provider": "gemini",
        "model_name": model,
        "response_time_ms": round((time.perf_counter() - t0) * 1000),
    }
    return result


# ---------------------------------------------------------------------------
# Writing evaluation facade
# ---------------------------------------------------------------------------

def evaluate_writing(
    prompt_text: str,
    essay_text: str,
    task_type: int,
    task1_data_json: str | None = None,
) -> WritingEvaluation:
    """Dispatch writing evaluation to the configured provider."""
    global _last_eval_meta
    provider = get_provider()
    t0 = time.perf_counter()

    if provider == "ollama":
        from speaking_test import ollama_evaluator

        result = ollama_evaluator.evaluate_writing(
            prompt_text, essay_text, task_type, task1_data_json
        )
        _last_eval_meta = {
            "provider": "ollama",
            "model_name": ollama_evaluator._OLLAMA_MODEL,
            "response_time_ms": round((time.perf_counter() - t0) * 1000),
        }
        return result

    from speaking_test.gemini_evaluator import (
        create_gemini_client,
        evaluate_writing as gemini_evaluate_writing,
        get_model_name,
    )

    client = create_gemini_client()
    model = get_model_name()
    result = gemini_evaluate_writing(
        client, model, prompt_text, essay_text, task_type, task1_data_json
    )
    _last_eval_meta = {
        "provider": "gemini",
        "model_name": model,
        "response_time_ms": round((time.perf_counter() - t0) * 1000),
    }
    return result


def evaluate_writing_enhanced(
    prompt_text: str,
    essay_text: str,
    task_type: int,
    task1_data_json: str | None = None,
) -> WritingEnhancedReview:
    """Dispatch enhanced writing evaluation to the configured provider."""
    global _last_eval_meta
    provider = get_provider()
    t0 = time.perf_counter()

    if provider == "ollama":
        from speaking_test import ollama_evaluator

        result = ollama_evaluator.evaluate_writing_enhanced(
            prompt_text, essay_text, task_type, task1_data_json
        )
        _last_eval_meta = {
            "provider": "ollama",
            "model_name": ollama_evaluator._OLLAMA_MODEL,
            "response_time_ms": round((time.perf_counter() - t0) * 1000),
        }
        return result

    from speaking_test.gemini_evaluator import (
        create_gemini_client,
        evaluate_writing_enhanced as gemini_evaluate_writing_enhanced,
        get_model_name,
    )

    client = create_gemini_client()
    model = get_model_name()
    result = gemini_evaluate_writing_enhanced(
        client, model, prompt_text, essay_text, task_type, task1_data_json
    )
    _last_eval_meta = {
        "provider": "gemini",
        "model_name": model,
        "response_time_ms": round((time.perf_counter() - t0) * 1000),
    }
    return result
