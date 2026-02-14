"""Load writing prompts from the database."""

from __future__ import annotations

import logging
import random

from speaking_test.models import WritingPrompt

logger = logging.getLogger(__name__)


def load_writing_prompts(
    test_type: str | None = None,
    task_type: int | None = None,
) -> list[WritingPrompt]:
    """Load writing prompts from DB, optionally filtered."""
    from speaking_test.database import get_all_writing_prompts

    rows = get_all_writing_prompts()
    logger.debug("Loaded %d writing prompts (test_type=%s, task_type=%s)", len(rows), test_type, task_type)
    result: list[WritingPrompt] = []
    for r in rows:
        if test_type and r["test_type"] != test_type:
            continue
        if task_type is not None and r["task_type"] != task_type:
            continue
        result.append(WritingPrompt(
            id=r["id"],
            test_type=r["test_type"],
            task_type=r["task_type"],
            topic=r.get("topic", ""),
            prompt_text=r["prompt_text"],
            chart_image_path=r.get("chart_image_path"),
            task1_data_json=r.get("task1_data_json", ""),
        ))
    return result


def get_random_writing_prompt(
    prompts: list[WritingPrompt],
    test_type: str | None = None,
    task_type: int | None = None,
    topic: str | None = None,
) -> WritingPrompt | None:
    """Pick a random writing prompt with optional filters."""
    pool = prompts
    if test_type:
        pool = [p for p in pool if p.test_type == test_type]
    if task_type is not None:
        pool = [p for p in pool if p.task_type == task_type]
    if topic:
        pool = [p for p in pool if topic.lower() in p.topic.lower()]
    if not pool:
        return None
    return random.choice(pool)
