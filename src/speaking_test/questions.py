"""Load IELTS questions and band-9 answers from the database."""

from __future__ import annotations

import random
import re

from speaking_test.models import MockTestPlan, Question, QuestionWithAnswer


# ---------------------------------------------------------------------------
# Load from DB
# ---------------------------------------------------------------------------

def load_all_questions() -> list[QuestionWithAnswer]:
    """Load all questions from the database, one random answer variant per question.

    The DB is seeded from CSV at init time. Caller should cache the result
    (e.g. via @st.cache_data) so the random variant choice is stable per session.
    """
    from speaking_test.database import get_all_questions_from_db

    rows = get_all_questions_from_db()
    result: list[QuestionWithAnswer] = []
    for r in rows:
        q = Question(
            part=r["part"],
            topic=r["topic"],
            text=r["question_text"],
            cue_card=r["cue_card"],
            source=r["source"],
            topic_category=r["topic"],
        )
        result.append(QuestionWithAnswer(question=q, band9_answer=r["band9_answer"]))
    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _norm_key(text: str) -> str:
    """Normalize text to a matching key."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text[:80]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_random_question(
    questions: list[QuestionWithAnswer],
    part: int | None = None,
) -> QuestionWithAnswer:
    """Pick a random question, optionally filtered by part number."""
    pool = questions
    if part is not None:
        pool = [q for q in questions if q.question.part == part]
    if not pool:
        pool = questions
    return random.choice(pool)


def get_all_topics(
    questions: list[QuestionWithAnswer],
) -> dict[int, list[str]]:
    """Get unique topics grouped by part number."""
    topics: dict[int, set[str]] = {1: set(), 2: set(), 3: set()}
    for qwa in questions:
        q = qwa.question
        if q.part in topics and q.topic:
            topics[q.part].add(q.topic)
    return {part: sorted(ts) for part, ts in topics.items()}


def assemble_mock_test(
    questions: list[QuestionWithAnswer] | None = None,
) -> MockTestPlan:
    """Build a complete mock test plan: Part 1 -> Part 2 -> Part 3.

    1. Pick 2 random Part 1 topics, select 4-5 questions from each (8-10 total)
    2. Pick 1 random Part 2 cue card
    3. Match Part 2 topic to a Part 3 theme via keyword overlap, pick 4-5 questions
    """
    if questions is None:
        questions = load_all_questions()

    # Group by part
    part1 = [q for q in questions if q.question.part == 1]
    part2 = [q for q in questions if q.question.part == 2]
    part3 = [q for q in questions if q.question.part == 3]

    # Part 1: pick 2 random topics, 4-5 questions each
    p1_by_topic: dict[str, list[QuestionWithAnswer]] = {}
    for q in part1:
        p1_by_topic.setdefault(q.question.topic, []).append(q)
    topics = list(p1_by_topic.keys())
    if len(topics) >= 2:
        chosen_topics = random.sample(topics, 2)
    else:
        chosen_topics = topics

    p1_questions: list[QuestionWithAnswer] = []
    for topic in chosen_topics:
        pool = p1_by_topic[topic]
        n = min(len(pool), random.choice([4, 5]))
        p1_questions.extend(random.sample(pool, n))

    # Part 2: pick 1 random cue card
    p2_card = random.choice(part2) if part2 else None

    # Part 3: match Part 2 topic to a Part 3 theme via keyword overlap
    p3_by_topic: dict[str, list[QuestionWithAnswer]] = {}
    for q in part3:
        p3_by_topic.setdefault(q.question.topic, []).append(q)

    p3_questions: list[QuestionWithAnswer] = []
    if p2_card and p3_by_topic:
        cue_words = set(_norm_key(p2_card.question.text).split())
        best_theme = ""
        best_overlap = 0
        for theme in p3_by_topic:
            theme_words = set(_norm_key(theme).split())
            overlap = len(cue_words & theme_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_theme = theme

        # If no good match, pick a random theme
        if best_overlap == 0:
            best_theme = random.choice(list(p3_by_topic.keys()))

        pool = p3_by_topic[best_theme]
        n = min(len(pool), random.choice([4, 5]))
        p3_questions = random.sample(pool, n)

    return MockTestPlan(
        part1_questions=p1_questions,
        part2_cue_card=p2_card,
        part3_questions=p3_questions,
    )
