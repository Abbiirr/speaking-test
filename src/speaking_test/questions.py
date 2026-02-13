"""Parse IELTS questions and band-9 answers from markdown files."""

from __future__ import annotations

import random
import re
from pathlib import Path

from speaking_test.models import MockTestPlan, Question, QuestionWithAnswer

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


# ---------------------------------------------------------------------------
# Original parser (ielts_questions.md format)
# ---------------------------------------------------------------------------

def _parse_questions(md_text: str) -> list[Question]:
    """Parse questions from the ielts_questions.md format."""
    questions: list[Question] = []
    current_test = ""
    current_topic = ""
    current_part = 0
    cue_card_lines: list[str] = []
    in_cue_card = False

    for line in md_text.splitlines():
        line_stripped = line.strip()

        # ## Test A — Flowers and plants / Reviews / Customer service (...)
        test_match = re.match(r"^##\s+(.+?)(?:\s*\(.*\))?\s*$", line_stripped)
        if test_match and not line_stripped.startswith("###"):
            full = test_match.group(1).strip()
            dash_idx = full.find("—")
            if dash_idx != -1:
                current_test = full[:dash_idx].strip()
            else:
                current_test = full
            in_cue_card = False
            continue

        # ### Part 1 — Topic
        part_match = re.match(r"^###\s+Part\s+(\d)\s*[—\-]\s*(.+)", line_stripped)
        if part_match:
            current_part = int(part_match.group(1))
            current_topic = part_match.group(2).strip()
            in_cue_card = False
            continue

        # #### Discussion sub-topic (Part 3)
        subtopic_match = re.match(r"^####\s+(.+)", line_stripped)
        if subtopic_match:
            current_topic = subtopic_match.group(1).strip()
            in_cue_card = False
            continue

        # Part 2 cue card: starts with **Describe ...** on its own line
        if current_part == 2 and line_stripped.startswith("**Describe"):
            q_text = line_stripped.strip("* ")
            cue_card_lines = []
            in_cue_card = True
            questions.append(
                Question(
                    test=current_test,
                    part=2,
                    topic=current_topic,
                    text=q_text,
                    source="ielts_questions",
                    topic_category=current_topic,
                )
            )
            continue

        # Collect cue card bullet points
        if in_cue_card:
            if line_stripped.startswith("- ") or line_stripped.startswith("You should"):
                cue_card_lines.append(line_stripped)
                questions[-1].cue_card = "\n".join(cue_card_lines)
                continue
            elif line_stripped == "":
                continue
            else:
                in_cue_card = False

        # Numbered questions: 1. Do you ...
        q_match = re.match(r"^\d+\.\s+(.+)", line_stripped)
        if q_match and current_part in (1, 3):
            q_text = q_match.group(1).strip()
            q_text = re.sub(r"\s*\*\*\[.*?\]\*\*\s*$", "", q_text)
            questions.append(
                Question(
                    test=current_test,
                    part=current_part,
                    topic=current_topic,
                    text=q_text,
                    source="ielts_questions",
                    topic_category=current_topic,
                )
            )

    return questions


# ---------------------------------------------------------------------------
# Question bank parser (ielts_speaking_question_bank.md)
# ---------------------------------------------------------------------------

def _parse_question_bank(md_text: str) -> list[Question]:
    """Parse questions from the ielts_speaking_question_bank.md format.

    Structure:
    - Part 1: ### N. Topic Name → numbered questions
    - Part 2: ### Cue Card N: Describe ... → You should say: + bullets
    - Part 3: ### Theme N: Topic → numbered questions
    """
    questions: list[Question] = []
    current_part = 0
    current_topic = ""
    cue_card_lines: list[str] = []
    in_cue_card = False

    for line in md_text.splitlines():
        stripped = line.strip()

        # Detect part sections
        if stripped.startswith("## Part 1"):
            current_part = 1
            current_topic = ""
            in_cue_card = False
            continue
        if stripped.startswith("## Part 2"):
            current_part = 2
            current_topic = ""
            in_cue_card = False
            continue
        if stripped.startswith("## Part 3"):
            current_part = 3
            current_topic = ""
            in_cue_card = False
            continue

        # Skip non-question ## headers
        if stripped.startswith("## "):
            current_part = 0
            in_cue_card = False
            continue

        # Part 1: ### N. Topic Name
        if current_part == 1:
            topic_match = re.match(r"^###\s+\d+\.\s+(.+)", stripped)
            if topic_match:
                current_topic = topic_match.group(1).strip()
                in_cue_card = False
                continue

            q_match = re.match(r"^\d+\.\s+(.+)", stripped)
            if q_match:
                questions.append(
                    Question(
                        part=1,
                        topic=current_topic,
                        text=q_match.group(1).strip(),
                        source="question_bank",
                        topic_category=current_topic,
                    )
                )
                continue

        # Part 2: ### Cue Card N: Describe ...
        if current_part == 2:
            cue_match = re.match(
                r"^###\s+Cue Card\s+\d+:\s+(.+?)(?:\s*\(.*\))?\s*$", stripped
            )
            if cue_match:
                in_cue_card = True
                cue_card_lines = []
                title = cue_match.group(1).strip()
                # Extract topic category from parenthetical if present
                cat_match = re.search(r"\(([^)]+)\)", stripped)
                cat = cat_match.group(1) if cat_match else title
                questions.append(
                    Question(
                        part=2,
                        topic=title,
                        text=title,
                        source="question_bank",
                        topic_category=cat,
                    )
                )
                continue

            if in_cue_card:
                if stripped.startswith("- ") or stripped.startswith("You should"):
                    cue_card_lines.append(stripped)
                    questions[-1].cue_card = "\n".join(cue_card_lines)
                    continue
                elif stripped == "":
                    continue
                elif stripped.startswith("###"):
                    in_cue_card = False
                    # fall through to re-process
                else:
                    continue

        # Part 3: ### Theme N: Topic
        if current_part == 3:
            theme_match = re.match(r"^###\s+Theme\s+\d+:\s+(.+)", stripped)
            if theme_match:
                current_topic = theme_match.group(1).strip()
                in_cue_card = False
                continue

            q_match = re.match(r"^\d+\.\s+(.+)", stripped)
            if q_match:
                questions.append(
                    Question(
                        part=3,
                        topic=current_topic,
                        text=q_match.group(1).strip(),
                        source="question_bank",
                        topic_category=current_topic,
                    )
                )
                continue

    return questions


# ---------------------------------------------------------------------------
# Master pack parser (ielts_speaking_master_pack_v2.md)
# ---------------------------------------------------------------------------

def _parse_master_pack(md_text: str) -> list[Question]:
    """Parse questions from the ielts_speaking_master_pack_v2.md format.

    Questions live under:
    - ## Part 1 question bank → ### N. Topic → numbered questions
    - ## Part 2 cue card bank → ### Cue Card N: Describe ...
    - ## Part 3 discussion bank → ### Theme N: Topic → numbered questions
    """
    questions: list[Question] = []
    current_section = ""  # "part1", "part2", "part3", or ""
    current_topic = ""
    cue_card_lines: list[str] = []
    in_cue_card = False

    for line in md_text.splitlines():
        stripped = line.strip()

        # Detect major sections
        lower = stripped.lower()
        if lower.startswith("## part 1 question bank") or lower.startswith("## part 1"):
            if "question bank" in lower or "question" in lower:
                current_section = "part1"
                current_topic = ""
                in_cue_card = False
                continue
        if lower.startswith("## part 2 cue card bank") or lower.startswith("## part 2"):
            if "cue card" in lower or "cue" in lower:
                current_section = "part2"
                current_topic = ""
                in_cue_card = False
                continue
        if lower.startswith("## part 3 discussion bank") or lower.startswith("## part 3"):
            if "discussion" in lower:
                current_section = "part3"
                current_topic = ""
                in_cue_card = False
                continue

        # Other ## headers reset section (e.g. "## Band-9 sample answers")
        if stripped.startswith("## ") and not stripped.startswith("###"):
            current_section = ""
            in_cue_card = False
            continue

        if current_section == "part1":
            topic_match = re.match(r"^###\s+\d+\.\s+(.+)", stripped)
            if topic_match:
                current_topic = topic_match.group(1).strip()
                continue

            q_match = re.match(r"^\d+\.\s+(.+)", stripped)
            if q_match:
                q_text = q_match.group(1).strip()
                # Strip trailing Why/Why not markers
                q_text = re.sub(r"\s*\*?\*?\[?Why/?Why not\??\]?\*?\*?\s*$", "", q_text)
                if q_text:
                    questions.append(
                        Question(
                            part=1,
                            topic=current_topic,
                            text=q_text,
                            source="master_pack",
                            topic_category=current_topic,
                        )
                    )
                continue

        if current_section == "part2":
            cue_match = re.match(
                r"^###\s+Cue Card\s+\d+:\s+(.+?)(?:\s*\(.*\))?\s*$", stripped
            )
            if cue_match:
                in_cue_card = True
                cue_card_lines = []
                title = cue_match.group(1).strip()
                cat_match = re.search(r"\(([^)]+)\)", stripped)
                cat = cat_match.group(1) if cat_match else title
                questions.append(
                    Question(
                        part=2,
                        topic=title,
                        text=title,
                        source="master_pack",
                        topic_category=cat,
                    )
                )
                continue

            if in_cue_card:
                if stripped.startswith("- ") or stripped.startswith("You should"):
                    cue_card_lines.append(stripped)
                    questions[-1].cue_card = "\n".join(cue_card_lines)
                    continue
                elif stripped == "":
                    continue
                elif stripped.startswith("###"):
                    in_cue_card = False
                else:
                    continue

        if current_section == "part3":
            theme_match = re.match(r"^###\s+Theme\s+\d+:\s+(.+)", stripped)
            if theme_match:
                current_topic = theme_match.group(1).strip()
                continue

            q_match = re.match(r"^\d+\.\s+(.+)", stripped)
            if q_match:
                questions.append(
                    Question(
                        part=3,
                        topic=current_topic,
                        text=q_match.group(1).strip(),
                        source="master_pack",
                        topic_category=current_topic,
                    )
                )
                continue

    return questions


# ---------------------------------------------------------------------------
# Answer parsing (unchanged)
# ---------------------------------------------------------------------------

def _parse_answers(md_text: str) -> dict[str, str]:
    """Parse band-9 answers keyed by a normalized question snippet."""
    answers: dict[str, str] = {}
    current_key = ""
    current_lines: list[str] = []

    def _flush():
        if current_key and current_lines:
            answers[current_key] = "\n".join(current_lines).strip()

    for line in md_text.splitlines():
        stripped = line.strip()

        q_match = re.match(r"^\*\*(?:\d+\)\s+)?(.+?)\*\*", stripped)
        if q_match:
            _flush()
            raw = q_match.group(1).strip()
            raw = re.sub(r"\s*\(.*?\)\s*", " ", raw).strip()
            current_key = _norm_key(raw)
            after_bold = re.sub(r"^\*\*.*?\*\*\s*", "", stripped).strip()
            current_lines = [after_bold] if after_bold else []
            continue

        if stripped.startswith("### Part 2"):
            _flush()
            current_key = "__part2__"
            current_lines = []
            continue

        if stripped.startswith("#") or stripped == "---":
            _flush()
            current_key = ""
            current_lines = []
            continue

        if current_key:
            current_lines.append(stripped)

    _flush()
    return answers


def _parse_master_pack_answers(md_text: str) -> dict[str, str]:
    """Parse band-9 sample answers from the master pack."""
    answers: dict[str, str] = {}
    in_samples = False
    current_key = ""
    current_lines: list[str] = []

    def _flush():
        if current_key and current_lines:
            answers[current_key] = "\n".join(current_lines).strip()

    for line in md_text.splitlines():
        stripped = line.strip()

        if stripped.lower().startswith("## band") and "sample" in stripped.lower():
            in_samples = True
            continue

        if not in_samples:
            continue

        # New ## section ends samples
        if stripped.startswith("## ") and not stripped.startswith("###"):
            _flush()
            in_samples = False
            continue

        # **Q:** or **Cue card:** patterns
        q_match = re.match(r"^\*\*Q:\*\*\s*(.+)", stripped)
        if q_match:
            _flush()
            current_key = _norm_key(q_match.group(1).strip())
            current_lines = []
            continue

        cue_match = re.match(r"^\*\*Cue card:\*\*\s*(.+)", stripped)
        if cue_match:
            _flush()
            current_key = _norm_key(cue_match.group(1).strip())
            current_lines = []
            continue

        # **A:** starts the answer
        a_match = re.match(r"^\*\*A:\*\*\s*(.*)", stripped)
        if a_match:
            current_lines.append(a_match.group(1).strip())
            continue

        # **Model answer** header
        if stripped.lower().startswith("**model answer"):
            continue

        # **1-minute notes** or **Short follow-up** — skip
        if stripped.lower().startswith("**1") or stripped.lower().startswith("**short follow"):
            continue

        # ### sub-headers within samples — skip
        if stripped.startswith("### "):
            _flush()
            current_key = ""
            current_lines = []
            continue

        if current_key:
            current_lines.append(stripped)

    _flush()
    return answers


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _norm_key(text: str) -> str:
    """Normalize text to a matching key."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text[:80]


_STOP_WORDS = frozenset(
    "a an the is are was were be been being do does did doing have has had "
    "having i me my you your he she it we they them his her its our their "
    "to in on at by for of with from about into through during before after "
    "and or but not no nor so yet if when what which who whom how where why "
    "will would can could may might shall should that this these those some "
    "any all each every much many more most than too also very just even "
    "because then there here up down out off over under again further once "
    "think like really important prefer describe tell talk kinds often".split()
)


def _content_words(text: str) -> set[str]:
    """Extract meaningful (non-stop) words from a normalized key."""
    return {w for w in text.split() if w not in _STOP_WORDS}


def _match_answer(question: Question, answers: dict[str, str]) -> str:
    """Find the best matching answer for a question.

    Uses strict matching to avoid returning wrong reference answers.
    A wrong reference answer is worse than no reference answer because
    it misleads the AI evaluator's calibration.
    """
    q_key = _norm_key(question.text)

    # Exact match
    if q_key in answers:
        return answers[q_key]

    q_content = _content_words(q_key)
    if not q_content:
        q_content = set(q_key.split())

    best_key = ""
    best_score = 0.0
    for a_key in answers:
        a_content = _content_words(a_key)
        if not a_content:
            a_content = set(a_key.split())
        overlap = len(q_content & a_content)
        min_size = min(len(q_content), len(a_content))
        if min_size == 0:
            continue
        score = overlap / min_size
        # Strict: require at least 3 content words AND 70% overlap
        if overlap >= 3 and score >= 0.7 and score > best_score:
            best_score = score
            best_key = a_key

    if best_key:
        return answers[best_key]

    if question.part == 2 and "__part2__" in answers:
        return answers["__part2__"]

    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_questions(
    questions_path: Path | None = None,
    answers_path: Path | None = None,
) -> list[QuestionWithAnswer]:
    """Load and pair questions with their band-9 reference answers (original 16)."""
    if questions_path is None:
        questions_path = DOCS_DIR / "ielts_questions.md"
    if answers_path is None:
        answers_path = DOCS_DIR / "ielts_band9_answers.md"

    questions = _parse_questions(questions_path.read_text(encoding="utf-8"))

    answers: dict[str, str] = {}
    if answers_path.exists():
        answers = _parse_answers(answers_path.read_text(encoding="utf-8"))

    result = []
    for q in questions:
        band9 = _match_answer(q, answers)
        result.append(QuestionWithAnswer(question=q, band9_answer=band9))

    return result


def load_all_questions() -> list[QuestionWithAnswer]:
    """Load questions from all 3 sources, deduplicate by normalized text."""
    all_questions: list[Question] = []

    # Source 1: ielts_questions.md
    path1 = DOCS_DIR / "ielts_questions.md"
    if path1.exists():
        all_questions.extend(_parse_questions(path1.read_text(encoding="utf-8")))

    # Source 2: ielts_speaking_question_bank.md
    path2 = DOCS_DIR / "ielts_speaking_question_bank.md"
    if path2.exists():
        all_questions.extend(_parse_question_bank(path2.read_text(encoding="utf-8")))

    # Source 3: ielts_speaking_master_pack_v2.md
    path3 = DOCS_DIR / "ielts_speaking_master_pack_v2.md"
    if path3.exists():
        all_questions.extend(_parse_master_pack(path3.read_text(encoding="utf-8")))

    # Collect all answers
    answers: dict[str, str] = {}
    answers_path = DOCS_DIR / "ielts_band9_answers.md"
    if answers_path.exists():
        answers.update(_parse_answers(answers_path.read_text(encoding="utf-8")))
    if path3.exists():
        answers.update(
            _parse_master_pack_answers(path3.read_text(encoding="utf-8"))
        )

    # Deduplicate by normalized text + part
    seen: set[str] = set()
    unique: list[Question] = []
    for q in all_questions:
        key = f"{q.part}:{_norm_key(q.text)}"
        if key not in seen:
            seen.add(key)
            unique.append(q)

    # Pair with answers
    result = []
    for q in unique:
        band9 = _match_answer(q, answers)
        result.append(QuestionWithAnswer(question=q, band9_answer=band9))

    return result


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
    """Build a complete mock test plan: Part 1 → Part 2 → Part 3.

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
