"""Shared data models for IELTS Speaking & Writing Practice."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Question models (migrated from questions.py, extended)
# ---------------------------------------------------------------------------

@dataclass
class Question:
    part: int  # 1, 2, or 3
    topic: str  # e.g. "Flowers and plants"
    text: str  # the question text
    cue_card: str = ""  # Part 2 only: bullet points
    source: str = ""  # e.g. "question_bank", "master_pack", "ielts_questions"
    topic_category: str = ""  # e.g. "Work / Study", "Technology"
    test: str = ""  # e.g. "Test A" (legacy)


@dataclass
class QuestionWithAnswer:
    question: Question
    band9_answer: str = ""


# ---------------------------------------------------------------------------
# Content evaluation models
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    score: float = Field(ge=0, le=9, description="Band score 0-9")
    feedback: str = Field(description="Brief examiner feedback for this criterion")


class ContentEvaluation(BaseModel):
    coherence: CriterionScore = Field(description="Coherence and cohesion")
    lexical_resource: CriterionScore = Field(description="Lexical resource / vocabulary")
    grammatical_range: CriterionScore = Field(
        description="Grammatical range and accuracy"
    )
    task_response: CriterionScore = Field(
        description="Task achievement / response relevance"
    )
    overall_feedback: str = Field(description="Overall examiner summary")


# ---------------------------------------------------------------------------
# Enhanced review models (new)
# ---------------------------------------------------------------------------

class GrammarCorrection(BaseModel):
    original: str = Field(description="The original phrase with the error")
    corrected: str = Field(description="The corrected version")
    explanation: str = Field(description="Brief grammar rule explanation")


class VocabularyUpgrade(BaseModel):
    basic_word: str = Field(description="The basic/common word used")
    alternatives: list[str] = Field(description="2-3 advanced alternatives")
    example: str = Field(description="Example sentence using one alternative")


class PronunciationWarning(BaseModel):
    word: str = Field(description="Word from transcript that may be mispronounced")
    phonetic: str = Field(description="Correct pronunciation guide (simplified)")
    tip: str = Field(description="Common mistake and how to fix it")


class EnhancedReview(BaseModel):
    coherence: CriterionScore = Field(description="Coherence and cohesion")
    lexical_resource: CriterionScore = Field(description="Lexical resource / vocabulary")
    grammatical_range: CriterionScore = Field(
        description="Grammatical range and accuracy"
    )
    task_response: CriterionScore = Field(
        description="Task achievement / response relevance"
    )
    overall_feedback: str = Field(description="Overall examiner summary")
    grammar_corrections: list[GrammarCorrection] = Field(
        default_factory=list,
        description="Specific grammar errors with corrections",
    )
    vocabulary_upgrades: list[VocabularyUpgrade] = Field(
        default_factory=list,
        description="Vocabulary upgrade suggestions",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="2-3 specific things done well",
    )
    improvement_priorities: list[str] = Field(
        default_factory=list,
        description="2-3 specific, actionable improvement tips",
    )
    pronunciation_warnings: list[PronunciationWarning] = Field(
        default_factory=list,
        description="Words the candidate used that are commonly mispronounced",
    )


# ---------------------------------------------------------------------------
# Mock test models (new)
# ---------------------------------------------------------------------------

@dataclass
class MockTestPlan:
    part1_questions: list[QuestionWithAnswer] = field(default_factory=list)
    part2_cue_card: QuestionWithAnswer | None = None
    part3_questions: list[QuestionWithAnswer] = field(default_factory=list)


@dataclass
class MockTestResponse:
    question: QuestionWithAnswer
    transcript: str = ""
    audio_metrics: dict = field(default_factory=dict)
    evaluation: EnhancedReview | None = None
    combined_band: dict = field(default_factory=dict)


@dataclass
class MockTestState:
    plan: MockTestPlan
    current_part: int = 1  # 1, 2, or 3
    current_index: int = 0  # index within current part's questions
    responses: list[MockTestResponse] = field(default_factory=list)
    completed: bool = False
    started: bool = False


# ---------------------------------------------------------------------------
# Database record models (new)
# ---------------------------------------------------------------------------

@dataclass
class SessionRecord:
    id: int | None = None
    timestamp: str = ""
    mode: str = ""  # "interview", "mock_test", "practice"
    overall_band: float = 0.0
    attempt_count: int = 0


@dataclass
class AttemptRecord:
    id: int | None = None
    session_id: int = 0
    timestamp: str = ""
    part: int = 0
    topic: str = ""
    question_text: str = ""
    transcript: str = ""
    duration: float = 0.0
    overall_band: float = 0.0
    fluency_coherence: float = 0.0
    lexical_resource: float = 0.0
    grammatical_range: float = 0.0
    pronunciation: float = 0.0
    speech_rate: float = 0.0
    pause_ratio: float = 0.0
    pronunciation_confidence: float = 0.0
    examiner_feedback: str = ""
    grammar_corrections: str = ""  # JSON string
    vocabulary_upgrades: str = ""  # JSON string
    improvement_tips: str = ""  # JSON string
    band9_answer: str = ""
    strengths: str = ""  # JSON string
    pronunciation_warnings: str = ""  # JSON string
    source: str = ""


# ---------------------------------------------------------------------------
# Writing evaluation models
# ---------------------------------------------------------------------------

class WritingEvaluation(BaseModel):
    """Standard writing evaluation — 4 criteria + overall feedback."""
    task_achievement: CriterionScore = Field(
        description="Task 1: Task Achievement; Task 2: Task Response"
    )
    coherence: CriterionScore = Field(description="Coherence & Cohesion")
    lexical_resource: CriterionScore = Field(description="Lexical Resource")
    grammatical_range: CriterionScore = Field(
        description="Grammatical Range & Accuracy"
    )
    overall_feedback: str = Field(description="Overall examiner summary")


class WritingEnhancedReview(BaseModel):
    """Enhanced writing review with corrections and upgrades."""
    task_achievement: CriterionScore = Field(
        description="Task 1: Task Achievement; Task 2: Task Response"
    )
    coherence: CriterionScore = Field(description="Coherence & Cohesion")
    lexical_resource: CriterionScore = Field(description="Lexical Resource")
    grammatical_range: CriterionScore = Field(
        description="Grammatical Range & Accuracy"
    )
    overall_feedback: str = Field(description="Overall examiner summary")
    grammar_corrections: list[GrammarCorrection] = Field(
        default_factory=list,
        description="Specific grammar errors with corrections",
    )
    vocabulary_upgrades: list[VocabularyUpgrade] = Field(
        default_factory=list,
        description="Vocabulary upgrade suggestions",
    )
    paragraph_feedback: list[str] = Field(
        default_factory=list,
        description="Per-paragraph analysis",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="2-3 specific things done well",
    )
    improvement_priorities: list[str] = Field(
        default_factory=list,
        description="2-3 specific, actionable improvement tips",
    )


# ---------------------------------------------------------------------------
# Writing data models
# ---------------------------------------------------------------------------

@dataclass
class WritingPrompt:
    id: int
    test_type: str          # 'academic' | 'gt'
    task_type: int           # 1 or 2
    topic: str
    prompt_text: str
    chart_image_path: str | None = None
    task1_data_json: str = ""


@dataclass
class WritingAttemptRecord:
    id: int | None = None
    session_id: int = 0
    timestamp: str = ""
    prompt_id: int = 0
    task_type: int = 0
    essay_text: str = ""
    word_count: int = 0
    task_score: float = 0.0
    coherence_score: float = 0.0
    lexical_score: float = 0.0
    grammar_score: float = 0.0
    overall_band: float = 0.0
    examiner_feedback: str = ""
    paragraph_feedback: list = field(default_factory=list)
    grammar_corrections: list = field(default_factory=list)
    vocabulary_upgrades: list = field(default_factory=list)
    improvement_tips: list = field(default_factory=list)
