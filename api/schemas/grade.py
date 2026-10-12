"""Pydantic v2 schemas for /api/v1/grade/* endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class KeywordMatch(BaseModel):
    keyword: str
    found: bool
    context: str | None = Field(
        default=None,
        description="Surrounding text snippet where the keyword was found, or null.",
    )


class QuestionScore(BaseModel):
    question_id: str
    max_marks: float
    awarded_marks: float | None = Field(
        default=None,
        description="Marks awarded based on keyword matching; null if answer key not provided.",
    )
    keyword_matches: list[KeywordMatch] = Field(default_factory=list)


class GradeExtractResponse(BaseModel):
    markdown: str = Field(description="Extracted text from the student document.")
    student_label: str | None = Field(
        default=None,
        description="Student name or roll number supplied by the teacher.",
    )
    keyword_matches: list[KeywordMatch] = Field(
        default_factory=list,
        description="Match results for each answer key term.",
    )
    question_scores: list[QuestionScore] = Field(
        default_factory=list,
        description="Per-question scores when a structured answer key is provided.",
    )
    total_awarded: float | None = Field(
        default=None,
        description="Sum of awarded marks across all questions.",
    )
    total_possible: float | None = Field(
        default=None,
        description="Sum of maximum marks across all questions.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extraction metadata: page_count, duration_ms, model, backend.",
    )
    request_id: str


class RubricQuestion(BaseModel):
    question_id: str
    question_text: str
    max_marks: float
    keywords: list[str] = Field(
        description="Answer key terms that must appear for full marks."
    )
    partial_keywords: list[str] = Field(
        default_factory=list,
        description="Terms that earn partial credit (50% of max_marks each, capped at max).",
    )


class Rubric(BaseModel):
    rubric_id: str
    name: str
    subject: str
    grade_level: str
    total_marks: float
    questions: list[RubricQuestion]


class RubricEvaluateResponse(BaseModel):
    rubric_id: str
    student_label: str | None = None
    question_scores: list[QuestionScore]
    total_awarded: float
    total_possible: float
    percentage: float
    grade: str = Field(description="Letter grade: A+, A, B, C, D, F")
    request_id: str
