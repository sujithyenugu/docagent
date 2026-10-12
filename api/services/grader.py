"""Grading service — keyword matching and rubric-based scoring.

Workflow
--------
1. Extract markdown from uploaded student document (delegates to extraction service).
2. Match extracted text against answer key keywords.
3. (Optional) Apply a structured rubric to compute per-question scores and
   a final letter grade.

Rubrics are stored in-memory; extend with a database-backed store as needed.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in rubrics
# ---------------------------------------------------------------------------

_RUBRICS: dict[str, dict] = {
    "science_class8_ch1": {
        "rubric_id": "science_class8_ch1",
        "name": "Science Class 8 — Chapter 1: Crop Production",
        "subject": "Science",
        "grade_level": "Class 8",
        "total_marks": 20,
        "questions": [
            {
                "question_id": "q1",
                "question_text": "Define the term 'agriculture'.",
                "max_marks": 2,
                "keywords": ["cultivation", "crops", "farming", "land"],
                "partial_keywords": ["plants", "food"],
            },
            {
                "question_id": "q2",
                "question_text": "List two types of crops grown in India.",
                "max_marks": 4,
                "keywords": ["kharif", "rabi", "rice", "wheat", "maize", "cotton"],
                "partial_keywords": ["paddy", "sugarcane", "mustard", "peas"],
            },
            {
                "question_id": "q3",
                "question_text": "What is the importance of irrigation?",
                "max_marks": 4,
                "keywords": ["water", "growth", "drought", "supply", "crops"],
                "partial_keywords": ["canal", "well", "rain", "sprinkler"],
            },
            {
                "question_id": "q4",
                "question_text": "Explain the nitrogen cycle briefly.",
                "max_marks": 5,
                "keywords": ["nitrogen", "atmosphere", "bacteria", "fixation", "decomposition"],
                "partial_keywords": ["soil", "plants", "legumes", "lightning"],
            },
            {
                "question_id": "q5",
                "question_text": "What are fertilisers? Name two organic fertilisers.",
                "max_marks": 5,
                "keywords": ["nutrients", "manure", "compost", "fertiliser"],
                "partial_keywords": ["chemical", "organic", "urea", "vermicompost"],
            },
        ],
    },
    "maths_class10_algebra": {
        "rubric_id": "maths_class10_algebra",
        "name": "Mathematics Class 10 — Algebra",
        "subject": "Mathematics",
        "grade_level": "Class 10",
        "total_marks": 25,
        "questions": [
            {
                "question_id": "q1",
                "question_text": "Solve: 2x + 3 = 7",
                "max_marks": 3,
                "keywords": ["x", "2", "4", "solution"],
                "partial_keywords": ["subtract", "divide"],
            },
            {
                "question_id": "q2",
                "question_text": "Factorise: x² - 5x + 6",
                "max_marks": 4,
                "keywords": ["x-2", "x-3", "factors", "2", "3"],
                "partial_keywords": ["middle", "split", "quadratic"],
            },
            {
                "question_id": "q3",
                "question_text": "Find the roots of x² - 4 = 0.",
                "max_marks": 4,
                "keywords": ["x=2", "x=-2", "roots", "±2"],
                "partial_keywords": ["square root", "factoring", "difference"],
            },
            {
                "question_id": "q4",
                "question_text": "What is the discriminant of ax² + bx + c = 0?",
                "max_marks": 4,
                "keywords": ["b²-4ac", "discriminant", "b", "a", "c"],
                "partial_keywords": ["formula", "roots", "real"],
            },
            {
                "question_id": "q5",
                "question_text": "Solve the pair of equations: x+y=5 and x-y=1.",
                "max_marks": 5,
                "keywords": ["x=3", "y=2", "solution", "3", "2"],
                "partial_keywords": ["elimination", "substitution", "add", "subtract"],
            },
            {
                "question_id": "q6",
                "question_text": "What is the sum of roots of ax² + bx + c = 0?",
                "max_marks": 5,
                "keywords": ["-b/a", "sum", "roots", "vieta"],
                "partial_keywords": ["coefficient", "product", "c/a"],
            },
        ],
    },
    "history_class9_ch2": {
        "rubric_id": "history_class9_ch2",
        "name": "History Class 9 — Chapter 2: French Revolution",
        "subject": "History",
        "grade_level": "Class 9",
        "total_marks": 20,
        "questions": [
            {
                "question_id": "q1",
                "question_text": "When did the French Revolution begin?",
                "max_marks": 2,
                "keywords": ["1789", "revolution", "France"],
                "partial_keywords": ["eighteenth", "century"],
            },
            {
                "question_id": "q2",
                "question_text": "What were the three estates in France?",
                "max_marks": 4,
                "keywords": ["clergy", "nobility", "commoners", "third estate"],
                "partial_keywords": ["church", "peasants", "bourgeoisie", "first", "second"],
            },
            {
                "question_id": "q3",
                "question_text": "What is the Declaration of the Rights of Man?",
                "max_marks": 4,
                "keywords": ["liberty", "equality", "rights", "citizens", "freedom"],
                "partial_keywords": ["document", "1789", "national assembly"],
            },
            {
                "question_id": "q4",
                "question_text": "Describe the Reign of Terror.",
                "max_marks": 5,
                "keywords": ["robespierre", "guillotine", "execution", "radical", "jacobins"],
                "partial_keywords": ["fear", "1793", "1794", "committee", "terror"],
            },
            {
                "question_id": "q5",
                "question_text": "What were the causes of the French Revolution?",
                "max_marks": 5,
                "keywords": ["debt", "taxation", "inequality", "enlightenment", "famine"],
                "partial_keywords": ["bread", "estates", "king", "louis", "war"],
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _find_context(text: str, keyword: str, window: int = 60) -> str | None:
    """Return a short surrounding snippet if keyword is found in text."""
    norm_text = _normalise(text)
    norm_kw = _normalise(keyword)
    idx = norm_text.find(norm_kw)
    if idx == -1:
        return None
    start = max(0, idx - window)
    end = min(len(norm_text), idx + len(norm_kw) + window)
    return "…" + norm_text[start:end] + "…"


def match_keywords(text: str, keywords: list[str]) -> list[dict]:
    """Return a list of match dicts for each keyword."""
    results = []
    for kw in keywords:
        context = _find_context(text, kw)
        results.append({
            "keyword": kw,
            "found": context is not None,
            "context": context,
        })
    return results


# ---------------------------------------------------------------------------
# Rubric scoring
# ---------------------------------------------------------------------------

def _letter_grade(percentage: float) -> str:
    if percentage >= 90:
        return "A+"
    if percentage >= 80:
        return "A"
    if percentage >= 70:
        return "B"
    if percentage >= 60:
        return "C"
    if percentage >= 50:
        return "D"
    return "F"


def score_against_rubric(
    markdown: str,
    rubric_id: str,
    student_label: str | None = None,
) -> dict[str, Any]:
    """Score extracted markdown against a rubric.

    Args:
        markdown:      Extracted text from the student's answer sheet.
        rubric_id:     Rubric identifier (must exist in ``_RUBRICS``).
        student_label: Optional student name / roll number for record-keeping.

    Returns:
        Dict matching ``RubricEvaluateResponse`` schema.

    Raises:
        ValueError: Unknown rubric_id.
    """
    if rubric_id not in _RUBRICS:
        raise ValueError(
            f"Unknown rubric '{rubric_id}'. Available: {', '.join(_RUBRICS)}"
        )

    rubric = _RUBRICS[rubric_id]
    question_scores = []
    total_awarded = 0.0
    total_possible = 0.0

    for q in rubric["questions"]:
        max_marks = q["max_marks"]
        keyword_matches = match_keywords(markdown, q["keywords"])
        partial_matches = match_keywords(markdown, q.get("partial_keywords", []))

        # Full keywords: each found keyword contributes max_marks / len(keywords)
        n_full = len(q["keywords"])
        full_found = sum(1 for m in keyword_matches if m["found"])
        full_score = (full_found / n_full * max_marks) if n_full else 0.0

        # Partial keywords: each found partial contributes 0.5 * max_marks / len(partial)
        n_partial = len(q.get("partial_keywords", []))
        partial_found = sum(1 for m in partial_matches if m["found"])
        partial_score = (partial_found / n_partial * max_marks * 0.5) if n_partial else 0.0

        awarded = min(max_marks, full_score + partial_score)
        awarded = round(awarded, 2)

        all_matches = keyword_matches + partial_matches
        question_scores.append({
            "question_id": q["question_id"],
            "max_marks": max_marks,
            "awarded_marks": awarded,
            "keyword_matches": all_matches,
        })
        total_awarded += awarded
        total_possible += max_marks

    total_awarded = round(total_awarded, 2)
    percentage = round(total_awarded / total_possible * 100, 1) if total_possible else 0.0

    return {
        "rubric_id": rubric_id,
        "student_label": student_label,
        "question_scores": question_scores,
        "total_awarded": total_awarded,
        "total_possible": total_possible,
        "percentage": percentage,
        "grade": _letter_grade(percentage),
    }


def extract_and_match(
    doc_bytes: bytes,
    filename: str,
    keywords: list[str],
    answer_key: dict[str, Any] | None,
    student_label: str | None,
    use_local: bool,
    api_key: str | None,
    model: str,
) -> dict[str, Any]:
    """Extract markdown from a document then run keyword matching.

    Args:
        doc_bytes:     Raw uploaded file bytes.
        filename:      Original filename (determines file type).
        keywords:      Flat list of answer key terms for keyword matching.
        answer_key:    Optional structured answer key:
                       ``[{question_id, max_marks, keywords, partial_keywords}]``
        student_label: Optional student identifier.
        use_local:     True → embedded local MLX model; False → Landing AI.
        api_key:       Required when ``use_local=False``.
        model:         ADE model variant.

    Returns:
        Dict matching ``GradeExtractResponse`` schema.
    """
    from api.services.extraction import run as extract_run

    result = extract_run(doc_bytes, filename, use_local=use_local, api_key=api_key, model=model)
    markdown = result["markdown"]

    # Flat keyword matching
    flat_matches = match_keywords(markdown, keywords)

    # Structured answer key scoring
    question_scores: list[dict] = []
    total_awarded: float | None = None
    total_possible: float | None = None

    if answer_key:
        ta = 0.0
        tp = 0.0
        for q in answer_key:
            q_keywords = q.get("keywords", [])
            q_partial = q.get("partial_keywords", [])
            q_max = float(q.get("max_marks", 0))
            q_id = q.get("question_id", "")

            kw_matches = match_keywords(markdown, q_keywords)
            pk_matches = match_keywords(markdown, q_partial)

            n_full = len(q_keywords)
            full_found = sum(1 for m in kw_matches if m["found"])
            full_score = (full_found / n_full * q_max) if n_full else 0.0

            n_partial = len(q_partial)
            partial_found = sum(1 for m in pk_matches if m["found"])
            partial_score = (partial_found / n_partial * q_max * 0.5) if n_partial else 0.0

            awarded = round(min(q_max, full_score + partial_score), 2)
            question_scores.append({
                "question_id": q_id,
                "max_marks": q_max,
                "awarded_marks": awarded,
                "keyword_matches": kw_matches + pk_matches,
            })
            ta += awarded
            tp += q_max

        total_awarded = round(ta, 2)
        total_possible = round(tp, 2)

    return {
        "markdown": markdown,
        "student_label": student_label,
        "keyword_matches": flat_matches,
        "question_scores": question_scores,
        "total_awarded": total_awarded,
        "total_possible": total_possible,
        "metadata": {
            "page_count": result["page_count"],
            "duration_ms": result["duration_ms"],
            "model": result["model"],
            "backend": result["backend"],
        },
    }


def list_rubrics() -> list[dict]:
    """Return a summary list of all available rubrics."""
    return [
        {
            "rubric_id": r["rubric_id"],
            "name": r["name"],
            "subject": r["subject"],
            "grade_level": r["grade_level"],
            "total_marks": r["total_marks"],
            "question_count": len(r["questions"]),
        }
        for r in _RUBRICS.values()
    ]
