"""Teacher grading endpoints.

POST /api/v1/grade/extract            — extract + keyword-match in one call
GET  /api/v1/grade/rubrics            — list available grading rubrics
GET  /api/v1/grade/rubrics/{id}       — get rubric detail
POST /api/v1/grade/rubrics/{id}/evaluate — score extracted markdown against a rubric
"""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.middleware.rate_limit import RateLimiter
from api.schemas.grade import GradeExtractResponse, Rubric, RubricEvaluateResponse
from api.services import grader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/grade", tags=["grading"])

_limiter = RateLimiter(local_rpm=10, cloud_rpm=1)


# ---------------------------------------------------------------------------
# POST /grade/extract
# ---------------------------------------------------------------------------

@router.post(
    "/extract",
    response_model=GradeExtractResponse,
    responses={
        429: {"description": "Rate limit exceeded"},
        400: {"description": "Bad request (unsupported file, missing key, bad JSON)"},
        502: {"description": "Backend not reachable"},
    },
    summary="Extract student answer sheet and match against answer key",
)
async def grade_extract(
    request: Request,
    file: Annotated[UploadFile, File(description="Student answer sheet (PDF/PNG/JPEG)")],
    use_local: Annotated[bool, Form()] = True,
    api_key: Annotated[str, Form()] = "",
    model: Annotated[str, Form()] = "dpt-2-latest",
    student_label: Annotated[str, Form()] = "",
    keywords: Annotated[str, Form(description="Comma-separated answer key terms")] = "",
    answer_key: Annotated[
        str,
        Form(
            description=(
                "Optional JSON array of questions: "
                "[{question_id, max_marks, keywords[], partial_keywords[]}]"
            )
        ),
    ] = "",
) -> GradeExtractResponse:
    """Extract text from a student answer sheet and optionally score it.

    - **keywords**: comma-separated terms to highlight in the extracted text
      (e.g. ``photosynthesis,chlorophyll,sunlight``).
    - **answer_key**: structured JSON for per-question scoring (optional).
      Format: ``[{"question_id":"q1","max_marks":5,"keywords":["..."],"partial_keywords":["..."]}]``
    """
    client_ip = request.client.host if request.client else "unknown"
    _limiter.check(client_ip, is_local=use_local)
    request_id: str = getattr(request.state, "request_id", "")

    doc_bytes = await file.read()
    filename = file.filename or "document"

    kw_list: list[str] = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []

    parsed_answer_key: list[dict[str, Any]] | None = None
    if answer_key:
        try:
            parsed_answer_key = json.loads(answer_key)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in answer_key: {exc}",
            ) from exc

    try:
        result = grader.extract_and_match(
            doc_bytes=doc_bytes,
            filename=filename,
            keywords=kw_list,
            answer_key=parsed_answer_key,
            student_label=student_label or None,
            use_local=use_local,
            api_key=api_key or None,
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return GradeExtractResponse(**result, request_id=request_id)


# ---------------------------------------------------------------------------
# GET /grade/rubrics
# ---------------------------------------------------------------------------

@router.get(
    "/rubrics",
    summary="List available grading rubrics",
)
async def list_rubrics() -> list[dict]:
    """Return all rubrics with subject, grade level, total marks, and question count."""
    return grader.list_rubrics()


# ---------------------------------------------------------------------------
# GET /grade/rubrics/{rubric_id}
# ---------------------------------------------------------------------------

@router.get(
    "/rubrics/{rubric_id}",
    response_model=Rubric,
    responses={404: {"description": "Rubric not found"}},
    summary="Get rubric detail including all questions and keywords",
)
async def get_rubric(rubric_id: str) -> Rubric:
    rubrics = {r["rubric_id"]: r for r in grader._RUBRICS.values()}
    if rubric_id not in rubrics:
        raise HTTPException(status_code=404, detail=f"Rubric '{rubric_id}' not found.")
    return Rubric(**rubrics[rubric_id])


# ---------------------------------------------------------------------------
# POST /grade/rubrics/{rubric_id}/evaluate
# ---------------------------------------------------------------------------

@router.post(
    "/rubrics/{rubric_id}/evaluate",
    response_model=RubricEvaluateResponse,
    responses={
        404: {"description": "Rubric not found"},
        429: {"description": "Rate limit exceeded"},
        400: {"description": "Bad request"},
        502: {"description": "Backend not reachable"},
    },
    summary="Extract student answer sheet then score against a rubric",
)
async def rubric_evaluate(
    request: Request,
    rubric_id: str,
    file: Annotated[UploadFile, File(description="Student answer sheet (PDF/PNG/JPEG)")],
    use_local: Annotated[bool, Form()] = True,
    api_key: Annotated[str, Form()] = "",
    model: Annotated[str, Form()] = "dpt-2-latest",
    student_label: Annotated[str, Form()] = "",
) -> RubricEvaluateResponse:
    """Two-stage pipeline: extract markdown → score against the named rubric.

    Returns per-question scores, total marks, percentage, and letter grade.
    """
    client_ip = request.client.host if request.client else "unknown"
    _limiter.check(client_ip, is_local=use_local)
    request_id: str = getattr(request.state, "request_id", "")

    doc_bytes = await file.read()
    filename = file.filename or "document"

    try:
        from api.services.extraction import run as extract_run

        extraction = extract_run(
            doc_bytes, filename, use_local=use_local, api_key=api_key or None, model=model
        )
        markdown = extraction["markdown"]
        scored = grader.score_against_rubric(
            markdown=markdown,
            rubric_id=rubric_id,
            student_label=student_label or None,
        )
    except ValueError as exc:
        status = 404 if "rubric" in str(exc).lower() else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RubricEvaluateResponse(**scored, request_id=request_id)
