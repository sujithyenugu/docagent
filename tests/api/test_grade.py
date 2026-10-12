"""Tests for /api/v1/grade/* endpoints.

All extraction and LLM calls are mocked so no local model or API key is needed.
"""

import io
import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app

_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

_MOCK_EXTRACTION = {
    "markdown": (
        "Photosynthesis is a process by which plants produce food using sunlight. "
        "Chlorophyll absorbs light energy. The products are glucose and oxygen."
    ),
    "page_count": 1,
    "duration_ms": 500,
    "model": "gemma-3-4b-instruct",
    "backend": "local",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _upload_file(filename: str = "answer_sheet.jpg", content: bytes = _PNG_BYTES):
    return ("file", (filename, io.BytesIO(content), "image/jpeg"))


# ---------------------------------------------------------------------------
# POST /grade/extract
# ---------------------------------------------------------------------------

class TestGradeExtract:
    @pytest.mark.anyio
    async def test_returns_200_without_keywords(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_markdown_in_response(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/extract",
                files=[_upload_file()],
                data={"use_local": "true", "keywords": "photosynthesis,chlorophyll"},
            )
        body = response.json()
        assert "markdown" in body
        assert "photosynthesis" in body["markdown"].lower()

    @pytest.mark.anyio
    async def test_keyword_matches_returned(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/extract",
                files=[_upload_file()],
                data={"use_local": "true", "keywords": "photosynthesis,chlorophyll,nitrogen"},
            )
        body = response.json()
        matches = {m["keyword"]: m["found"] for m in body["keyword_matches"]}
        assert matches["photosynthesis"] is True
        assert matches["chlorophyll"] is True
        assert matches["nitrogen"] is False

    @pytest.mark.anyio
    async def test_answer_key_scoring(self, client):
        answer_key = json.dumps([
            {
                "question_id": "q1",
                "max_marks": 4,
                "keywords": ["photosynthesis", "chlorophyll"],
                "partial_keywords": ["sunlight"],
            }
        ])
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/extract",
                files=[_upload_file()],
                data={"use_local": "true", "answer_key": answer_key},
            )
        body = response.json()
        assert body["total_awarded"] is not None
        assert body["total_possible"] == 4.0
        assert len(body["question_scores"]) == 1

    @pytest.mark.anyio
    async def test_invalid_answer_key_json_returns_400(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/extract",
                files=[_upload_file()],
                data={"use_local": "true", "answer_key": "not-valid-json"},
            )
        assert response.status_code == 400

    @pytest.mark.anyio
    async def test_student_label_in_response(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/extract",
                files=[_upload_file()],
                data={"use_local": "true", "student_label": "Priya Sharma / Roll 42"},
            )
        body = response.json()
        assert body["student_label"] == "Priya Sharma / Roll 42"

    @pytest.mark.anyio
    async def test_x_data_retained_header(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert response.headers.get("x-data-retained") == "false"

    @pytest.mark.anyio
    async def test_backend_unreachable_returns_502(self, client):
        with patch(
            "api.services.extraction.run_local",
            side_effect=RuntimeError("Local model is not available"),
        ):
            response = await client.post(
                "/api/v1/grade/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert response.status_code == 502


# ---------------------------------------------------------------------------
# GET /grade/rubrics
# ---------------------------------------------------------------------------

class TestListRubrics:
    @pytest.mark.anyio
    async def test_returns_200(self, client):
        response = await client.get("/api/v1/grade/rubrics")
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_returns_list(self, client):
        response = await client.get("/api/v1/grade/rubrics")
        assert isinstance(response.json(), list)
        assert len(response.json()) > 0

    @pytest.mark.anyio
    async def test_rubric_has_required_keys(self, client):
        response = await client.get("/api/v1/grade/rubrics")
        for rubric in response.json():
            assert "rubric_id" in rubric
            assert "name" in rubric
            assert "subject" in rubric
            assert "grade_level" in rubric
            assert "total_marks" in rubric
            assert "question_count" in rubric

    @pytest.mark.anyio
    async def test_known_rubric_in_list(self, client):
        response = await client.get("/api/v1/grade/rubrics")
        ids = [r["rubric_id"] for r in response.json()]
        assert "science_class8_ch1" in ids
        assert "maths_class10_algebra" in ids


# ---------------------------------------------------------------------------
# GET /grade/rubrics/{rubric_id}
# ---------------------------------------------------------------------------

class TestGetRubric:
    @pytest.mark.anyio
    async def test_known_rubric_returns_200(self, client):
        response = await client.get("/api/v1/grade/rubrics/science_class8_ch1")
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_rubric_has_questions(self, client):
        response = await client.get("/api/v1/grade/rubrics/science_class8_ch1")
        body = response.json()
        assert "questions" in body
        assert len(body["questions"]) == 5

    @pytest.mark.anyio
    async def test_unknown_rubric_returns_404(self, client):
        response = await client.get("/api/v1/grade/rubrics/does_not_exist")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /grade/rubrics/{rubric_id}/evaluate
# ---------------------------------------------------------------------------

class TestRubricEvaluate:
    @pytest.mark.anyio
    async def test_returns_200(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/rubrics/science_class8_ch1/evaluate",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_response_has_scores(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/rubrics/science_class8_ch1/evaluate",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        body = response.json()
        assert "question_scores" in body
        assert "total_awarded" in body
        assert "percentage" in body
        assert "grade" in body

    @pytest.mark.anyio
    async def test_grade_is_letter(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/rubrics/science_class8_ch1/evaluate",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert response.json()["grade"] in {"A+", "A", "B", "C", "D", "F"}

    @pytest.mark.anyio
    async def test_unknown_rubric_returns_404(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/rubrics/no_such_rubric/evaluate",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_student_label_in_response(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_EXTRACTION):
            response = await client.post(
                "/api/v1/grade/rubrics/science_class8_ch1/evaluate",
                files=[_upload_file()],
                data={"use_local": "true", "student_label": "Roll-007"},
            )
        assert response.json()["student_label"] == "Roll-007"


# ---------------------------------------------------------------------------
# Grading templates in /forms/templates
# ---------------------------------------------------------------------------

class TestGradingTemplatesInForms:
    @pytest.mark.anyio
    async def test_grading_templates_present(self, client):
        response = await client.get("/api/v1/forms/templates")
        ids = [t["template_id"] for t in response.json()]
        assert "cbse_marksheet" in ids
        assert "icse_marksheet" in ids
        assert "school_progress_report" in ids
        assert "admit_card_verification" in ids

    @pytest.mark.anyio
    async def test_grading_templates_have_use_case_grading(self, client):
        response = await client.get("/api/v1/forms/templates")
        grading = [t for t in response.json() if t.get("use_case") == "grading"]
        assert len(grading) >= 4

    @pytest.mark.anyio
    async def test_admin_templates_have_use_case_admin(self, client):
        response = await client.get("/api/v1/forms/templates")
        admin = [t for t in response.json() if t.get("use_case") == "admin"]
        ids = [t["template_id"] for t in admin]
        assert "lease_agreement" in ids
