"""Tests for POST /api/v1/forms/fill and GET /api/v1/forms/templates.

All extraction and LLM calls are mocked.
"""

import io
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.services.form_filler import FormFillResult

_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

_MOCK_FILL_RESULT = FormFillResult(
    fields={
        "tenant_name": "Ramesh Kumar",
        "landlord_name": "Suresh Patel",
        "property_address": "42 MG Road, Bengaluru 560001",
        "monthly_rent": "15000",
        "start_date": "01/07/2026",
        "duration_months": "11",
        "security_deposit": None,
    },
    field_sources={
        "tenant_name": "aadhaar_front.jpg",
        "landlord_name": "aadhaar_front.jpg",
        "property_address": "aadhaar_front.jpg",
        "monthly_rent": "aadhaar_front.jpg",
        "start_date": "aadhaar_front.jpg",
        "duration_months": "aadhaar_front.jpg",
    },
    missing_fields=[],
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _upload_file(filename: str = "aadhaar_front.jpg", content: bytes = _PNG_BYTES):
    return ("files", (filename, io.BytesIO(content), "image/jpeg"))


# ---------------------------------------------------------------------------
# Tests: form fill
# ---------------------------------------------------------------------------

class TestFormFillSuccess:
    @pytest.mark.anyio
    async def test_returns_200(self, client):
        with patch("api.services.form_filler.fill", return_value=_MOCK_FILL_RESULT):
            response = await client.post(
                "/api/v1/forms/fill",
                files=[_upload_file()],
                data={"template_id": "lease_agreement", "use_local": "true"},
            )
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_response_has_fields(self, client):
        with patch("api.services.form_filler.fill", return_value=_MOCK_FILL_RESULT):
            response = await client.post(
                "/api/v1/forms/fill",
                files=[_upload_file()],
                data={"template_id": "lease_agreement", "use_local": "true"},
            )
        body = response.json()
        assert "fields" in body
        assert body["fields"]["tenant_name"] == "Ramesh Kumar"

    @pytest.mark.anyio
    async def test_response_has_field_sources(self, client):
        with patch("api.services.form_filler.fill", return_value=_MOCK_FILL_RESULT):
            response = await client.post(
                "/api/v1/forms/fill",
                files=[_upload_file()],
                data={"template_id": "lease_agreement", "use_local": "true"},
            )
        body = response.json()
        assert "field_sources" in body
        assert body["field_sources"]["tenant_name"] == "aadhaar_front.jpg"

    @pytest.mark.anyio
    async def test_response_has_missing_fields(self, client):
        with patch("api.services.form_filler.fill", return_value=_MOCK_FILL_RESULT):
            response = await client.post(
                "/api/v1/forms/fill",
                files=[_upload_file()],
                data={"template_id": "lease_agreement", "use_local": "true"},
            )
        assert response.json()["missing_fields"] == []

    @pytest.mark.anyio
    async def test_response_has_request_id(self, client):
        with patch("api.services.form_filler.fill", return_value=_MOCK_FILL_RESULT):
            response = await client.post(
                "/api/v1/forms/fill",
                files=[_upload_file()],
                data={"template_id": "lease_agreement", "use_local": "true"},
            )
        assert "request_id" in response.json()

    @pytest.mark.anyio
    async def test_x_data_retained_is_false(self, client):
        with patch("api.services.form_filler.fill", return_value=_MOCK_FILL_RESULT):
            response = await client.post(
                "/api/v1/forms/fill",
                files=[_upload_file()],
                data={"template_id": "lease_agreement", "use_local": "true"},
            )
        assert response.headers.get("x-data-retained") == "false"

    @pytest.mark.anyio
    async def test_missing_fields_reported(self, client):
        partial_result = FormFillResult(
            fields={"tenant_name": "Ramesh Kumar", "landlord_name": None,
                    "property_address": None, "monthly_rent": None},
            field_sources={"tenant_name": "aadhaar_front.jpg"},
            missing_fields=["landlord_name", "property_address", "monthly_rent"],
        )
        with patch("api.services.form_filler.fill", return_value=partial_result):
            response = await client.post(
                "/api/v1/forms/fill",
                files=[_upload_file()],
                data={"template_id": "lease_agreement", "use_local": "true"},
            )
        assert "landlord_name" in response.json()["missing_fields"]


class TestFormFillValidation:
    @pytest.mark.anyio
    async def test_unknown_template_returns_400(self, client):
        with patch(
            "api.services.form_filler.fill",
            side_effect=ValueError("Unknown template 'bad_template'"),
        ):
            response = await client.post(
                "/api/v1/forms/fill",
                files=[_upload_file()],
                data={"template_id": "bad_template", "use_local": "true"},
            )
        assert response.status_code == 400

    @pytest.mark.anyio
    async def test_backend_unreachable_returns_502(self, client):
        with patch(
            "api.services.form_filler.fill",
            side_effect=RuntimeError("Local model is not available"),
        ):
            response = await client.post(
                "/api/v1/forms/fill",
                files=[_upload_file()],
                data={"template_id": "lease_agreement", "use_local": "true"},
            )
        assert response.status_code == 502


# ---------------------------------------------------------------------------
# Tests: list templates
# ---------------------------------------------------------------------------

class TestListTemplates:
    @pytest.mark.anyio
    async def test_returns_200(self, client):
        response = await client.get("/api/v1/forms/templates")
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_returns_list(self, client):
        response = await client.get("/api/v1/forms/templates")
        assert isinstance(response.json(), list)

    @pytest.mark.anyio
    async def test_contains_expected_templates(self, client):
        response = await client.get("/api/v1/forms/templates")
        ids = [t["template_id"] for t in response.json()]
        assert "lease_agreement" in ids
        assert "rent_receipt" in ids
        assert "dl_renewal" in ids

    @pytest.mark.anyio
    async def test_each_template_has_required_fields_key(self, client):
        response = await client.get("/api/v1/forms/templates")
        for tmpl in response.json():
            assert "required_fields" in tmpl
            assert isinstance(tmpl["required_fields"], list)
