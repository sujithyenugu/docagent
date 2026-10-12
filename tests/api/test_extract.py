"""Tests for POST /api/v1/extract.

All extraction calls are mocked — no local model or Landing AI API required.
"""

import io
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app

# Minimal valid PNG bytes (1×1 pixel)
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

_MOCK_RESULT = {
    "markdown": "# Invoice\n**Total**: ₹1,000",
    "page_count": 1,
    "duration_ms": 350,
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


def _upload_file(filename: str = "doc.png", content: bytes = _PNG_BYTES):
    return ("file", (filename, io.BytesIO(content), "image/png"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractLocalSuccess:
    @pytest.mark.anyio
    async def test_returns_200(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_RESULT):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_response_contains_markdown(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_RESULT):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert "markdown" in response.json()
        assert "Invoice" in response.json()["markdown"]

    @pytest.mark.anyio
    async def test_response_contains_metadata(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_RESULT):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        meta = response.json()["metadata"]
        assert meta["page_count"] == 1
        assert meta["backend"] == "local"
        assert "duration_ms" in meta
        assert "model" in meta

    @pytest.mark.anyio
    async def test_response_contains_request_id(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_RESULT):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert "request_id" in response.json()
        assert response.json()["request_id"] != ""

    @pytest.mark.anyio
    async def test_x_request_id_echoed_if_provided(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_RESULT):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
                headers={"X-Request-ID": "test-id-abc"},
            )
        assert response.headers.get("x-request-id") == "test-id-abc"
        assert response.json()["request_id"] == "test-id-abc"

    @pytest.mark.anyio
    async def test_x_data_retained_is_false(self, client):
        with patch("api.services.extraction.run_local", return_value=_MOCK_RESULT):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert response.headers.get("x-data-retained") == "false"


class TestExtractValidation:
    @pytest.mark.anyio
    async def test_unsupported_file_type_returns_400(self, client):
        with patch(
            "api.services.extraction.run_local",
            side_effect=ValueError("Unsupported file type '.tiff'"),
        ):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file("doc.tiff", b"\x00" * 20)],
                data={"use_local": "true"},
            )
        assert response.status_code == 400

    @pytest.mark.anyio
    async def test_cloud_without_api_key_returns_400(self, client):
        with patch(
            "api.services.extraction.run",
            side_effect=ValueError("api_key is required for cloud extraction."),
        ):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file()],
                data={"use_local": "false", "api_key": ""},
            )
        assert response.status_code == 400

    @pytest.mark.anyio
    async def test_backend_unreachable_returns_502(self, client):
        with patch(
            "api.services.extraction.run_local",
            side_effect=RuntimeError("Local model is not available"),
        ):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file()],
                data={"use_local": "true"},
            )
        assert response.status_code == 502


class TestExtractCloudSuccess:
    @pytest.mark.anyio
    async def test_cloud_backend_returns_200(self, client):
        cloud_result = {**_MOCK_RESULT, "backend": "cloud", "model": "dpt-2-mini"}
        with patch("api.services.extraction.run_cloud", return_value=cloud_result):
            response = await client.post(
                "/api/v1/extract",
                files=[_upload_file()],
                data={"use_local": "false", "api_key": "test-api-key"},
            )
        assert response.status_code == 200
        assert response.json()["metadata"]["backend"] == "cloud"
