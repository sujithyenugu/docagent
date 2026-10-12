"""Tests for GET /api/v1/health."""

import pytest
from httpx import ASGITransport, AsyncClient

import local_model
from api.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    @pytest.mark.anyio
    async def test_returns_200(self, client):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_response_has_required_keys(self, client):
        response = await client.get("/api/v1/health")
        body = response.json()
        assert "status" in body
        assert "local_model_available" in body
        assert "local_model_downloaded" in body
        assert "cloud_api_configured" in body

    @pytest.mark.anyio
    async def test_status_is_ok(self, client):
        response = await client.get("/api/v1/health")
        assert response.json()["status"] == "ok"

    @pytest.mark.anyio
    async def test_local_model_available_false_on_unsupported_platform(self, client, monkeypatch):
        monkeypatch.setattr(local_model, "is_platform_supported", lambda: False)
        response = await client.get("/api/v1/health")
        assert response.json()["local_model_available"] is False
        assert response.json()["local_model_downloaded"] is False

    @pytest.mark.anyio
    async def test_local_model_available_true_on_supported_platform(self, client, monkeypatch):
        monkeypatch.setattr(local_model, "is_platform_supported", lambda: True)
        monkeypatch.setattr(local_model, "is_model_ready", lambda *a, **kw: False)
        response = await client.get("/api/v1/health")
        assert response.json()["local_model_available"] is True

    @pytest.mark.anyio
    async def test_local_model_downloaded_true_when_weights_cached(self, client, monkeypatch):
        monkeypatch.setattr(local_model, "is_platform_supported", lambda: True)
        monkeypatch.setattr(local_model, "is_model_ready", lambda *a, **kw: True)
        response = await client.get("/api/v1/health")
        assert response.json()["local_model_downloaded"] is True

    @pytest.mark.anyio
    async def test_cloud_api_configured_false_when_no_key(self, client, monkeypatch):
        monkeypatch.delenv("VISION_AGENT_API_KEY", raising=False)
        response = await client.get("/api/v1/health")
        assert response.json()["cloud_api_configured"] is False

    @pytest.mark.anyio
    async def test_cloud_api_configured_true_when_key_set(self, client, monkeypatch):
        monkeypatch.setenv("VISION_AGENT_API_KEY", "test-key-123")
        response = await client.get("/api/v1/health")
        assert response.json()["cloud_api_configured"] is True

    @pytest.mark.anyio
    async def test_response_has_x_request_id_header(self, client):
        response = await client.get("/api/v1/health")
        assert "x-request-id" in response.headers

    @pytest.mark.anyio
    async def test_response_has_x_data_retained_header(self, client):
        response = await client.get("/api/v1/health")
        assert response.headers.get("x-data-retained") == "false"
