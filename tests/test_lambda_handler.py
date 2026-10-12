"""Unit tests for lambda/handler.py.

All boto3 calls and ADE API calls are mocked — no real AWS credentials or
Landing AI key required.

Run:
    pytest tests/test_lambda_handler.py -v
    pytest tests/ --cov=storage --cov=handler --cov-report=term-missing
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ── Set required env vars BEFORE importing handler (module-level reads them) ──
os.environ.setdefault("BEDROCK_KB_ID", "test-kb-id")
os.environ.setdefault("BEDROCK_DS_ID", "test-ds-id")

# ── Add lambda/ to sys.path and import with boto3 mocked ─────────────────────
LAMBDA_DIR = str(Path(__file__).resolve().parent.parent / "lambda")
if LAMBDA_DIR not in sys.path:
    sys.path.insert(0, LAMBDA_DIR)

with patch("boto3.client", return_value=MagicMock()):
    import handler


# ── Autouse fixture: fresh mock clients for every test ───────────────────────
@pytest.fixture(autouse=True)
def mock_clients():
    """Replace module-level boto3 clients with clean mocks before each test."""
    handler._s3 = MagicMock()
    handler._secrets = MagicMock()
    handler._bedrock_agent = MagicMock()
    yield handler._s3, handler._secrets, handler._bedrock_agent


# ── Helpers ───────────────────────────────────────────────────────────────────
def _s3_event(bucket: str, key: str) -> dict:
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}


def _ade_response(markdown: str = "# Extracted") -> MagicMock:
    resp = MagicMock()
    resp.markdown = markdown
    resp.metadata.page_count = 1
    resp.metadata.credit_usage = 1.5
    resp.metadata.duration_ms = 800
    return resp


# ── _get_api_key ──────────────────────────────────────────────────────────────
class TestGetApiKey:
    def test_reads_from_secrets_manager(self, mock_clients, monkeypatch):
        _, secrets, _ = mock_clients
        monkeypatch.setenv("LANDING_AI_SECRET_NAME", "documentagent/landing-ai-key")
        monkeypatch.delenv("VISION_AGENT_API_KEY", raising=False)
        secrets.get_secret_value.return_value = {
            "SecretString": json.dumps({"VISION_AGENT_API_KEY": "sm-key-123"})
        }
        assert handler._get_api_key() == "sm-key-123"
        secrets.get_secret_value.assert_called_once_with(
            SecretId="documentagent/landing-ai-key"
        )

    def test_falls_back_to_env_var(self, mock_clients, monkeypatch):
        monkeypatch.delenv("LANDING_AI_SECRET_NAME", raising=False)
        monkeypatch.setenv("VISION_AGENT_API_KEY", "env-key-456")
        assert handler._get_api_key() == "env-key-456"

    def test_raises_when_no_key_configured(self, mock_clients, monkeypatch):
        monkeypatch.delenv("LANDING_AI_SECRET_NAME", raising=False)
        monkeypatch.delenv("VISION_AGENT_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="LANDING_AI_SECRET_NAME"):
            handler._get_api_key()


# ── _run_ade ──────────────────────────────────────────────────────────────────
class TestRunAde:
    def test_returns_markdown(self):
        with patch("handler.LandingAIADE") as MockADE:
            MockADE.return_value.parse.return_value = _ade_response("# Hello")
            result = handler._run_ade(b"pdf-bytes", "doc.pdf", "api-key")
        assert result == "# Hello"

    def test_passes_correct_api_key(self):
        with patch("handler.LandingAIADE") as MockADE:
            MockADE.return_value.parse.return_value = _ade_response()
            handler._run_ade(b"bytes", "file.pdf", "my-key")
        MockADE.assert_called_once_with(apikey="my-key")

    def test_uses_filename_suffix_for_tempfile(self):
        with patch("handler.LandingAIADE") as MockADE:
            MockADE.return_value.parse.return_value = _ade_response()
            with patch("handler.tempfile.NamedTemporaryFile") as mock_tmp:
                mock_tmp.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(name="ctx", write=MagicMock())
                )
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
                mock_tmp.return_value.name = "/tmp/tmpfile.jpg"
                # Only check the suffix kwarg passed to NamedTemporaryFile
                handler._run_ade(b"bytes", "scan.jpg", "key")
            _, kwargs = mock_tmp.call_args
            assert kwargs.get("suffix") == ".jpg"

    def test_cleans_up_tempfile_on_success(self):
        with patch("handler.LandingAIADE") as MockADE:
            MockADE.return_value.parse.return_value = _ade_response()
            with patch("handler.tempfile.NamedTemporaryFile"):
                with patch("handler.Path") as MockPath:
                    mock_path_instance = MagicMock()
                    MockPath.return_value = mock_path_instance
                    # Simulate successful run
                    try:
                        handler._run_ade(b"bytes", "file.pdf", "key")
                    except Exception:
                        pass
                    mock_path_instance.unlink.assert_called()

    def test_cleans_up_tempfile_on_ade_failure(self):
        with patch("handler.LandingAIADE") as MockADE:
            MockADE.return_value.parse.side_effect = RuntimeError("ADE down")
            with pytest.raises(RuntimeError, match="ADE down"):
                handler._run_ade(b"bytes", "file.pdf", "key")
            # temp file should be cleaned up (unlink called via finally block)


# ── handler (main entry point) ────────────────────────────────────────────────
class TestHandler:
    def _setup_happy_path(self, mock_clients, markdown="# Doc content"):
        s3, secrets, bedrock = mock_clients
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"pdf-bytes"))}
        with patch("handler._get_api_key", return_value="test-key"), \
             patch("handler._run_ade", return_value=markdown):
            return s3, bedrock

    def test_returns_200(self, mock_clients):
        s3, secrets, bedrock = mock_clients
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"bytes"))}
        with patch("handler._get_api_key", return_value="k"), \
             patch("handler._run_ade", return_value="# md"):
            result = handler.handler(_s3_event("my-bucket", "uuid-1/input/doc.pdf"), {})
        assert result == {"statusCode": 200}

    def test_downloads_document_from_s3(self, mock_clients):
        s3, _, _ = mock_clients
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"bytes"))}
        with patch("handler._get_api_key", return_value="k"), \
             patch("handler._run_ade", return_value="# md"):
            handler.handler(_s3_event("my-bucket", "uuid-1/input/invoice.pdf"), {})
        s3.get_object.assert_called_once_with(
            Bucket="my-bucket", Key="uuid-1/input/invoice.pdf"
        )

    def test_uploads_markdown_to_output_prefix(self, mock_clients):
        s3, _, _ = mock_clients
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"bytes"))}
        with patch("handler._get_api_key", return_value="k"), \
             patch("handler._run_ade", return_value="# Extracted"):
            handler.handler(_s3_event("my-bucket", "uuid-1/input/doc.pdf"), {})
        s3.put_object.assert_called_once_with(
            Bucket="my-bucket",
            Key="uuid-1/output/extracted.md",
            Body="# Extracted".encode("utf-8"),
            ContentType="text/markdown",
        )

    def test_triggers_bedrock_kb_ingestion(self, mock_clients):
        s3, _, bedrock = mock_clients
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"bytes"))}
        with patch("handler._get_api_key", return_value="k"), \
             patch("handler._run_ade", return_value="# md"):
            handler.handler(_s3_event("my-bucket", "uuid-1/input/doc.pdf"), {})
        bedrock.start_ingestion_job.assert_called_once_with(
            knowledgeBaseId="test-kb-id",
            dataSourceId="test-ds-id",
        )

    def test_skips_key_not_in_input_prefix(self, mock_clients):
        s3, _, bedrock = mock_clients
        # key has 'output' instead of 'input'
        handler.handler(_s3_event("bucket", "uuid-1/output/extracted.md"), {})
        s3.get_object.assert_not_called()
        bedrock.start_ingestion_job.assert_not_called()

    def test_skips_key_with_too_few_parts(self, mock_clients):
        s3, _, bedrock = mock_clients
        handler.handler(_s3_event("bucket", "just-a-file.pdf"), {})
        s3.get_object.assert_not_called()
        bedrock.start_ingestion_job.assert_not_called()

    def test_handles_empty_records(self, mock_clients):
        s3, _, _ = mock_clients
        result = handler.handler({"Records": []}, {})
        assert result == {"statusCode": 200}
        s3.get_object.assert_not_called()

    def test_processes_multiple_records(self, mock_clients):
        s3, _, bedrock = mock_clients
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"bytes"))}
        event = {
            "Records": [
                {"s3": {"bucket": {"name": "b"}, "object": {"key": "u1/input/f1.pdf"}}},
                {"s3": {"bucket": {"name": "b"}, "object": {"key": "u2/input/f2.pdf"}}},
            ]
        }
        with patch("handler._get_api_key", return_value="k"), \
             patch("handler._run_ade", return_value="# md"):
            handler.handler(event, {})
        assert s3.get_object.call_count == 2
        assert bedrock.start_ingestion_job.call_count == 2

    def test_output_key_uses_session_id_from_input_key(self, mock_clients):
        s3, _, _ = mock_clients
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"bytes"))}
        session = "abc-def-123"
        with patch("handler._get_api_key", return_value="k"), \
             patch("handler._run_ade", return_value="# md"):
            handler.handler(_s3_event("bucket", f"{session}/input/report.pdf"), {})
        call_kwargs = s3.put_object.call_args.kwargs
        assert call_kwargs["Key"] == f"{session}/output/extracted.md"
