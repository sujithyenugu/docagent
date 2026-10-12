"""Unit tests for S3StorageBackend.

All boto3 calls are mocked — no real AWS credentials required.
Run: pytest tests/test_storage.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError

from storage import S3StorageBackend, build_backend, s3_enabled


PREFIX = "session-uuid-1234"
BUCKET = "test-bucket"
REGION = "us-east-1"


@pytest.fixture
def mock_s3_client():
    """Patch boto3.client so no real AWS calls are made."""
    with patch("storage.boto3.client") as mock_client_constructor:
        mock_client = MagicMock()
        mock_client_constructor.return_value = mock_client
        yield mock_client


@pytest.fixture
def backend(mock_s3_client):
    return S3StorageBackend(bucket=BUCKET, region=REGION, session_prefix=PREFIX)


# ---------------------------------------------------------------------------
# upload_document
# ---------------------------------------------------------------------------

class TestUploadDocument:
    def test_returns_correct_key(self, backend, mock_s3_client):
        key = backend.upload_document(b"pdf-bytes", "invoice.pdf")
        assert key == f"{PREFIX}/input/invoice.pdf"

    def test_calls_put_object_with_correct_args(self, backend, mock_s3_client):
        backend.upload_document(b"pdf-bytes", "invoice.pdf")
        mock_s3_client.put_object.assert_called_once_with(
            Bucket=BUCKET,
            Key=f"{PREFIX}/input/invoice.pdf",
            Body=b"pdf-bytes",
            ContentType="application/octet-stream",
        )

    def test_raises_on_client_error(self, backend, mock_s3_client):
        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": ""}}, "PutObject"
        )
        with pytest.raises(RuntimeError, match="S3 upload failed"):
            backend.upload_document(b"bytes", "file.pdf")


# ---------------------------------------------------------------------------
# upload_output
# ---------------------------------------------------------------------------

class TestUploadOutput:
    def test_returns_correct_key(self, backend, mock_s3_client):
        key = backend.upload_output("# Heading\ntext", "extracted.md")
        assert key == f"{PREFIX}/output/extracted.md"

    def test_encodes_markdown_as_utf8(self, backend, mock_s3_client):
        backend.upload_output("# Hello €", "out.md")
        call_kwargs = mock_s3_client.put_object.call_args.kwargs
        assert call_kwargs["Body"] == "# Hello €".encode("utf-8")
        assert call_kwargs["ContentType"] == "text/markdown"

    def test_raises_on_client_error(self, backend, mock_s3_client):
        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": ""}}, "PutObject"
        )
        with pytest.raises(RuntimeError, match="S3 upload failed"):
            backend.upload_output("text", "out.md")


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

class TestDownload:
    def test_returns_bytes(self, backend, mock_s3_client):
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b"file-content"))
        }
        result = backend.download(f"{PREFIX}/input/invoice.pdf")
        assert result == b"file-content"

    def test_calls_get_object_with_correct_args(self, backend, mock_s3_client):
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b""))
        }
        key = f"{PREFIX}/output/extracted.md"
        backend.download(key)
        mock_s3_client.get_object.assert_called_once_with(Bucket=BUCKET, Key=key)

    def test_raises_on_client_error(self, backend, mock_s3_client):
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject"
        )
        with pytest.raises(RuntimeError, match="S3 download failed"):
            backend.download("missing/key.pdf")


# ---------------------------------------------------------------------------
# build_backend / s3_enabled helpers
# ---------------------------------------------------------------------------

class TestBuildBackend:
    def test_raises_when_bucket_missing(self, mock_s3_client, monkeypatch):
        monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        with pytest.raises(RuntimeError, match="S3_BUCKET_NAME"):
            build_backend("some-prefix")

    def test_raises_when_region_missing(self, mock_s3_client, monkeypatch):
        monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")
        monkeypatch.delenv("AWS_REGION", raising=False)
        with pytest.raises(RuntimeError, match="AWS_REGION"):
            build_backend("some-prefix")

    def test_returns_backend_with_correct_attrs(self, mock_s3_client, monkeypatch):
        monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        b = build_backend("abc-123")
        assert b.bucket == "my-bucket"
        assert b.region == "eu-west-1"
        assert b.session_prefix == "abc-123"


class TestS3Enabled:
    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("S3_STORAGE_ENABLED", val)
        assert s3_enabled() is True

    @pytest.mark.parametrize("val", ["false", "0", "no", ""])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("S3_STORAGE_ENABLED", val)
        assert s3_enabled() is False

    def test_default_is_false(self, monkeypatch):
        monkeypatch.delenv("S3_STORAGE_ENABLED", raising=False)
        assert s3_enabled() is False
