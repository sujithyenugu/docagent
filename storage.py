"""S3 storage backend for DocumentAgent.

Disabled by default. Enable via S3_STORAGE_ENABLED=true in .env or the UI toggle.
Documents are stored under a per-session UUID prefix to avoid collisions.
"""

import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class S3StorageBackend:
    """Handles upload and download of documents and extracted outputs to/from S3."""

    def __init__(self, bucket: str, region: str, session_prefix: str):
        """
        Args:
            bucket: S3 bucket name.
            region: AWS region (e.g. 'us-east-1').
            session_prefix: Per-session UUID prefix to namespace all objects.
        """
        self.bucket = bucket
        self.region = region
        self.session_prefix = session_prefix
        self._client = boto3.client("s3", region_name=region)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_document(self, data: bytes, filename: str) -> str:
        """Upload raw document bytes to S3.

        Returns the S3 object key on success.
        Raises RuntimeError on failure.
        """
        key = f"{self.session_prefix}/input/{filename}"
        self._put(key, data)
        logger.info("Uploaded document to s3://%s/%s", self.bucket, key)
        return key

    def upload_output(self, markdown: str, filename: str = "extracted.md") -> str:
        """Upload extracted markdown text to S3.

        Returns the S3 object key on success.
        Raises RuntimeError on failure.
        """
        key = f"{self.session_prefix}/output/{filename}"
        self._put(key, markdown.encode("utf-8"), content_type="text/markdown")
        logger.info("Uploaded output to s3://%s/%s", self.bucket, key)
        return key

    def download(self, key: str) -> bytes:
        """Download an object from S3 by key.

        Returns raw bytes on success.
        Raises RuntimeError on failure.
        """
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            data: bytes = resp["Body"].read()
            logger.info("Downloaded s3://%s/%s (%d bytes)", self.bucket, key, len(data))
            return data
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"S3 download failed for key '{key}': {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _put(self, key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"S3 upload failed for key '{key}': {exc}") from exc


def s3_enabled() -> bool:
    """Return True if S3 storage is enabled via environment variable."""
    return os.getenv("S3_STORAGE_ENABLED", "false").lower() in {"true", "1", "yes"}


def build_backend(session_prefix: str) -> S3StorageBackend:
    """Construct an S3StorageBackend from environment variables.

    Required env vars: S3_BUCKET_NAME, AWS_REGION
    Optional: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (falls back to IAM role)

    Raises RuntimeError if required variables are missing.
    """
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    region = os.getenv("AWS_REGION", "").strip()

    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME environment variable is not set.")
    if not region:
        raise RuntimeError("AWS_REGION environment variable is not set.")

    return S3StorageBackend(bucket=bucket, region=region, session_prefix=session_prefix)
