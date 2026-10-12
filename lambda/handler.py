"""Lambda handler — DocumentAgent Phase 3 event-driven pipeline.

Triggered by S3 ObjectCreated on */input/* prefix.
Downloads the document, runs Landing AI ADE, uploads extracted markdown to
*/output/*, then syncs the Bedrock Knowledge Base.

Required environment variables:
  BEDROCK_KB_ID          — Knowledge Base ID
  BEDROCK_DS_ID          — Data Source ID within the KB
  AWS_REGION             — e.g. us-east-1

One of (in order of preference):
  LANDING_AI_SECRET_NAME — Secrets Manager secret name (recommended for prod)
  VISION_AGENT_API_KEY   — direct env var (local testing only)

Optional:
  ADE_MODEL              — 'dpt-2-latest' (default) or 'dpt-2-mini' (has returned
                           422 Unsupported model against current API keys — verify
                           before relying on it)
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import boto3
from landingai_ade import LandingAIADE

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_REGION = os.environ.get("AWS_REGION", "us-east-1")  # Lambda sets this automatically
_KB_ID = os.environ["BEDROCK_KB_ID"]
_DS_ID = os.environ["BEDROCK_DS_ID"]
_MODEL = os.getenv("ADE_MODEL", "dpt-2-latest")

_s3 = boto3.client("s3", region_name=_REGION)
_secrets = boto3.client("secretsmanager", region_name=_REGION)
_bedrock_agent = boto3.client("bedrock-agent", region_name=_REGION)


def _get_api_key() -> str:
    """Return Landing AI API key from Secrets Manager or env var fallback."""
    secret_name = os.getenv("LANDING_AI_SECRET_NAME")
    if secret_name:
        resp = _secrets.get_secret_value(SecretId=secret_name)
        return json.loads(resp["SecretString"])["VISION_AGENT_API_KEY"]
    key = os.getenv("VISION_AGENT_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Set LANDING_AI_SECRET_NAME (Secrets Manager) or VISION_AGENT_API_KEY."
        )
    return key


def _run_ade(doc_bytes: bytes, filename: str, api_key: str) -> str:
    """Write doc_bytes to a temp file, call ADE, return extracted markdown."""
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(doc_bytes)
        tmp_path = Path(tmp.name)
    try:
        client = LandingAIADE(apikey=api_key)
        response = client.parse(document=tmp_path, model=_MODEL)
        logger.info(
            "ADE complete: pages=%s credits=%s duration_ms=%s",
            response.metadata.page_count,
            response.metadata.credit_usage,
            response.metadata.duration_ms,
        )
        return response.markdown
    finally:
        tmp_path.unlink(missing_ok=True)


def handler(event, context):
    """Entry point — processes one S3 record at a time."""
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        # Expected key format: {uuid}/input/{filename}
        parts = key.split("/")
        if len(parts) < 3 or parts[1] != "input":
            logger.warning("Unexpected key format, skipping: %s", key)
            continue

        session_id = parts[0]
        filename = parts[2]
        logger.info("Processing session=%s file=%s bucket=%s", session_id, filename, bucket)

        # 1. Download document from S3
        obj = _s3.get_object(Bucket=bucket, Key=key)
        doc_bytes = obj["Body"].read()

        # 2. Run ADE extraction
        api_key = _get_api_key()
        markdown = _run_ade(doc_bytes, filename, api_key)

        # 3. Upload extracted markdown to output prefix
        output_key = f"{session_id}/output/extracted.md"
        _s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=markdown.encode("utf-8"),
            ContentType="text/markdown",
        )
        logger.info("Uploaded output → s3://%s/%s", bucket, output_key)

        # 4. Trigger Bedrock Knowledge Base ingestion
        _bedrock_agent.start_ingestion_job(
            knowledgeBaseId=_KB_ID,
            dataSourceId=_DS_ID,
        )
        logger.info("Started KB ingestion (kb=%s ds=%s)", _KB_ID, _DS_ID)

    return {"statusCode": 200}
