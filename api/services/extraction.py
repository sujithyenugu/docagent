"""Extraction service — thin adapter around LocalModelClient and LandingAIADE.

Both backends share the same interface: accept ``doc_bytes`` + ``filename``,
write to a temporary file, call the backend, clean up, return a plain dict so
FastAPI routers stay decoupled from the backend SDKs.

Returned dict shape:
    {
        "markdown": str,
        "page_count": int,
        "duration_ms": int,
        "model": str,
        "backend": "local" | "cloud",
    }
"""

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}


def _validate_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_SUFFIXES))}"
        )
    return suffix


def run_local(doc_bytes: bytes, filename: str, repo_id: str | None = None) -> dict:
    """Extract text using the embedded local MLX model.

    Args:
        doc_bytes: Raw file bytes.
        filename:  Original filename (used to determine extension).
        repo_id:   Optional Hugging Face repo id to override the configured
                   default model for this call only (e.g. for benchmarking
                   multiple local models against the same dataset).

    Returns:
        Extraction result dict (see module docstring).

    Raises:
        ValueError:   Unsupported file extension.
        RuntimeError: Local model unavailable (unsupported platform or
                      first-run download failed).
    """
    from local_model import LocalModelClient

    suffix = _validate_suffix(filename)
    client = LocalModelClient(repo_id=repo_id)
    model_name = client.model_id

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="api_local_"
        ) as tmp:
            tmp.write(doc_bytes)
            tmp_path = Path(tmp.name)

        response = client.parse(document=tmp_path)
        logger.info(
            "local extract | file=%s pages=%s duration_ms=%s",
            filename,
            response.metadata.page_count,
            response.metadata.duration_ms,
        )
        return {
            "markdown": response.markdown,
            "page_count": response.metadata.page_count,
            "duration_ms": response.metadata.duration_ms,
            "model": model_name,
            "backend": "local",
        }
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def run_cloud(doc_bytes: bytes, filename: str, api_key: str, model: str = "dpt-2-latest") -> dict:
    """Extract text using the Landing AI ADE cloud API.

    Args:
        doc_bytes: Raw file bytes.
        filename:  Original filename.
        api_key:   Landing AI API key.
        model:     ADE model variant ('dpt-2-latest' or 'dpt-2-mini' — the latter has
                   returned 422 Unsupported model against current API keys).

    Returns:
        Extraction result dict (see module docstring).

    Raises:
        ValueError:   Unsupported file extension.
        RuntimeError: API error or missing key.
    """
    from landingai_ade import LandingAIADE

    suffix = _validate_suffix(filename)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="api_cloud_"
        ) as tmp:
            tmp.write(doc_bytes)
            tmp_path = Path(tmp.name)

        client = LandingAIADE(apikey=api_key)
        response = client.parse(document=tmp_path, model=model)
        logger.info(
            "cloud extract | file=%s model=%s pages=%s credits=%s duration_ms=%s",
            filename,
            model,
            response.metadata.page_count,
            getattr(response.metadata, "credit_usage", "n/a"),
            response.metadata.duration_ms,
        )
        return {
            "markdown": response.markdown,
            "page_count": response.metadata.page_count,
            "duration_ms": response.metadata.duration_ms,
            "model": model,
            "backend": "cloud",
        }
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def run(
    doc_bytes: bytes,
    filename: str,
    *,
    use_local: bool = True,
    api_key: str | None = None,
    model: str = "dpt-2-latest",
) -> dict:
    """Dispatch to local or cloud backend.

    Args:
        doc_bytes:  Raw file bytes.
        filename:   Original filename.
        use_local:  If True use the embedded local MLX model; if False use Landing AI ADE.
        api_key:    Required when ``use_local=False``.
        model:      ADE model variant (ignored for local).

    Raises:
        ValueError: ``use_local=False`` but ``api_key`` not provided.
    """
    if use_local:
        return run_local(doc_bytes, filename)
    if not api_key:
        raise ValueError("api_key is required for cloud extraction.")
    return run_cloud(doc_bytes, filename, api_key=api_key, model=model)
