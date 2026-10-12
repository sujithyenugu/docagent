"""POST /api/v1/extract — multipart document upload → markdown extraction."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.middleware.rate_limit import RateLimiter
from api.schemas.extract import ExtractResponse, ExtractionMetadata
from api.services import extraction as extraction_svc

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extract"])

_limiter = RateLimiter(local_rpm=10, cloud_rpm=1)

_PRIVACY_HEADERS = {"X-Data-Retained": "false"}


@router.post(
    "/extract",
    response_model=ExtractResponse,
    responses={
        429: {"description": "Rate limit exceeded"},
        400: {"description": "Bad request (unsupported file type, missing API key)"},
        502: {"description": "Backend not reachable"},
    },
)
async def extract(
    request: Request,
    file: Annotated[UploadFile, File(description="Document to extract (PDF/PNG/JPEG)")],
    use_local: Annotated[bool, Form()] = True,
    api_key: Annotated[str, Form()] = "",
    model: Annotated[str, Form()] = "dpt-2-latest",
) -> ExtractResponse:
    """Extract text from an uploaded document.

    Multipart form fields:
    - **file**: Document file (required).
    - **use_local**: ``true`` → embedded local MLX model; ``false`` → Landing AI ADE (default: true).
    - **api_key**: Landing AI API key (required when ``use_local=false``).
    - **model**: ADE model variant — ``dpt-2-latest`` | ``dpt-2-mini`` (default: dpt-2-latest;
      ``dpt-2-mini`` has returned ``422 Unsupported model`` against current API keys — verify
      before relying on it).

    Response headers:
    - ``X-Request-ID``: Unique request identifier.
    - ``X-Data-Retained``: Always ``false`` — documents are never persisted by the API.
    """
    client_ip = request.client.host if request.client else "unknown"
    _limiter.check(client_ip, is_local=use_local)

    request_id: str = getattr(request.state, "request_id", "")

    doc_bytes = await file.read()
    filename = file.filename or "document"

    try:
        result = extraction_svc.run(
            doc_bytes,
            filename,
            use_local=use_local,
            api_key=api_key or None,
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ExtractResponse(
        markdown=result["markdown"],
        metadata=ExtractionMetadata(
            page_count=result["page_count"],
            duration_ms=result["duration_ms"],
            model=result["model"],
            backend=result["backend"],
        ),
        request_id=request_id,
    )
