"""POST /api/v1/forms/fill — multi-document form-fill pipeline."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.middleware.rate_limit import RateLimiter
from api.schemas.forms import FormFillResponse
from api.services import form_filler

logger = logging.getLogger(__name__)

router = APIRouter(tags=["forms"])

_limiter = RateLimiter(local_rpm=10, cloud_rpm=1)


@router.post(
    "/forms/fill",
    response_model=FormFillResponse,
    responses={
        429: {"description": "Rate limit exceeded"},
        400: {"description": "Bad request (unknown template, unsupported file, missing key)"},
        502: {"description": "Backend not reachable"},
    },
)
async def forms_fill(
    request: Request,
    files: Annotated[
        list[UploadFile],
        File(description="One or more ID documents or source documents"),
    ],
    template_id: Annotated[
        str,
        Form(description="Form template to fill — e.g. 'lease_agreement', 'rent_receipt'"),
    ],
    use_local: Annotated[bool, Form()] = True,
    api_key: Annotated[str, Form()] = "",
    model: Annotated[str, Form()] = "dpt-2-latest",
) -> FormFillResponse:
    """Extract structured fields from uploaded ID/source documents.

    Runs a two-stage pipeline:
    1. Extract markdown from each uploaded document (local or cloud backend).
    2. Use the local LM to map extracted text to the template's required fields.

    Available templates: lease_agreement, rent_receipt, dl_renewal, affidavit, bank_account.

    Response headers:
    - ``X-Request-ID``: Unique request identifier.
    - ``X-Data-Retained``: Always ``false``.
    """
    client_ip = request.client.host if request.client else "unknown"
    _limiter.check(client_ip, is_local=use_local)

    request_id: str = getattr(request.state, "request_id", "")

    docs: list[tuple[bytes, str]] = []
    for upload in files:
        raw = await upload.read()
        docs.append((raw, upload.filename or "document"))

    try:
        result = form_filler.fill(
            docs,
            template_id,
            use_local=use_local,
            api_key=api_key or None,
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return FormFillResponse(
        fields=result.fields,
        field_sources=result.field_sources,
        missing_fields=result.missing_fields,
        request_id=request_id,
    )


@router.get("/forms/templates", tags=["forms"])
async def list_templates() -> list[dict]:
    """List all available form templates with their required and optional fields."""
    return form_filler.list_templates()
