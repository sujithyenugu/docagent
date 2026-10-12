"""GET /api/v1/health — service readiness probe."""

import os

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Return service status and backend availability flags.

    Response:
        status:                  "ok"
        local_model_available:   True if this platform can run the embedded
                                 MLX local model (Apple Silicon macOS).
        local_model_downloaded:  True if the local model's weights are
                                 already cached (no download needed).
        cloud_api_configured:    True if VISION_AGENT_API_KEY is set.
    """
    from local_model import is_model_ready, is_platform_supported

    local_model_available = is_platform_supported()
    local_model_downloaded = is_model_ready() if local_model_available else False
    cloud_api_configured = bool(os.environ.get("VISION_AGENT_API_KEY"))
    return {
        "status": "ok",
        "local_model_available": local_model_available,
        "local_model_downloaded": local_model_downloaded,
        "cloud_api_configured": cloud_api_configured,
    }
