"""DocumentAgent FastAPI application.

Run alongside the Streamlit UI:

    streamlit run app.py          # port 8501 (unchanged)
    uvicorn api.main:app --port 8000 --reload

Endpoints
---------
GET  /api/v1/health                          — readiness probe
POST /api/v1/extract                         — document → markdown (local or cloud)
POST /api/v1/forms/fill                      — documents + template → structured fields
GET  /api/v1/forms/templates                 — list available form templates
POST /api/v1/grade/extract                   — extract + keyword-match answer sheet
GET  /api/v1/grade/rubrics                   — list grading rubrics
GET  /api/v1/grade/rubrics/{id}              — rubric detail
POST /api/v1/grade/rubrics/{id}/evaluate     — extract + score against rubric

Privacy guarantee
-----------------
All responses carry ``X-Data-Retained: false``.  No uploaded document or
extracted text is persisted by the API layer (S3 upload remains opt-in via
the Streamlit UI only).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.request_id import RequestIDMiddleware
from api.routers import extract, forms, grade, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DocumentAgent API starting up")
    yield
    logger.info("DocumentAgent API shut down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="DocumentAgent API",
        description=(
            "Privacy-first document extraction API. "
            "Supports local (embedded Apple MLX / Qwen2.5-VL-3B) and cloud (Landing AI ADE) backends."
        ),
        version="0.4.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (applied in reverse registration order) ─────────────────
    # RequestIDMiddleware runs first (outermost), injects X-Request-ID
    app.add_middleware(RequestIDMiddleware)

    # CORS — allow all origins by default; tighten via CORS_ORIGINS env var
    raw_origins = os.environ.get("CORS_ORIGINS", "*")
    origins = [o.strip() for o in raw_origins.split(",")] if raw_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Data-Retained"],
    )

    # ── Privacy response header ────────────────────────────────────────────
    @app.middleware("http")
    async def add_privacy_header(request, call_next):
        response = await call_next(request)
        response.headers["X-Data-Retained"] = "false"
        return response

    # ── Routers ────────────────────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(extract.router, prefix=prefix)
    app.include_router(forms.router, prefix=prefix)
    app.include_router(grade.router, prefix=prefix)

    return app


app = create_app()
