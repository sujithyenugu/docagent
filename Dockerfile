# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies for pymupdf and pytesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libmupdf-dev \
    tesseract-ocr \
    tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python deps into a prefix so we can copy them cleanly to runtime stage
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Non-root user for security
RUN useradd --system --uid 1001 --no-create-home appuser

# Copy application source
COPY api/         ./api/
COPY views/       ./views/
COPY .streamlit/  ./.streamlit/
COPY local_model.py storage.py agent.py app.py common.py nav.py ./

# Ensure Python path includes /app so `from api.* import` works
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Default env vars (override at runtime)
# Note: local extraction (embedded Apple MLX) requires Apple Silicon macOS and
# is unavailable inside this Linux container — Cloud mode is unaffected.
ENV CORS_ORIGINS="*"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
