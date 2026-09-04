FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY evals/ ./evals/
COPY scripts/ ./scripts/

# Checkpoints and traces are state, so mount a volume here in production.
RUN mkdir -p /app/data
ENV CHECKPOINT_DB=/app/data/checkpoints.sqlite \
    TRACE_DIR=/app/data/traces

# Run as a non-root user: this container is allowed to grant system access,
# so it should not also be root inside its own sandbox.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
