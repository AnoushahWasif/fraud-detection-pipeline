# Scoring service only: runtime dependencies, the package, and the hash-checked model.
# Build after `make all` (the model isn't in git):  make docker

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
# Dependencies first, so code changes don't rebuild this layer. No dev group, no MLflow extra.
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv ./.venv
COPY --from=build --chown=app:app /app/src ./src
COPY --chown=app:app models/fraud ./models/fraud
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 FRAUD_MODEL_DIR=/app/models/fraud
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
# 0.0.0.0 inside the container only; `make serve-docker` publishes it to 127.0.0.1 on the host.
CMD ["uvicorn", "fraud_pipeline.service:app", "--host", "0.0.0.0", "--port", "8000"]
