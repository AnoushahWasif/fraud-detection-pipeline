"""Day 9: the scoring API.

    FRAUD_API_KEY=... uv run uvicorn fraud_pipeline.service:app --host 127.0.0.1 --port 8000

POST /score takes up to 100 transactions and returns a score and an alert flag for each, with
the model's version. It never blocks anything itself: the alert goes to a human review queue.
"""

import hmac
import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraud_pipeline.model_store import ModelIntegrityError, load_model
from fraud_pipeline.schema import FEATURES, MAX_TIME

MAX_BATCH = 100
# A full batch of 100 is about 65 KB as JSON, so the cap leaves room for it.
MAX_BODY_BYTES = 256 * 1024
MIN_KEY_LENGTH = 32
log = logging.getLogger("fraud_pipeline.service")


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(ge=0, le=MAX_TIME)
    Amount: float = Field(ge=0, le=1_000_000)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    @field_validator("*")
    @classmethod
    def finite_and_sane(cls, value: float, info):
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        if info.field_name.startswith("V") and abs(value) >= 1e3:
            raise ValueError("outside the range the model was trained on")
        return value


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH)


class Scored(BaseModel):
    score: float
    alert: bool


class ScoreResponse(BaseModel):
    model_version: str
    threshold: float
    results: list[Scored]


class Settings:
    def __init__(self, model_dir: Path, api_key: str, pinned_sha: str | None = None):
        if len(api_key) < MIN_KEY_LENGTH:
            raise RuntimeError(f"FRAUD_API_KEY must be at least {MIN_KEY_LENGTH} characters")
        loaded = load_model(model_dir, pinned_sha)
        self.model = loaded.model
        self.threshold = float(loaded.manifest["threshold"])
        self.version = loaded.manifest["sha256"][:12]
        self.api_key = api_key.encode()


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="fraud scoring", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        # FastAPI's default 422 echoes the rejected input. That would put transaction values
        # in responses, and a NaN in them can't even be encoded (it turned into a 500).
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.middleware("http")
    async def limits_and_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex[:16]
        start = time.perf_counter()
        declared = request.headers.get("content-length")
        if request.method == "POST" and declared is None:
            # A chunked body has no declared size, so the cap couldn't be checked up front.
            return JSONResponse({"detail": "Content-Length required"}, status_code=411)
        if declared is not None and (not declared.isdigit() or int(declared) > MAX_BODY_BYTES):
            return JSONResponse({"detail": "request too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        # Counts and timings only: transaction values never reach the logs.
        log.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(1000 * (time.perf_counter() - start), 1),
                }
            )
        )
        return response

    def authorised(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), settings.api_key):
            raise HTTPException(status_code=401, detail="missing or wrong API key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": settings.version}

    @app.post("/score", dependencies=[Depends(authorised)])
    def score(body: ScoreRequest) -> ScoreResponse:
        frame = pd.DataFrame([t.model_dump() for t in body.transactions])[FEATURES]
        scores = settings.model.predict_proba(frame)[:, 1]
        results = [
            Scored(score=round(float(s), 6), alert=bool(s >= settings.threshold)) for s in scores
        ]
        log.info(json.dumps({"scored": len(results), "alerts": sum(r.alert for r in results)}))
        return ScoreResponse(
            model_version=settings.version, threshold=settings.threshold, results=results
        )

    return app


def from_environment() -> FastAPI:
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    key = os.environ.get("FRAUD_API_KEY", "")
    try:
        settings = Settings(
            Path(os.environ.get("FRAUD_MODEL_DIR", "models/fraud")),
            key,
            os.environ.get("FRAUD_MODEL_SHA256") or None,
        )
    except ModelIntegrityError as exc:
        raise RuntimeError(f"refusing to start: {exc}") from exc
    return create_app(settings)


def __getattr__(name: str):
    # `uvicorn fraud_pipeline.service:app` builds the app on first access, so importing this
    # module (for tests) doesn't need a key or a model.
    if name == "app":
        return from_environment()
    raise AttributeError(name)
