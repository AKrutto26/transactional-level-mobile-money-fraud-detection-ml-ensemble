"""
api.py
------
FastAPI service for the explainable mobile money fraud detection prototype.
A thin layer over DecisionEngine: authentication, input validation, routing.

Security:
  - Every endpoint except /health requires the X-API-Key header. The key is read
    from the FRAUD_API_KEY environment variable or the project's .env file and
    compared in constant time. The server refuses to start without a key.
  - Strict input validation: allowed transaction types only, bounded numbers,
    account-ID format checks, unknown fields rejected (HTTP 422).
  - Verification codes are never returned by the API; they go to the simulated
    SMS outbox only.
  - /health reveals only whether the service is up and the model version; operational
    detail (feature-store state, counters, mode) is on the authenticated /status endpoint.

Reliability:
  - Idempotency-Key header (optional, 8-64 characters of A-Z a-z 0-9 _ -). A retry
    with the same key returns the original decision (HTTP 200, idempotent_replay=true)
    and is not recorded twice; the same key with a different body is refused (409).
  - Shadow mode (FRAUD_MODE=shadow): decisions are made and audited but the caller is
    always told APPROVED; the real decision is in "shadow_status".
  - Degraded mode: if the feature store or model fails, decisions fall back to a safe
    policy instead of scoring with default features (see decision_engine.py).

Endpoints:
  GET  /health                         service status and model version (no key needed)
  GET  /status                         mode, model versions, feature-store state, counters
  POST /transactions                   score and decide one transaction
  POST /transactions/{tx_id}/verify    submit a step-up verification code
  GET  /transactions/{tx_id}           full audit record with its event history
  GET  /alerts                         open alert queue, highest risk first
  POST /alerts/{tx_id}/review          analyst marks FRAUD or LEGITIMATE
  GET  /payments/review                payments the anomaly screen suggests reviewing
  POST /payments/{tx_id}/review        analyst closes a suggested payment review
  GET  /audit/verify                   verify the audit hash chain
  GET  /metrics                        counts and monitoring indicators
  GET  /outbox                         simulated SMS messages (demo only)

SHAP explanations are computed in the background after the response is sent,
so the customer-facing decision returns at model speed.

Run with ONE worker (the feature store must process transactions in order):
    python -m uvicorn src.api:app --port 8000
Shadow mode (PowerShell):
    $env:FRAUD_MODE="shadow"; python -m uvicorn src.api:app --port 8000
Interactive documentation: http://127.0.0.1:8000/docs
(click "Authorize" and paste the API key once)

Changelog:
  v1.1 — Idempotency-Key header, /status, payment review endpoints, shadow and
         degraded modes surfaced in responses.
"""

import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Path, Query, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from .config import ROOT
from .decision_engine import DecisionEngine, IdempotencyConflict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fraud_api")

TX_ID_PATTERN = r"^TX-[0-9A-F]{12}$"
ACCOUNT_PATTERN = r"^[CM][0-9]{1,15}$"
IDEMPOTENCY_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


def _load_api_key() -> str:
    key = os.environ.get("FRAUD_API_KEY")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("FRAUD_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key or len(key) < 16:
        raise RuntimeError("FRAUD_API_KEY is missing or shorter than 16 characters. "
                           "Add FRAUD_API_KEY=<long random value> to the .env file in the project root.")
    return key


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.api_key = _load_api_key()
    logger.info("Loading decision engine ...")
    app.state.engine = DecisionEngine()
    eng = app.state.engine
    logger.info(f"Warm-up complete in {eng.warmup():.0f} ms; model {eng.model_version}; mode {eng.mode.upper()}; "
                f"payment screening {'ON' if eng.payment else 'OFF'}")
    st = eng.store.status()
    if not st["clean"]:
        logger.info(f"Feature store: {st['note']}")
    yield
    app.state.engine.close()


app = FastAPI(
    title="Explainable Mobile Money Fraud Detection API",
    version="1.1",
    description="Prototype decision service: calibrated Random Forest + XGBoost ensemble, "
                "rules engine, SHAP explanations, step-up verification, a tamper-evident audit log, "
                "advisory payment anomaly screening, idempotent requests and shadow mode. "
                "Research prototype on simulated (PaySim) data; not connected to any live payment system.",
    lifespan=lifespan,
)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="API key from the .env file")


def require_api_key(x_api_key: str | None = Security(api_key_header)):
    if not x_api_key or not secrets.compare_digest(x_api_key, app.state.api_key):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def engine() -> DecisionEngine:
    return app.state.engine


# ── Request models ─────────────────────────────────────────────────────────
class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(..., ge=1, le=100_000, description="Simulation hour (1 step = 1 hour)")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., gt=0, le=1e10, allow_inf_nan=False)
    nameOrig: str = Field(..., pattern=ACCOUNT_PATTERN)
    nameDest: str = Field(..., pattern=ACCOUNT_PATTERN)
    oldbalanceOrg: float | None = Field(None, ge=0, le=1e11, allow_inf_nan=False,
                                        description="Origin balance before the transaction")


class VerifyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(..., pattern=r"^[0-9]{6}$")


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: Literal["FRAUD", "LEGITIMATE"]
    analyst: str = Field(..., min_length=2, max_length=50)
    note: str = Field("", max_length=500)


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    eng = getattr(app.state, "engine", None)
    return {"status": "ok" if eng else "starting",
            "model_version": eng.model_version if eng else None}


@app.get("/status", dependencies=[Depends(require_api_key)])
def status(eng: DecisionEngine = Depends(engine)):
    return eng.status()


def _explain_in_background(eng: DecisionEngine, tx_id: str) -> None:
    try:
        eng.complete_explanation(tx_id)
    except Exception:                                   # never crash the server on an explanation
        logger.exception(f"Explanation failed for {tx_id}")


@app.post("/transactions", status_code=201, dependencies=[Depends(require_api_key)])
def score_transaction(tx: TransactionIn, background: BackgroundTasks,
                      idempotency_key: str | None = Header(None, alias="Idempotency-Key",
                                                           pattern=IDEMPOTENCY_PATTERN),
                      eng: DecisionEngine = Depends(engine)):
    try:
        result = eng.process(tx.model_dump(), defer_explanation=True, idempotency_key=idempotency_key)
    except IdempotencyConflict:
        raise HTTPException(status_code=409,
                            detail="Idempotency-Key was already used with a different transaction")
    if result.get("idempotent_replay"):
        return JSONResponse(status_code=200, content=result)
    if result["explanation_status"] == "PENDING":
        background.add_task(_explain_in_background, eng, result["tx_id"])
    return result


@app.post("/transactions/{tx_id}/verify", dependencies=[Depends(require_api_key)])
def verify_code(body: VerifyIn, tx_id: str = Path(..., pattern=TX_ID_PATTERN),
                eng: DecisionEngine = Depends(engine)):
    result = eng.verify(tx_id, body.code)
    if result["result"] == "NO_CHALLENGE" and result["status"] is None:
        raise HTTPException(status_code=404, detail="Unknown transaction")
    return result


@app.get("/transactions/{tx_id}", dependencies=[Depends(require_api_key)])
def get_transaction(tx_id: str = Path(..., pattern=TX_ID_PATTERN), eng: DecisionEngine = Depends(engine)):
    record = eng.audit.get_transaction(tx_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown transaction")
    return record


@app.get("/alerts", dependencies=[Depends(require_api_key)])
def alerts(limit: int = Query(50, ge=1, le=500), eng: DecisionEngine = Depends(engine)):
    return eng.audit.open_alerts(limit)


@app.post("/alerts/{tx_id}/review", dependencies=[Depends(require_api_key)])
def review(body: ReviewIn, tx_id: str = Path(..., pattern=TX_ID_PATTERN), eng: DecisionEngine = Depends(engine)):
    try:
        return eng.review(tx_id, body.outcome, body.analyst, body.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown transaction")


@app.get("/payments/review", dependencies=[Depends(require_api_key)])
def payment_review_queue(status: Literal["OPEN", "CLOSED"] = Query("OPEN"),
                         limit: int = Query(50, ge=1, le=500), eng: DecisionEngine = Depends(engine)):
    return eng.payment_reviews(status, limit)


@app.post("/payments/{tx_id}/review", dependencies=[Depends(require_api_key)])
def close_payment_review(body: ReviewIn, tx_id: str = Path(..., pattern=TX_ID_PATTERN),
                         eng: DecisionEngine = Depends(engine)):
    try:
        return eng.close_payment_review(tx_id, body.outcome, body.analyst, body.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="No payment review for this transaction")


@app.get("/audit/verify", dependencies=[Depends(require_api_key)])
def audit_verify(eng: DecisionEngine = Depends(engine)):
    return eng.audit.verify_chain()


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics(eng: DecisionEngine = Depends(engine)):
    return eng.audit.metrics()


@app.get("/outbox", dependencies=[Depends(require_api_key)])
def outbox(tx_id: str | None = Query(None, pattern=TX_ID_PATTERN), limit: int = Query(20, ge=1, le=200),
           eng: DecisionEngine = Depends(engine)):
    return eng.audit.outbox(tx_id, limit)
