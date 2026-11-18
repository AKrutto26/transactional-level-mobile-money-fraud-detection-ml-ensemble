"""
decision_engine.py
------------------
End-to-end processing of a single transaction. The API is a thin layer over
this class, so the same logic can be tested without a web server.

For each transaction:
   0. idempotency: if the client re-sends a request with the same Idempotency-Key,
      the stored response is returned and NOTHING is recorded again
   1. route by type: CASH_IN and DEBIT are approved directly; PAYMENT is approved
      and screened by the advisory payment anomaly model (if trained)
   2. build the 10 model features from the feature store (history only)
   3. score with the ensemble (mean of component probabilities)
   4. calibrate the score into a fraud probability
   5. apply the rules engine with the roles saved in rules_config.json
   6. final band = the higher of the ML band and any rule-forced band
   7. action from the band:
        LOW -> APPROVED, MEDIUM -> PENDING_VERIFICATION (code issued),
        HIGH -> HELD (alert), CRITICAL -> DECLINED (alert, escalated)
   8. SHAP reasons, for MEDIUM and above only
   9. write the decision to the hash-chained audit log
  10. record the transaction in the feature store (AFTER scoring), together with
      the idempotency entry, in ONE store transaction

Steps 0-10 run inside a lock, one transaction at a time. Without it, two
simultaneous transactions to the same destination could both read its history
before either is recorded, giving one of them wrong features (a race condition).

Operating modes (environment variable FRAUD_MODE):
  enforce (default)  decisions are returned as made
  shadow             decisions are made, explained and audited exactly as in
                     enforce mode, but the caller is always told APPROVED and no
                     verification code is sent. The real decision is returned in
                     "shadow_status". This is how an operator would trial the model
                     on live traffic before letting it act.

Safe degraded mode:
  If the feature store or the model fails, the engine does NOT score with default
  features (that would make every recipient look new and flood the alert queue).
  Instead, scorable transactions at or above DEGRADED_STEPUP_AMOUNT (environment
  variable, default 100,000) go to step-up verification and smaller ones are
  approved; the decision is marked degraded=true. Transactions that could not be
  recorded are written to data/state/pending_records.jsonl and applied by
  replay_pending() once the store is healthy again.

Quick check (processes ONE example transaction and writes it to the live store
and audit log; reset with `python -m src.audit_log --reset` afterwards):
    python -m src.decision_engine

Changelog:
  v1.1 — start-up warm-up; deferred (background) explanations; every rule that
         fires is reported with its role, including UNUSED rules; transactions
         are processed under a lock to prevent feature-store race conditions.
  v1.2 — idempotency keys; advisory PAYMENT anomaly screening with an analyst
         review queue; shadow mode; safe degraded mode with a replay journal;
         status() for monitoring.
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid

import numpy as np
import pandas as pd

from .audit_log import AuditLog
from .config import MODEL_DIR, ROOT
from .explain import FraudExplainer
from .feature_store import FeatureStore
from .payment_anomaly import PaymentScorer
from .rules import (
    BAND_RANK, RULE_DESCRIPTIONS, combine_bands, evaluate_rules, load_rules_config,
    route_transaction_type, rule_bands,
)

logger = logging.getLogger("decision_engine")

FINAL_DIR = MODEL_DIR / "final"
STATUS_BY_BAND = {"LOW": "APPROVED", "MEDIUM": "PENDING_VERIFICATION",
                  "HIGH": "HELD", "CRITICAL": "DECLINED"}
EXPLAIN_FROM_BAND = "MEDIUM"
PENDING_JOURNAL = ROOT / "data" / "state" / "pending_records.jsonl"


class IdempotencyConflict(Exception):
    """The same Idempotency-Key was reused with a different request body."""


def _request_hash(tx: dict) -> str:
    return hashlib.sha256(json.dumps(tx, sort_keys=True, default=str).encode()).hexdigest()


class DecisionEngine:
    def __init__(self, feature_store: FeatureStore | None = None, audit: AuditLog | None = None):
        self.explainer = FraudExplainer()
        # Single-transaction scoring: one thread avoids thread start-up overhead per request
        for model in self.explainer.components.values():
            if "n_jobs" in model.get_params():
                model.set_params(n_jobs=1)
        self.config = self.explainer.config
        self.rules = load_rules_config(FINAL_DIR / "rules_config.json")
        self.store = feature_store or FeatureStore()
        self.audit = audit or AuditLog()
        self.model_version = f"{self.config['source_experiment']} ({self.config['created']})"
        self.mode = os.environ.get("FRAUD_MODE", "enforce").strip().lower()
        if self.mode not in ("enforce", "shadow"):
            raise ValueError("FRAUD_MODE must be 'enforce' or 'shadow'")
        self.degraded_stepup_amount = float(os.environ.get("DEGRADED_STEPUP_AMOUNT", 100_000))
        self.payment = PaymentScorer() if PaymentScorer.available() else None
        if self.payment is None:
            logger.warning("Payment anomaly model not found; PAYMENT is approved without screening. "
                           "Train it with: python -m src.payment_anomaly --train")
        self._lock = threading.Lock()
        self._pending = {}
        self.counters = {"decisions": 0, "degraded": 0, "idempotent_replays": 0,
                         "payments_screened": 0, "payment_reviews": 0, "journal_pending": 0}
        self.replay_pending()

    def warmup(self) -> float:
        """Score and explain one background row so the first real request is not slow."""
        t = time.perf_counter()
        X = self.explainer.background[:1]
        self.explainer.risk(X)
        self.explainer.shap_values(X)
        if self.payment is not None:
            self.payment.assess({c: 0.0 for c in self.payment.config["features"]})
        return (time.perf_counter() - t) * 1000

    def _rules_fired(self, flags: pd.DataFrame) -> list[dict]:
        return [{"rule": r, "role": self.rules["roles"].get(r, "UNUSED"),
                 "description": RULE_DESCRIPTIONS[r]}
                for r in flags.columns if bool(flags[r].iloc[0])]

    # ── Decision paths ────────────────────────────────────────────────────
    def _routed(self, tx: dict, timings: dict) -> dict:
        """CASH_IN, DEBIT and PAYMENT: not scored by the fraud model."""
        tx_type = tx["type"]
        decision = {
            "risk_score": 0.0, "ml_band": None, "rule_band": "LOW", "risk_band": "LOW",
            "recommended_action": self.config["band_actions"]["LOW"], "status": "APPROVED",
            "triggered_rules": [], "rules_fired": [],
            "top_reasons": [{"reason": "Transaction type",
                             "description": f"{tx_type} transactions are outside the fraud model's coverage: "
                                            "no fraud labels exist for this type in the training data",
                             "impact_pct_points": 0.0, "direction": "routing rule"}],
            "explanation_status": "NOT_REQUIRED",
            "scored_by_model": False,
        }
        if tx_type == "PAYMENT" and self.payment is not None:
            t = time.perf_counter()
            feats = self.store.compute_features(tx["step"], tx_type, tx["amount"], tx["nameDest"])
            assessment = self.payment.assess(feats, tx.get("oldbalanceOrg"))
            timings["payment_ms"] = (time.perf_counter() - t) * 1000
            decision["payment_screening"] = assessment
            decision["review_suggested"] = assessment["review_suggested"]
            if assessment["review_suggested"]:
                decision["top_reasons"] = [
                    {"reason": r["reason"], "description": r["description"],
                     "impact_pct_points": None, "direction": "advisory"} for r in assessment["reasons"]
                ] + decision["top_reasons"]
        return decision

    def _scored(self, tx: dict, timings: dict, defer_explanation: bool):
        """TRANSFER and CASH_OUT: the calibrated ensemble, rules and bands."""
        tx_type = tx["type"]
        t = time.perf_counter()
        feats = self.store.compute_features(tx["step"], tx_type, tx["amount"], tx["nameDest"],
                                            tx.get("oldbalanceOrg"))
        X = FeatureStore.model_vector(feats)
        timings["features_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        risk = float(self.explainer.risk(X)[0])
        ml_band = str(self.explainer.band(np.array([risk]))[0])
        timings["model_ms"] = (time.perf_counter() - t) * 1000

        rule_inputs = pd.DataFrame([{**feats, "empties_orig_account": feats.get("empties_orig_account", 0)}])
        flags = evaluate_rules(rule_inputs)
        forced, triggered = rule_bands(flags, self.rules["roles"])
        final_band = str(combine_bands(np.array([ml_band], dtype=object), forced)[0])
        needs_reasons = BAND_RANK[final_band] >= BAND_RANK[EXPLAIN_FROM_BAND]

        reasons, explanation_status = [], "NOT_REQUIRED"
        if needs_reasons and not defer_explanation:
            t = time.perf_counter()
            values, _, _ = self.explainer.shap_values(X)
            reasons = self.explainer.grouped_reasons(values[0], X[0], k=3)
            timings["shap_ms"] = (time.perf_counter() - t) * 1000
            explanation_status = "COMPLETE"
        elif needs_reasons:
            explanation_status = "PENDING"

        decision = {
            "risk_score": round(risk, 4), "ml_band": ml_band, "rule_band": str(forced[0]),
            "risk_band": final_band,
            "recommended_action": self.config["band_actions"][final_band],
            "status": STATUS_BY_BAND[final_band],
            "triggered_rules": [{"rule": r, "description": RULE_DESCRIPTIONS[r]} for r in triggered[0]],
            "rules_fired": self._rules_fired(flags),
            "top_reasons": reasons,
            "explanation_status": explanation_status,
            "scored_by_model": True,
        }
        return decision, X

    def _degraded(self, tx: dict, error: Exception) -> dict:
        """Safe fallback when the store or model fails: never score with default features."""
        scorable = route_transaction_type(tx["type"]) != "LOW"
        band = "MEDIUM" if scorable and float(tx["amount"]) >= self.degraded_stepup_amount else "LOW"
        return {
            "risk_score": None, "ml_band": None, "rule_band": None, "risk_band": band,
            "recommended_action": self.config["band_actions"][band], "status": STATUS_BY_BAND[band],
            "triggered_rules": [], "rules_fired": [],
            "top_reasons": [{"reason": "Degraded mode",
                             "description": "The fraud model could not be applied "
                                            f"({type(error).__name__}); "
                                            + ("large transfers and withdrawals are sent to step-up verification"
                                               if band == "MEDIUM" else "the transaction was approved by policy"),
                             "impact_pct_points": 0.0, "direction": "policy"}],
            "explanation_status": "NOT_REQUIRED",
            "scored_by_model": False,
            "degraded": True,
        }

    # ── Main entry point ──────────────────────────────────────────────────
    def process(self, tx: dict, defer_explanation: bool = False, idempotency_key: str | None = None) -> dict:
        t_start = time.perf_counter()
        timings = {}
        tx_type = tx["type"].upper().replace("-", "_")
        tx = {**tx, "type": tx_type}
        req_hash = _request_hash(tx)

        with self._lock:
            if idempotency_key:
                previous = self.store.idem_get(idempotency_key)
                if previous is not None:
                    if previous[0] != req_hash:
                        raise IdempotencyConflict(idempotency_key)
                    self.counters["idempotent_replays"] += 1
                    return {**previous[1], "idempotent_replay": True}

            tx_id = f"TX-{uuid.uuid4().hex[:12].upper()}"
            X = None
            try:
                if route_transaction_type(tx_type) == "LOW":
                    decision = self._routed(tx, timings)
                else:
                    decision, X = self._scored(tx, timings, defer_explanation)
            except (sqlite3.Error, OSError, ValueError, KeyError) as exc:
                logger.exception(f"Degraded decision for {tx_id}")
                decision, X = self._degraded(tx, exc), None
                self.counters["degraded"] += 1

            decision["model_version"] = self.model_version
            decision["mode"] = self.mode.upper()
            t = time.perf_counter()
            self.audit.record_decision(tx_id, tx, decision)
            if decision["status"] == "PENDING_VERIFICATION" and self.mode == "enforce":
                self.audit.issue_otp(tx_id, recipient=tx.get("nameOrig") or "account holder")
            timings["audit_ms"] = (time.perf_counter() - t) * 1000

            timings["decision_total_ms"] = (time.perf_counter() - t_start) * 1000
            result = {"tx_id": tx_id, **decision,
                      "timings_ms": {k: round(v, 3) for k, v in timings.items()}}
            if self.mode == "shadow":
                result["shadow_status"] = result["status"]
                result["status"] = "APPROVED"

            # Feature history, idempotency entry and payment review: one store transaction
            try:
                self.store.record(tx["step"], tx["amount"], tx["nameDest"], commit=False)
                if idempotency_key:
                    self.store.idem_put(idempotency_key, req_hash, tx_id, result, commit=False)
                if decision.get("review_suggested"):
                    self.store.add_payment_review(tx_id, tx, decision["payment_screening"], commit=False)
                self.store.commit()
            except sqlite3.Error:
                logger.exception(f"Feature store write failed for {tx_id}; journalled for replay")
                try:
                    self.store.rollback()
                except sqlite3.Error:
                    pass
                self._journal(tx)

            self.counters["decisions"] += 1
            if tx_type == "PAYMENT" and "payment_screening" in decision:
                self.counters["payments_screened"] += 1
                self.counters["payment_reviews"] += int(decision["review_suggested"])
            if decision["explanation_status"] == "PENDING":
                self._pending[tx_id] = X
        return result

    # ── Replay journal (records that could not be written) ──────────────────
    def _journal(self, tx: dict) -> None:
        PENDING_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with open(PENDING_JOURNAL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": tx["step"], "amount": tx["amount"], "nameDest": tx["nameDest"]}) + "\n")
        self.counters["journal_pending"] += 1

    def replay_pending(self) -> int:
        """Apply journalled records to the store, in order; returns how many were applied."""
        if not PENDING_JOURNAL.exists():
            return 0
        lines = [json.loads(x) for x in PENDING_JOURNAL.read_text(encoding="utf-8").splitlines() if x.strip()]
        try:
            for r in lines:
                self.store.record(r["step"], r["amount"], r["nameDest"], commit=False)
            self.store.commit()
        except sqlite3.Error:
            self.store.rollback()
            logger.exception("Replay of pending records failed; journal kept")
            return 0
        PENDING_JOURNAL.unlink()
        self.counters["journal_pending"] = 0
        logger.info(f"Replayed {len(lines)} pending feature-store records")
        return len(lines)

    # ── Explanations, verification, review ────────────────────────────────
    def complete_explanation(self, tx_id: str) -> dict | None:
        """Compute deferred SHAP reasons and attach them to the audit record."""
        X = self._pending.pop(tx_id, None)
        if X is None:
            return None
        t = time.perf_counter()
        values, _, _ = self.explainer.shap_values(X)
        reasons = self.explainer.grouped_reasons(values[0], X[0], k=3)
        shap_ms = (time.perf_counter() - t) * 1000
        self.audit.attach_explanation(tx_id, reasons, {"shap_ms": round(shap_ms, 3)})
        return {"tx_id": tx_id, "top_reasons": reasons, "shap_ms": round(shap_ms, 3)}

    def verify(self, tx_id: str, code: str) -> dict:
        return self.audit.verify_otp(tx_id, code)

    def review(self, tx_id: str, outcome: str, analyst: str, note: str = "") -> dict:
        return self.audit.record_review(tx_id, outcome, analyst, note)

    def payment_reviews(self, status: str = "OPEN", limit: int = 50) -> list[dict]:
        return self.store.payment_reviews(status, limit)

    def close_payment_review(self, tx_id: str, outcome: str, analyst: str, note: str = "") -> dict:
        if not self.store.close_payment_review(tx_id, outcome):
            raise KeyError(tx_id)
        try:
            self.audit.record_review(tx_id, "FRAUD" if outcome == "FRAUD" else "LEGITIMATE", analyst, note)
        except Exception:                               # the audit may not treat approved payments as alerts
            logger.info(f"Payment review for {tx_id} stored in the payment queue only")
        return {"tx_id": tx_id, "status": "CLOSED", "outcome": outcome}

    # ── Monitoring ────────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "mode": self.mode.upper(),
            "model_version": self.model_version,
            "payment_model": self.payment.version if self.payment else None,
            "degraded_stepup_amount": self.degraded_stepup_amount,
            "feature_store": self.store.status(),
            "counters": dict(self.counters),
        }

    def close(self) -> None:
        self.store.close()
        self.audit.close()


if __name__ == "__main__":
    engine = DecisionEngine()
    print(f"Warm-up: {engine.warmup():.0f} ms")
    example = {"step": 700, "type": "TRANSFER", "amount": 181_000.0,
               "nameOrig": "C0000000001", "nameDest": "C9999999999", "oldbalanceOrg": 181_000.0}
    print(json.dumps(engine.process(example), indent=2, default=str))
    engine.close()
