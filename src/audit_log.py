"""
audit_log.py
------------
Tamper-evident audit log, step-up verification codes, and a simulated
message outbox for the fraud detection system.

Design:
  events        append-only ledger. Every decision, verification attempt and
                analyst review is an event. Each event stores the SHA-256 hash
                of the previous event, forming a hash chain: altering or
                deleting any past event breaks every hash after it, and
                verify_chain() reports the first broken event.
  transactions  current state of each transaction (for fast queries such as
                the alert queue). It can always be checked against the ledger.
  otp_challenges  step-up verification codes, stored ONLY as salted SHA-256
                hashes, valid for OTP_TTL_SECONDS, at most OTP_MAX_ATTEMPTS
                attempts, compared in constant time.
  outbox        simulated SMS channel. The code is "delivered" here instead of
                through a real SMS or email provider, so no credentials are
                needed and the demo does not depend on a network.

Transaction lifecycle:
  APPROVED | PENDING_VERIFICATION -> VERIFIED | VERIFICATION_FAILED | EXPIRED
  HELD | DECLINED -> REVIEWED_FRAUD | REVIEWED_LEGITIMATE  (analyst review)

Changelog:
  v1.1 — explanations can be attached after the decision (EXPLANATION event,
         hash-chained like every other event); transactions gain an
         explanation_status column; older databases are upgraded automatically.

Reset the log (demo use):
    python -m src.audit_log --reset
Verify the hash chain:
    python -m src.audit_log --verify
"""

import argparse
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import ROOT

AUDIT_PATH = ROOT / "data" / "state" / "audit.db"
GENESIS_HASH = "0" * 64
OTP_TTL_SECONDS = 300
OTP_MAX_ATTEMPTS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_id       TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    hash        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_tx ON events(tx_id);
CREATE TABLE IF NOT EXISTS transactions (
    tx_id               TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    step                INTEGER,
    tx_type             TEXT,
    amount              REAL,
    name_orig           TEXT,
    name_dest           TEXT,
    risk_score          REAL,
    risk_band           TEXT,
    recommended_action  TEXT,
    status              TEXT NOT NULL,
    reasons             TEXT,
    triggered_rules     TEXT,
    explanation_status  TEXT,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tx_status ON transactions(status);
CREATE TABLE IF NOT EXISTS otp_challenges (
    tx_id       TEXT PRIMARY KEY,
    salt        TEXT NOT NULL,
    code_hash   TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_id       TEXT NOT NULL,
    channel     TEXT NOT NULL,
    recipient   TEXT NOT NULL,
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

ALERT_STATUSES = ("HELD", "DECLINED", "PENDING_VERIFICATION", "VERIFICATION_FAILED")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def _canonical(obj) -> str:
    """Deterministic JSON so the same content always hashes the same way."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _event_hash(prev_hash: str, tx_id: str, event_type: str, payload: str, created_at: str) -> str:
    material = "|".join([prev_hash, tx_id, event_type, payload, created_at])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path = AUDIT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(transactions)")}
        if "explanation_status" not in cols:            # upgrade databases created by v1.0
            self.conn.execute("ALTER TABLE transactions ADD COLUMN explanation_status TEXT")
            self.conn.commit()
        self.lock = threading.RLock()

    # ── Hash-chained ledger ─────────────────────────────────────────────
    def _append_event(self, tx_id: str, event_type: str, payload: dict) -> dict:
        row = self.conn.execute("SELECT hash FROM events ORDER BY event_id DESC LIMIT 1").fetchone()
        prev_hash = row["hash"] if row else GENESIS_HASH
        body = _canonical(payload)
        created = _iso(_now())
        h = _event_hash(prev_hash, tx_id, event_type, body, created)
        self.conn.execute(
            "INSERT INTO events (tx_id, event_type, payload, created_at, prev_hash, hash) VALUES (?,?,?,?,?,?)",
            (tx_id, event_type, body, created, prev_hash, h))
        return {"event_type": event_type, "hash": h}

    def verify_chain(self) -> dict:
        """Recompute every hash; report the first event that does not match."""
        prev = GENESIS_HASH
        count = 0
        for r in self.conn.execute("SELECT * FROM events ORDER BY event_id"):
            count += 1
            expected = _event_hash(prev, r["tx_id"], r["event_type"], r["payload"], r["created_at"])
            if r["prev_hash"] != prev or r["hash"] != expected:
                return {"valid": False, "events_checked": count, "first_broken_event_id": r["event_id"],
                        "tx_id": r["tx_id"], "event_type": r["event_type"]}
            prev = r["hash"]
        return {"valid": True, "events_checked": count, "first_broken_event_id": None}

    # ── Decisions ───────────────────────────────────────────────────────
    def record_decision(self, tx_id: str, tx: dict, decision: dict) -> None:
        now = _iso(_now())
        with self.lock:
            self.conn.execute(
                """INSERT INTO transactions (tx_id, created_at, step, tx_type, amount, name_orig, name_dest,
                       risk_score, risk_band, recommended_action, status, reasons, triggered_rules,
                       explanation_status, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (tx_id, now, tx["step"], tx["type"], tx["amount"], tx.get("nameOrig"), tx["nameDest"],
                 decision["risk_score"], decision["risk_band"], decision["recommended_action"],
                 decision["status"], _canonical(decision.get("top_reasons", [])),
                 _canonical(decision.get("triggered_rules", [])),
                 decision.get("explanation_status"), now))
            self._append_event(tx_id, "DECISION", {"transaction": tx, "decision": decision})
            self.conn.commit()

    def attach_explanation(self, tx_id: str, reasons: list, meta: dict | None = None) -> None:
        """Add SHAP reasons computed after the decision (hash-chained EXPLANATION event)."""
        with self.lock:
            self.conn.execute(
                "UPDATE transactions SET reasons = ?, explanation_status = 'COMPLETE', updated_at = ? WHERE tx_id = ?",
                (_canonical(reasons), _iso(_now()), tx_id))
            self._append_event(tx_id, "EXPLANATION", {"top_reasons": reasons, **(meta or {})})
            self.conn.commit()

    def _set_status(self, tx_id: str, status: str) -> None:
        self.conn.execute("UPDATE transactions SET status = ?, updated_at = ? WHERE tx_id = ?",
                          (status, _iso(_now()), tx_id))

    # ── Step-up verification (one-time codes) ───────────────────────────
    def issue_otp(self, tx_id: str, recipient: str) -> None:
        code = f"{secrets.randbelow(10**6):06d}"
        salt = secrets.token_hex(16)
        code_hash = hashlib.sha256((salt + code).encode()).hexdigest()
        expires = _now() + timedelta(seconds=OTP_TTL_SECONDS)
        message = (f"Your verification code is {code}. It expires in {OTP_TTL_SECONDS // 60} minutes. "
                   f"Never share this code with anyone, including callers claiming to be from your provider.")
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO otp_challenges VALUES (?,?,?,?,0,'ACTIVE')",
                              (tx_id, salt, code_hash, _iso(expires)))
            self.conn.execute("INSERT INTO outbox (tx_id, channel, recipient, message, created_at) VALUES (?,?,?,?,?)",
                              (tx_id, "SIMULATED_SMS", recipient, message, _iso(_now())))
            self._append_event(tx_id, "OTP_ISSUED", {"expires_at": _iso(expires), "channel": "SIMULATED_SMS"})
            self.conn.commit()

    def verify_otp(self, tx_id: str, code: str) -> dict:
        with self.lock:
            ch = self.conn.execute("SELECT * FROM otp_challenges WHERE tx_id = ?", (tx_id,)).fetchone()
            if ch is None:
                return {"tx_id": tx_id, "result": "NO_CHALLENGE", "status": self._status(tx_id)}
            if ch["status"] != "ACTIVE":
                return {"tx_id": tx_id, "result": f"CHALLENGE_{ch['status']}", "status": self._status(tx_id)}
            if _now() > datetime.fromisoformat(ch["expires_at"]):
                self.conn.execute("UPDATE otp_challenges SET status = 'EXPIRED' WHERE tx_id = ?", (tx_id,))
                self._set_status(tx_id, "EXPIRED")
                self._append_event(tx_id, "OTP_EXPIRED", {})
                self.conn.commit()
                return {"tx_id": tx_id, "result": "EXPIRED", "status": "EXPIRED"}
            attempts = ch["attempts"] + 1
            candidate = hashlib.sha256((ch["salt"] + str(code)).encode()).hexdigest()
            if hmac.compare_digest(candidate, ch["code_hash"]):
                self.conn.execute("UPDATE otp_challenges SET status = 'USED', attempts = ? WHERE tx_id = ?",
                                  (attempts, tx_id))
                self._set_status(tx_id, "VERIFIED")
                self._append_event(tx_id, "OTP_VERIFIED", {"attempts": attempts})
                self.conn.commit()
                return {"tx_id": tx_id, "result": "VERIFIED", "status": "VERIFIED"}
            if attempts >= OTP_MAX_ATTEMPTS:
                self.conn.execute("UPDATE otp_challenges SET status = 'LOCKED', attempts = ? WHERE tx_id = ?",
                                  (attempts, tx_id))
                self._set_status(tx_id, "VERIFICATION_FAILED")
                self._append_event(tx_id, "OTP_LOCKED", {"attempts": attempts})
                self.conn.commit()
                return {"tx_id": tx_id, "result": "LOCKED", "status": "VERIFICATION_FAILED"}
            self.conn.execute("UPDATE otp_challenges SET attempts = ? WHERE tx_id = ?", (attempts, tx_id))
            self._append_event(tx_id, "OTP_FAILED", {"attempts": attempts})
            self.conn.commit()
            return {"tx_id": tx_id, "result": "WRONG_CODE", "attempts_left": OTP_MAX_ATTEMPTS - attempts,
                    "status": "PENDING_VERIFICATION"}

    # ── Analyst review ──────────────────────────────────────────────────
    def record_review(self, tx_id: str, outcome: str, analyst: str, note: str = "") -> dict:
        outcome = outcome.upper()
        if outcome not in ("FRAUD", "LEGITIMATE"):
            raise ValueError("outcome must be FRAUD or LEGITIMATE")
        status = "REVIEWED_FRAUD" if outcome == "FRAUD" else "REVIEWED_LEGITIMATE"
        with self.lock:
            if self._status(tx_id) is None:
                raise KeyError(tx_id)
            self._set_status(tx_id, status)
            self._append_event(tx_id, "ANALYST_REVIEW", {"outcome": outcome, "analyst": analyst, "note": note})
            self.conn.commit()
        return {"tx_id": tx_id, "status": status}

    # ── Queries ─────────────────────────────────────────────────────────
    def _status(self, tx_id: str):
        r = self.conn.execute("SELECT status FROM transactions WHERE tx_id = ?", (tx_id,)).fetchone()
        return r["status"] if r else None

    def get_transaction(self, tx_id: str):
        r = self.conn.execute("SELECT * FROM transactions WHERE tx_id = ?", (tx_id,)).fetchone()
        if r is None:
            return None
        record = dict(r)
        record["reasons"] = json.loads(record["reasons"] or "[]")
        record["triggered_rules"] = json.loads(record["triggered_rules"] or "[]")
        record["events"] = [
            {"event_id": e["event_id"], "event_type": e["event_type"], "created_at": e["created_at"],
             "payload": json.loads(e["payload"]), "hash": e["hash"]}
            for e in self.conn.execute("SELECT * FROM events WHERE tx_id = ? ORDER BY event_id", (tx_id,))]
        return record

    def open_alerts(self, limit: int = 100) -> list[dict]:
        q = f"""SELECT tx_id, created_at, tx_type, amount, name_dest, risk_score, risk_band,
                       recommended_action, status
                FROM transactions WHERE status IN ({",".join("?" * len(ALERT_STATUSES))})
                ORDER BY risk_score DESC LIMIT ?"""
        return [dict(r) for r in self.conn.execute(q, (*ALERT_STATUSES, limit))]

    def outbox(self, tx_id: str | None = None, limit: int = 20) -> list[dict]:
        if tx_id:
            rows = self.conn.execute("SELECT * FROM outbox WHERE tx_id = ? ORDER BY id DESC", (tx_id,))
        else:
            rows = self.conn.execute("SELECT * FROM outbox ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def metrics(self) -> dict:
        by_band = {r[0]: r[1] for r in self.conn.execute(
            "SELECT risk_band, COUNT(*) FROM transactions GROUP BY risk_band")}
        by_status = {r[0]: r[1] for r in self.conn.execute(
            "SELECT status, COUNT(*) FROM transactions GROUP BY status")}
        reviewed = self.conn.execute(
            """SELECT COUNT(*) AS n, AVG(risk_score) AS mean_risk,
                      SUM(CASE WHEN status = 'REVIEWED_FRAUD' THEN 1 ELSE 0 END) AS fraud
               FROM transactions WHERE status IN ('REVIEWED_FRAUD', 'REVIEWED_LEGITIMATE')""").fetchone()
        n = reviewed["n"] or 0
        monitoring = {
            "reviewed_alerts": n,
            "confirmed_fraud_rate_among_reviewed": round(reviewed["fraud"] / n, 4) if n else None,
            "mean_predicted_risk_among_reviewed": round(reviewed["mean_risk"], 4) if n else None,
        }
        if n:
            gap = monitoring["confirmed_fraud_rate_among_reviewed"] - monitoring["mean_predicted_risk_among_reviewed"]
            monitoring["calibration_gap"] = round(gap, 4)
            monitoring["recalibration_suggested"] = n >= 50 and abs(gap) > 0.10
        return {"transactions_by_band": by_band, "transactions_by_status": by_status, "monitoring": monitoring}

    def close(self) -> None:
        self.conn.close()


def reset_audit_log(path: Path = AUDIT_PATH) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete the audit log (demo use)")
    parser.add_argument("--verify", action="store_true", help="verify the hash chain")
    args = parser.parse_args()
    if args.reset:
        reset_audit_log()
        print(f"Audit log reset: {AUDIT_PATH}")
    if args.verify or not args.reset:
        log = AuditLog()
        print(json.dumps(log.verify_chain(), indent=2))
        log.close()


if __name__ == "__main__":
    main()
