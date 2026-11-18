"""
test_api_live.py
----------------
Automated functional tests for the RUNNING API (Chapter 5, Table 5.8).

Start the API first, in another terminal:
    python -m uvicorn src.api:app --port 8000
Then run:
    python -m src.test_api_live

Every check prints PASS, FAIL or SKIPPED, and the full table is saved to
results/metrics/api_functional_tests.csv.

Note: these tests send real requests, so they add transactions to the live
feature store and audit log. Before the final demo, reset with:
    python -m src.feature_store --seed
    python -m src.audit_log --reset
"""

import json
import re
import secrets
import time
import urllib.error
import urllib.request

import pandas as pd

from .config import METRICS_DIR, ROOT

BASE = "http://127.0.0.1:8000"
RESULTS = []


def read_key() -> str:
    raw = (ROOT / ".env").read_bytes()
    text = raw.decode("utf-16") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8-sig")
    for line in text.splitlines():
        if line.strip().startswith("FRAUD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("No FRAUD_API_KEY line in .env")


def call(method: str, path: str, key: str | None = None, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("X-API-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def record(test_id, group, scenario, expected, actual, passed, skipped=False):
    status = "SKIPPED" if skipped else ("PASS" if passed else "FAIL")
    RESULTS.append({"test_id": test_id, "group": group, "scenario": scenario,
                    "expected": expected, "actual": actual, "status": status})
    print(f"{status:<8} {test_id:<4} {scenario:<55} -> {actual}")


def new_account(prefix="C"):
    return prefix + str(secrets.randbelow(10**12) + 10**12)


def tx(step, tx_type, amount, balance=None, dest=None):
    body = {"step": step, "type": tx_type, "amount": amount,
            "nameOrig": new_account(), "nameDest": dest or new_account()}
    if balance is not None:
        body["oldbalanceOrg"] = balance
    return body


def find_pending(key, exclude=()):
    """Send candidate transactions until one lands in MEDIUM (PENDING_VERIFICATION)."""
    for step in (706, 712, 718, 700):
        for amount in (60_000, 120_000, 250_000, 30_000, 500_000, 1_000_000):
            for t in ("TRANSFER", "CASH_OUT"):
                code, r = call("POST", "/transactions", key, tx(step, t, amount, amount))
                if code == 201 and r["status"] == "PENDING_VERIFICATION" and r["tx_id"] not in exclude:
                    return r
    return None


def code_from_outbox(key, tx_id):
    _, msgs = call("GET", f"/outbox?tx_id={tx_id}", key)
    m = re.search(r"\b(\d{6})\b", msgs[0]["message"]) if msgs else None
    return m.group(1) if m else None


def main():
    key = read_key()
    try:
        code, body = call("GET", "/health")
    except urllib.error.URLError:
        raise SystemExit("The API is not running. Start it first: python -m uvicorn src.api:app --port 8000")

    print("=" * 110)
    print("API FUNCTIONAL TESTS")
    print("=" * 110)
    record("T01", "Availability", "Health check", "200, status ok", f"{code}, {body.get('status')}",
           code == 200 and body.get("status") == "ok")

    # Security
    code, _ = call("GET", "/alerts")
    record("T02", "Security", "Request without an API key", "401", code, code == 401)
    code, _ = call("GET", "/alerts", "wrong-key-" + "x" * 20)
    record("T03", "Security", "Request with a wrong API key", "401", code, code == 401)

    # Input validation
    bad = [("T04", "Unknown transaction type (BITCOIN)", {**tx(700, "TRANSFER", 1000), "type": "BITCOIN"}),
           ("T05", "Negative amount", tx(700, "TRANSFER", -500)),
           ("T06", "Extra field sent (isFraud)", {**tx(700, "TRANSFER", 1000), "isFraud": 0}),
           ("T07", "Injection attempt in account name", {**tx(700, "TRANSFER", 1000), "nameOrig": "'; DROP TABLE--"})]
    for tid, scenario, body in bad:
        code, _ = call("POST", "/transactions", key, body)
        record(tid, "Input validation", scenario, "422 (rejected)", code, code == 422)

    # Decisions
    code, r = call("POST", "/transactions", key, tx(710, "PAYMENT", 1500, dest=new_account("M")))
    record("T08", "Decisions", "PAYMENT routed by type", "201, APPROVED, not scored by model",
           f"{code}, {r.get('status')}, scored={r.get('scored_by_model')}",
           code == 201 and r.get("status") == "APPROVED" and r.get("scored_by_model") is False)

    code, risky = call("POST", "/transactions", key, tx(700, "TRANSFER", 181_000, 181_000))
    record("T09", "Decisions", "Risky transfer (181,000, new account, 04:00)", "201, flagged (not APPROVED)",
           f"{code}, {risky.get('risk_band')}, {risky.get('status')}, risk {risky.get('risk_score')}",
           code == 201 and risky.get("status") != "APPROVED")

    explained = None
    if risky.get("explanation_status") == "PENDING":
        for _ in range(30):
            time.sleep(0.5)
            _, rec = call("GET", f"/transactions/{risky['tx_id']}", key)
            if rec.get("explanation_status") == "COMPLETE":
                explained = rec
                break
    record("T10", "Decisions", "Explanation attached in the background", "COMPLETE with reasons",
           ("COMPLETE: " + "; ".join(x["description"] for x in explained["reasons"])) if explained else "not complete",
           explained is not None and len(explained["reasons"]) > 0)

    # Lookups
    code, _ = call("GET", "/transactions/TX-000000000000", key)
    record("T11", "Lookups", "Unknown transaction ID", "404", code, code == 404)
    code, _ = call("GET", "/transactions/hello", key)
    record("T12", "Lookups", "Malformed transaction ID", "422", code, code == 422)

    # Verification codes
    pending = find_pending(key)
    if pending:
        tid = pending["tx_id"]
        real = code_from_outbox(key, tid)
        wrong = "000000" if real != "000000" else "111111"
        _, r = call("POST", f"/transactions/{tid}/verify", key, {"code": wrong})
        record("T13", "Verification", "Wrong code", "WRONG_CODE, attempts left",
               f"{r.get('result')}, attempts_left={r.get('attempts_left')}", r.get("result") == "WRONG_CODE")
        _, r = call("POST", f"/transactions/{tid}/verify", key, {"code": real})
        record("T14", "Verification", "Correct code from the simulated SMS", "VERIFIED", r.get("result"),
               r.get("result") == "VERIFIED")
        _, r = call("POST", f"/transactions/{tid}/verify", key, {"code": real})
        record("T15", "Verification", "Reusing a code that already worked", "rejected (CHALLENGE_USED)",
               r.get("result"), r.get("result") == "CHALLENGE_USED")
        second = find_pending(key, exclude=(tid,))
        if second:
            for _ in range(3):
                _, r = call("POST", f"/transactions/{second['tx_id']}/verify", key, {"code": "999999"})
            record("T16", "Verification", "Three wrong codes", "LOCKED, VERIFICATION_FAILED",
                   f"{r.get('result')}, {r.get('status')}", r.get("result") == "LOCKED")
        else:
            record("T16", "Verification", "Three wrong codes", "LOCKED", "no second MEDIUM transaction found", False, True)
    else:
        for t_id, s in (("T13", "Wrong code"), ("T14", "Correct code"), ("T15", "Reused code"), ("T16", "Three wrong codes")):
            record(t_id, "Verification", s, "-", "no MEDIUM transaction found", False, True)

    # Analyst review
    _, alerts = call("GET", "/alerts?limit=500", key)
    in_queue = any(a["tx_id"] == risky.get("tx_id") for a in alerts)
    code, r = call("POST", f"/alerts/{risky.get('tx_id')}/review", key,
                   {"outcome": "FRAUD", "analyst": "test_script", "note": "automated functional test"})
    _, alerts_after = call("GET", "/alerts?limit=500", key)
    left_queue = not any(a["tx_id"] == risky.get("tx_id") for a in alerts_after)
    record("T17", "Analyst", "Flagged transaction in queue, then confirmed as fraud",
           "in queue, REVIEWED_FRAUD, removed from queue",
           f"in_queue={in_queue}, {r.get('status')}, removed={left_queue}",
           in_queue and r.get("status") == "REVIEWED_FRAUD" and left_queue)

    # Integrity
    _, chain = call("GET", "/audit/verify", key)
    code, m = call("GET", "/metrics", key)
    record("T18", "Integrity", "Audit hash chain and monitoring after all tests",
           "chain valid, metrics returned",
           f"valid={chain.get('valid')}, events={chain.get('events_checked')}, metrics={code}",
           chain.get("valid") is True and code == 200)

    df = pd.DataFrame(RESULTS)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS_DIR / "api_functional_tests.csv", index=False)
    counts = df["status"].value_counts().to_dict()
    print("-" * 110)
    print(f"PASS {counts.get('PASS', 0)} | FAIL {counts.get('FAIL', 0)} | SKIPPED {counts.get('SKIPPED', 0)}"
          f"  ->  saved to results/metrics/api_functional_tests.csv")
    print("=" * 110)


if __name__ == "__main__":
    main()
