"""
feature_store.py
----------------
Live feature state store for the fraud detection API.

The model's behavioural features describe each DESTINATION account's history
before the current transaction: how many transactions it received, their total
and average value, and when it last transacted. Training computed these from
all earlier transactions (all five types). The live system must reproduce them
exactly, so this store keeps one row of running state per destination:

    dest_state(name, txn_count, total_amount, last_step)

For every incoming transaction:
  1. compute_features()  reads the destination's CURRENT state (history only)
  2. the API scores and decides
  3. record()            updates the state with this transaction

Computing before recording guarantees a transaction never sees itself in its
own history — the same rule that kept training leakage-free.

Arithmetic mirrors build_features_v2.py (float64 sums, float32 outputs) so the
live features match the training features; check_feature_parity.py verifies it.

Seed the store (state as of the end of validation, step 631):
    python -m src.feature_store --seed
Show its state, including whether live transactions have been added since seeding:
    python -m src.feature_store --status
Exit with an error if the store is no longer clean (use before the parity check):
    python -m src.feature_store --require-clean

Changelog:
  v1.1 — the store creates its folder if missing (fresh setups no longer fail).
  v1.2 — high-water mark: every live record updates max_step_recorded and
         live_records, so status() can tell whether the store still matches its
         seeding point (a store that has absorbed test-period replays will fail
         the parity check, and now says so).
         Idempotency table: a retried request with the same Idempotency-Key is
         answered from the stored response and never recorded twice.
         record(), the idempotency entry and the high-water mark are written in
         ONE SQLite transaction (commit=False + commit()), so they cannot diverge.
"""

import argparse
import json
import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .build_features_v2 import FS2_COLUMNS
from .config import RAW_DATA_PATH, ROOT, VAL_MAX_STEP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

STATE_DIR = ROOT / "data" / "state"
STORE_PATH = STATE_DIR / "feature_store.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS dest_state (
    name          TEXT PRIMARY KEY,
    txn_count     INTEGER NOT NULL,
    total_amount  REAL    NOT NULL,
    last_step     INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS store_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency (
    key           TEXT PRIMARY KEY,
    request_hash  TEXT NOT NULL,
    tx_id         TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS payment_review (
    tx_id         TEXT PRIMARY KEY,
    step          INTEGER NOT NULL,
    amount        REAL    NOT NULL,
    name_orig     TEXT,
    name_dest     TEXT    NOT NULL,
    anomaly_score REAL    NOT NULL,
    percentile    REAL    NOT NULL,
    signals_json  TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'OPEN',
    outcome       TEXT,
    created_at    TEXT    NOT NULL
) WITHOUT ROWID;
"""


class FeatureStore:
    """Thread-safe SQLite-backed destination history."""

    def __init__(self, path: Path = STORE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    # ── Live path ───────────────────────────────────────────────────────
    def _state(self, dest: str):
        row = self.conn.execute(
            "SELECT txn_count, total_amount, last_step FROM dest_state WHERE name = ?", (dest,)
        ).fetchone()
        return row if row else (0, 0.0, None)

    def compute_features(self, step: int, tx_type: str, amount: float, name_dest: str,
                         old_balance_orig: float | None = None) -> dict:
        """Model features (FS2) plus rule inputs, from history BEFORE this transaction."""
        with self.lock:
            count, total, last_step = self._state(name_dest)
        amount = float(amount)
        total32 = np.float32(total)
        avg32 = np.float32(total32 / count) if count > 0 else np.float32(0.0)
        features = {
            "amount": np.float32(amount),
            "log_amount": np.float32(np.log1p(max(amount, 0.0))),
            "hour_of_day": int(step % 24),
            "type_TRANSFER": int(tx_type.upper().replace("-", "_") == "TRANSFER"),
            "dest_txn_count_before": int(count),
            "dest_total_amount_before": total32,
            "dest_avg_amount_before": avg32,
            "dest_time_since_last_step": int(step - last_step) if count > 0 else 0,
            "dest_is_new": int(count == 0),
            "dest_amount_to_avg_ratio": np.float32(amount / float(avg32)) if avg32 > 0 else np.float32(0.0),
        }
        if old_balance_orig is not None:
            bal = float(old_balance_orig)
            features["orig_balance_before"] = np.float32(bal)
            features["amount_to_orig_balance"] = np.float32(amount / bal) if bal > 0 else np.float32(-1.0)
            features["empties_orig_account"] = int(bal > 0 and amount >= bal)
        return features

    def record(self, step: int, amount: float, name_dest: str, commit: bool = True) -> None:
        """Add this transaction to the destination's history (call AFTER scoring).

        Also advances the high-water mark, in the same transaction, so status()
        always knows the latest step the store contains.
        """
        with self.lock:
            self.conn.execute(
                """INSERT INTO dest_state (name, txn_count, total_amount, last_step)
                   VALUES (?, 1, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       txn_count = txn_count + 1,
                       total_amount = total_amount + excluded.total_amount,
                       last_step = excluded.last_step""",
                (name_dest, float(amount), int(step)),
            )
            self.conn.execute(
                """INSERT INTO store_meta (key, value) VALUES ('max_step_recorded', ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value = CAST(MAX(CAST(value AS INTEGER), CAST(excluded.value AS INTEGER)) AS TEXT)""",
                (str(int(step)),),
            )
            self.conn.execute(
                """INSERT INTO store_meta (key, value) VALUES ('live_records', '1')
                   ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)""")
            if commit:
                self.conn.commit()

    def commit(self) -> None:
        with self.lock:
            self.conn.commit()

    def rollback(self) -> None:
        with self.lock:
            self.conn.rollback()

    @staticmethod
    def model_vector(features: dict) -> np.ndarray:
        """FS2 features in training column order, ready for the model."""
        return np.array([[features[c] for c in FS2_COLUMNS]], dtype="float32")

    # ── Idempotency ─────────────────────────────────────────────────────
    def idem_get(self, key: str):
        """(request_hash, response dict) for a previously processed key, or None."""
        with self.lock:
            row = self.conn.execute(
                "SELECT request_hash, response_json FROM idempotency WHERE key = ?", (key,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def idem_put(self, key: str, request_hash: str, tx_id: str, response: dict, commit: bool = True) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO idempotency (key, request_hash, tx_id, response_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (key, request_hash, tx_id, json.dumps(response, default=str),
                 pd.Timestamp.now().isoformat(timespec="seconds")))
            if commit:
                self.conn.commit()

    # ── Payment review queue ────────────────────────────────────────────
    def add_payment_review(self, tx_id: str, tx: dict, assessment: dict, commit: bool = True) -> None:
        with self.lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO payment_review
                   (tx_id, step, amount, name_orig, name_dest, anomaly_score, percentile, signals_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tx_id, int(tx["step"]), float(tx["amount"]), tx.get("nameOrig"), tx["nameDest"],
                 float(assessment["anomaly_score"]), float(assessment["anomaly_percentile"]),
                 json.dumps(assessment["signals"]), pd.Timestamp.now().isoformat(timespec="seconds")))
            if commit:
                self.conn.commit()

    def payment_reviews(self, status: str = "OPEN", limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                """SELECT tx_id, step, amount, name_orig, name_dest, anomaly_score, percentile, signals_json,
                          status, outcome, created_at
                   FROM payment_review WHERE status = ? ORDER BY anomaly_score DESC LIMIT ?""",
                (status, int(limit))).fetchall()
        cols = ["tx_id", "step", "amount", "nameOrig", "nameDest", "anomaly_score", "anomaly_percentile",
                "signals", "status", "outcome", "created_at"]
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            d["signals"] = json.loads(d["signals"])
            out.append(d)
        return out

    def close_payment_review(self, tx_id: str, outcome: str) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "UPDATE payment_review SET status = 'CLOSED', outcome = ? WHERE tx_id = ?", (outcome, tx_id))
            self.conn.commit()
        return cur.rowcount > 0

    # ── Metadata and health ─────────────────────────────────────────────
    def set_meta(self, key: str, value) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO store_meta VALUES (?, ?)", (key, str(value)))
            self.conn.commit()

    def get_meta(self, key: str):
        with self.lock:
            row = self.conn.execute("SELECT value FROM store_meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def size(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM dest_state").fetchone()[0]

    def status(self, deep: bool = False) -> dict:
        """Seeding point, high-water mark and whether the store still matches its seed.

        deep=True also scans the data for the latest step it contains. That detects
        contamination even in a store seeded by an older version without counters
        (a full scan, so it is used by the command line, not by the live API).
        """
        seeded = self.get_meta("seeded_through_step")
        seeded = int(seeded) if seeded is not None else None
        hwm = self.get_meta("max_step_recorded")
        hwm = int(hwm) if hwm is not None else None
        live = int(self.get_meta("live_records") or 0)
        data_latest = None
        if deep:
            with self.lock:
                data_latest = self.conn.execute("SELECT MAX(last_step) FROM dest_state").fetchone()[0]
        beyond_seed = (seeded is not None and data_latest is not None and int(data_latest) > seeded)
        clean = live == 0 and not beyond_seed
        latest = max(x for x in (hwm, data_latest, seeded, 0) if x is not None)
        return {
            "destinations": self.size(),
            "seeded_through_step": seeded,
            "seeded_at": self.get_meta("seeded_at"),
            "live_records_since_seed": live,
            "latest_step_in_store": latest,
            "clean": clean,
            "note": ("store matches its seeding point" if clean else
                     f"store contains transactions after its seeding point (latest step {latest}); "
                     "re-seed before running the parity check"),
        }

    def close(self) -> None:
        with self.lock:
            self.conn.close()


def seed_store(path: Path = STORE_PATH, max_step: int = VAL_MAX_STEP) -> None:
    """Build destination state from all transactions (all types) up to max_step."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if Path(path).exists():
        Path(path).unlink()
    for suffix in ("-wal", "-shm"):
        side = Path(str(path) + suffix)
        if side.exists():
            side.unlink()

    t0 = time.time()
    logger.info(f"Reading {RAW_DATA_PATH} ...")
    raw = pd.read_csv(RAW_DATA_PATH, usecols=["step", "amount", "nameDest"],
                      dtype={"step": "int32", "amount": "float64", "nameDest": "object"})
    hist = raw[raw["step"] <= max_step]
    logger.info(f"{len(hist):,} transactions up to step {max_step} (all types)")
    state = hist.groupby("nameDest", sort=False).agg(
        txn_count=("amount", "size"), total_amount=("amount", "sum"), last_step=("step", "max"))
    del raw, hist

    store = FeatureStore(path)
    with store.lock:
        store.conn.executemany(
            "INSERT INTO dest_state (name, txn_count, total_amount, last_step) VALUES (?, ?, ?, ?)",
            zip(state.index, state["txn_count"].astype(int), state["total_amount"].astype(float),
                state["last_step"].astype(int)))
        store.conn.commit()
    store.set_meta("seeded_through_step", max_step)
    store.set_meta("seeded_at", pd.Timestamp.now().isoformat(timespec="seconds"))
    store.set_meta("live_records", 0)
    logger.info(f"Seeded {store.size():,} destinations in {time.time() - t0:.1f}s -> {path}")
    store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true", help="build the store up to the end of validation")
    parser.add_argument("--status", action="store_true", help="show seeding point and live records")
    parser.add_argument("--require-clean", action="store_true",
                        help="exit with code 1 if live transactions were recorded since seeding")
    args = parser.parse_args()
    if args.seed:
        seed_store()
        return
    store = FeatureStore()
    st = store.status(deep=True)
    store.close()
    print(f"Store: {STORE_PATH}")
    for k, v in st.items():
        print(f"  {k:<26} {v:,}" if isinstance(v, int) and not isinstance(v, bool) else f"  {k:<26} {v}")
    if args.require_clean and not st["clean"]:
        print("\nNOT CLEAN: run  python -m src.feature_store --seed  (with the API stopped) and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
