"""
check_feature_parity.py
-----------------------
Proves the live feature store computes exactly the features the model was
trained on (no training-serving skew).

Procedure:
  1. Copy the seeded store (state as of step 631) to a temporary file, so the
     real store is never modified.
  2. Replay every test-period transaction (steps 632+, ALL types) in original
     file order. For each TRANSFER / CASH_OUT, compute features from the store
     BEFORE recording it, exactly as the API will.
  3. Compare with the test rows of features_v2.parquet, feature by feature.

Pass criteria (fixed in advance):
  - counts, hours, flags, time gaps: identical
  - amount-based features: relative difference <= 1e-5 (training summed amounts
    in a different order, so the last decimal place can differ)

Outputs:
  results/metrics/feature_parity.csv

Run (after `python -m src.feature_store --seed`):
    python -m src.check_feature_parity
"""

import logging
import shutil
import time

import numpy as np
import pandas as pd

from .build_features_v2 import FEATURES_V2_PATH, FS2_COLUMNS, MODELLED_TYPES
from .config import METRICS_DIR, RAW_DATA_PATH, VAL_MAX_STEP
from .feature_store import STATE_DIR, STORE_PATH, FeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXACT = ["hour_of_day", "type_TRANSFER", "dest_txn_count_before",
         "dest_time_since_last_step", "dest_is_new"]
APPROX = ["amount", "log_amount", "dest_total_amount_before",
          "dest_avg_amount_before", "dest_amount_to_avg_ratio"]
RTOL = 1e-5


def main():
    if not STORE_PATH.exists():
        raise SystemExit("Feature store not found. Run: python -m src.feature_store --seed")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    tmp = STATE_DIR / "parity_check.db"
    for p in (tmp, tmp.with_name(tmp.name + "-wal"), tmp.with_name(tmp.name + "-shm")):
        if p.exists():
            p.unlink()
    shutil.copy(STORE_PATH, tmp)
    store = FeatureStore(tmp)
    logger.info(f"Store copy: {store.size():,} destinations, seeded through step "
                f"{store.get_meta('seeded_through_step')}")

    raw = pd.read_csv(RAW_DATA_PATH, usecols=["step", "type", "amount", "nameDest"],
                      dtype={"step": "int32", "type": "object", "amount": "float64", "nameDest": "object"})
    replay = raw[raw["step"] > VAL_MAX_STEP]
    del raw
    logger.info(f"Replaying {len(replay):,} test-period transactions (all types) in file order ...")

    live_rows = []
    t0 = time.time()
    compute_time = 0.0
    for step, tx_type, amount, dest in replay.itertuples(index=False, name=None):
        if tx_type in MODELLED_TYPES:
            t1 = time.perf_counter()
            f = store.compute_features(int(step), tx_type, amount, dest)
            compute_time += time.perf_counter() - t1
            live_rows.append([f[c] for c in FS2_COLUMNS])
        store.record(int(step), amount, dest, commit=False)
    store.commit()
    elapsed = time.time() - t0
    store.close()
    for p in (tmp, tmp.with_name(tmp.name + "-wal"), tmp.with_name(tmp.name + "-shm")):
        if p.exists():
            p.unlink()

    live = pd.DataFrame(live_rows, columns=FS2_COLUMNS)
    trained = pd.read_parquet(FEATURES_V2_PATH, columns=["step"] + FS2_COLUMNS)
    trained = trained[trained["step"] > VAL_MAX_STEP].reset_index(drop=True)
    if len(live) != len(trained):
        raise SystemExit(f"Row count mismatch: live {len(live):,} vs training {len(trained):,}")

    rows = []
    for col in FS2_COLUMNS:
        a = live[col].to_numpy(dtype="float64")
        b = trained[col].to_numpy(dtype="float64")
        if col in EXACT:
            mismatched = int((a != b).sum())
            max_rel = float("nan")
            rule = "exact"
        else:
            denom = np.maximum(np.abs(b), 1e-12)
            rel = np.where((a == 0) & (b == 0), 0.0, np.abs(a - b) / denom)
            mismatched = int((rel > RTOL).sum())
            max_rel = float(rel.max())
            rule = f"relative <= {RTOL:g}"
        rows.append({"feature": col, "rule": rule, "rows_compared": len(a),
                     "mismatched_rows": mismatched, "max_relative_difference": max_rel,
                     "status": "PASS" if mismatched == 0 else "FAIL"})
    result = pd.DataFrame(rows)
    result.to_csv(METRICS_DIR / "feature_parity.csv", index=False)

    n_scored = len(live)
    print("\n" + "=" * 100)
    print("FEATURE PARITY — live feature store vs training features (test period)")
    print("=" * 100)
    print(result.to_string(index=False))
    print("-" * 100)
    print(f"Replayed {len(replay):,} transactions in {elapsed:.1f}s; computed features for {n_scored:,} "
          f"TRANSFER/CASH_OUT transactions")
    print(f"Feature computation: {compute_time / n_scored * 1000:.3f} ms per transaction (store lookup included)")
    if (result["status"] == "PASS").all():
        print("VERDICT: PASS — the live store reproduces the training features. No training-serving skew.")
    else:
        print("VERDICT: FAIL — see mismatched features above.")
    print("=" * 100)


if __name__ == "__main__":
    main()
