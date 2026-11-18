"""
benchmark_latency.py
--------------------
Measures warm, steady-state latency of the decision engine for Chapter 5.

Runs on COPIES of the feature store and a temporary audit log, so the live
state used by the API is never modified.

Procedure:
  1. Warm up the engine (one dummy score and explanation).
  2. Replay the first N test-period transactions (all types, original order)
     with deferred explanations, timing every stage of each decision.
  3. Compute the deferred SHAP explanations for every MEDIUM-or-higher
     decision and time them separately.

Reports median, 95th percentile and maximum per stage:
  customer-facing decision latency  (what the customer waits for)
  explanation latency               (what the analyst waits for, in the background)

Outputs:
  results/metrics/latency_benchmark.csv

Run:
    python -m src.benchmark_latency            # 2,000 transactions
    python -m src.benchmark_latency --n 500    # quicker
"""

import argparse
import logging
import shutil

import numpy as np
import pandas as pd

from .audit_log import AuditLog
from .config import METRICS_DIR, RAW_DATA_PATH, VAL_MAX_STEP
from .decision_engine import DecisionEngine
from .feature_store import STATE_DIR, STORE_PATH, FeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _remove_db(path):
    for suffix in ("", "-wal", "-shm"):
        p = path.with_name(path.name + suffix)
        if p.exists():
            p.unlink()


def stats(values, label, path_name):
    v = np.asarray(values, dtype=float)
    return {"measure": label, "path": path_name, "count": len(v),
            "median_ms": round(float(np.median(v)), 3),
            "p95_ms": round(float(np.percentile(v, 95)), 3),
            "max_ms": round(float(v.max()), 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    if not STORE_PATH.exists():
        raise SystemExit("Feature store not found. Run: python -m src.feature_store --seed")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    store_copy = STATE_DIR / "benchmark_store.db"
    audit_copy = STATE_DIR / "benchmark_audit.db"
    _remove_db(store_copy)
    _remove_db(audit_copy)
    shutil.copy(STORE_PATH, store_copy)

    engine = DecisionEngine(FeatureStore(store_copy), AuditLog(audit_copy))
    logger.info(f"Warm-up: {engine.warmup():.0f} ms")

    raw = pd.read_csv(RAW_DATA_PATH, usecols=["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest"])
    sample = raw[raw["step"] > VAL_MAX_STEP].head(args.n)
    del raw
    logger.info(f"Replaying {len(sample):,} test-period transactions ...")

    results = []
    for r in sample.itertuples(index=False):
        tx = {"step": int(r.step), "type": r.type, "amount": float(r.amount),
              "nameOrig": r.nameOrig, "nameDest": r.nameDest, "oldbalanceOrg": float(r.oldbalanceOrg)}
        results.append(engine.process(tx, defer_explanation=True))

    shap_times = []
    for res in results:
        if res["explanation_status"] == "PENDING":
            shap_times.append(engine.complete_explanation(res["tx_id"])["shap_ms"])
    engine.close()
    _remove_db(store_copy)
    _remove_db(audit_copy)

    scored = [r for r in results if r["scored_by_model"]]
    routed = [r for r in results if not r["scored_by_model"]]
    rows = [stats([r["timings_ms"]["decision_total_ms"] for r in results], "decision (customer-facing)", "all transactions")]
    if scored:
        rows += [stats([r["timings_ms"]["decision_total_ms"] for r in scored], "decision (customer-facing)", "scored by model"),
                 stats([r["timings_ms"]["features_ms"] for r in scored], "feature store", "scored by model"),
                 stats([r["timings_ms"]["model_ms"] for r in scored], "ensemble + calibration", "scored by model"),
                 stats([r["timings_ms"]["audit_ms"] for r in scored], "audit log write", "scored by model")]
    if routed:
        rows.append(stats([r["timings_ms"]["decision_total_ms"] for r in routed], "decision (customer-facing)", "routed by type"))
    if shap_times:
        rows.append(stats(shap_times, "SHAP explanation (background)", "MEDIUM or above"))
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "latency_benchmark.csv", index=False)

    bands = pd.Series([r["risk_band"] for r in results]).value_counts()
    print("\n" + "=" * 100)
    print(f"LATENCY BENCHMARK — {len(results):,} test-period transactions, warm engine, deferred explanations")
    print("=" * 100)
    print(df.to_string(index=False))
    print("-" * 100)
    print(f"Scored by model: {len(scored):,} | routed by type: {len(routed):,} | "
          f"explanations computed: {len(shap_times):,}")
    print(f"Bands: {bands.to_dict()}")
    print("=" * 100)


if __name__ == "__main__":
    main()
