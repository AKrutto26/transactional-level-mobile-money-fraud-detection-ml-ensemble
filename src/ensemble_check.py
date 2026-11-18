"""
ensemble_check.py
-----------------
Closes the three-model question: does averaging the fraud probabilities of
Random Forest (V2-04) and XGBoost with square-root class weighting (V2-10)
outperform XGBoost alone?

No retraining: both models are loaded from models/v2/.
Threshold is selected on validation; the test set is scored once.

Decision rule (fixed BEFORE running):
  ADOPT  -> soft voting is significantly BETTER than V2-10 (paired bootstrap,
            95% CI of the PR-AUC difference entirely above zero)
  KEEP   -> otherwise; XGBoost (V2-10) remains the final model

The result is logged as V2-11 in results/metrics/experiment_log.csv.

Run:
    python -m src.ensemble_check
"""

import logging
import time
from datetime import datetime

import __main__
import joblib
import pandas as pd

from . import run_experiments as rx
from .run_experiments import (
    LOG_PATH, V2_MODEL_DIR, bootstrap_ap_ci, load_splits, metrics_at,
    paired_bootstrap, save_scores, select_threshold, upsert_log,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_SET = "FS2_clean"


def expose_wrappers_for_unpickling():
    """
    The V2 models were saved while running `python -m src.run_experiments`,
    so pickle recorded their classes under the module name `__main__`.
    Registering the classes on the current __main__ lets joblib load them.
    """
    for name in ("SklearnClassifier", "IsoForestScorer", "XGBScorer", "IFXGBHybrid"):
        setattr(__main__, name, getattr(rx, name))


def main():
    expose_wrappers_for_unpickling()
    X_cache, y = load_splits()
    X = X_cache[FEATURE_SET]

    rf = joblib.load(V2_MODEL_DIR / "V2-04.joblib")
    xgb = joblib.load(V2_MODEL_DIR / "V2-10.joblib")

    # Validation: threshold selection only
    val_scores = (rf.score(X["val"]) + xgb.score(X["val"])) / 2
    threshold = select_threshold(y["val"], val_scores)

    # Test: scored once
    t0 = time.time()
    rf_test = rf.score(X["test"])
    xgb_test = xgb.score(X["test"])
    test_scores = (rf_test + xgb_test) / 2
    inference_ms = (time.time() - t0) * 1000 / len(X["test"])

    m = metrics_at(y["test"], test_scores, threshold)
    ci_low, ci_high = bootstrap_ap_ci(y["test"], test_scores)

    log = pd.read_csv(LOG_PATH).set_index("exp_id")
    train_time = float(log.loc["V2-04", "train_time_s"] + log.loc["V2-10", "train_time_s"])
    size_mb = float(log.loc["V2-04", "model_size_mb"] + log.loc["V2-10", "model_size_mb"])

    row = {
        "exp_id": "V2-11", "model": "RF + XGBoost sqrt (soft voting)", "feature_set": FEATURE_SET,
        "n_features": X["train"].shape[1],
        "pr_auc": round(m["pr_auc"], 4),
        "pr_auc_ci_low": round(ci_low, 4), "pr_auc_ci_high": round(ci_high, 4),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k != "pr_auc"},
        "threshold": round(threshold, 4),
        "train_time_s": round(train_time, 1),
        "inference_ms_per_txn": round(inference_ms, 5),
        "model_size_mb": round(size_mb, 2),
        "best_iteration": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    upsert_log(row)
    save_scores("V2-11", y["test"], test_scores)

    rows = []
    for ref, label in (("V2-10", "XGBoost sqrt (final)"), ("V2-04", "Random Forest")):
        ref_scores = xgb_test if ref == "V2-10" else rf_test
        res = paired_bootstrap(y["test"], test_scores, ref_scores)
        if res["ci_low"] > 0:
            verdict = "soft voting significantly BETTER"
        elif res["ci_high"] < 0:
            verdict = "soft voting significantly WORSE"
        else:
            verdict = "no significant difference"
        rows.append({"reference": f"{ref} {label}", **{k: round(v, 4) for k, v in res.items()},
                     "verdict": verdict})
    comp = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("V2-11 — RF + XGBoost soft voting (test set)")
    print("=" * 100)
    print(f"PR-AUC {row['pr_auc']} [{row['pr_auc_ci_low']}, {row['pr_auc_ci_high']}]  "
          f"Precision {row['precision']}  Recall {row['recall']}  F1 {row['f1']}  "
          f"FP {row['fp']}  threshold {row['threshold']}")
    print(f"Combined model size {row['model_size_mb']} MB  |  scoring {row['inference_ms_per_txn']} ms/txn")
    print("-" * 100)
    print(comp.to_string(index=False))
    print("-" * 100)
    adopt = comp.iloc[0]["ci_low"] > 0
    if adopt:
        print("DECISION: ADOPT — soft voting is significantly better than XGBoost alone.")
    else:
        print("DECISION: KEEP XGBoost (V2-10) — combining with Random Forest gives no significant "
              "improvement, at extra cost.")
    print("=" * 100)


if __name__ == "__main__":
    main()
