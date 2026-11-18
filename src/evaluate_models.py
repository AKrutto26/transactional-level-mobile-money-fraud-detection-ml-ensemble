"""
evaluate_models.py
------------------
Evaluates all trained models on the held-out test set.
Produces a full comparison table and saves results to
results/metrics/.

Metrics computed:
  PR-AUC, ROC-AUC, Precision, Recall, F1,
  FPR, FNR, TP, FP, TN, FN,
  Inference time (ms/transaction)

Threshold selection:
  Determined on validation set, never on test set.

Run:
    python -m src.evaluate_models
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    FEATURE_DATA_PATH,
    IF_MODEL_PATH,
    METRICS_DIR,
    MODEL_DIR,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    TRAIN_MAX_STEP,
    VAL_MAX_STEP,
)
from .features import MODEL_FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_splits(df: pd.DataFrame):
    train = df[df["step"] <= TRAIN_MAX_STEP]
    val   = df[(df["step"] > TRAIN_MAX_STEP) & (df["step"] <= VAL_MAX_STEP)]
    test  = df[df["step"] > VAL_MAX_STEP]

    X_train = train[MODEL_FEATURE_COLUMNS].values.astype("float32")
    y_train = train["isFraud"].values.astype("int8")
    X_val   = val[MODEL_FEATURE_COLUMNS].values.astype("float32")
    y_val   = val["isFraud"].values.astype("int8")
    X_test  = test[MODEL_FEATURE_COLUMNS].values.astype("float32")
    y_test  = test["isFraud"].values.astype("int8")

    return X_train, y_train, X_val, y_val, X_test, y_test


def select_threshold_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Select classification threshold that maximises F1 on validation set.
    Never called on the test set.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = np.where(
        (precision + recall) == 0,
        0,
        2 * precision * recall / (precision + recall),
    )
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    logger.info(f"  Best threshold (max F1 on val): {best_threshold:.4f}  "
                f"F1={f1_scores[best_idx]:.4f}")
    return best_threshold


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    model_name: str,
    inference_ms_per_txn: float,
) -> dict:
    """Compute full metric suite at a given threshold."""
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pr_auc     = average_precision_score(y_true, scores)
    roc_auc    = roc_auc_score(y_true, scores)
    precision  = precision_score(y_true, y_pred, zero_division=0)
    recall     = recall_score(y_true, y_pred, zero_division=0)
    f1         = f1_score(y_true, y_pred, zero_division=0)
    fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr        = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "model":               model_name,
        "threshold":           round(threshold, 4),
        "pr_auc":              round(pr_auc, 4),
        "roc_auc":             round(roc_auc, 4),
        "precision":           round(precision, 4),
        "recall":              round(recall, 4),
        "f1":                  round(f1, 4),
        "fpr":                 round(fpr, 4),
        "fnr":                 round(fnr, 4),
        "tp":                  int(tp),
        "fp":                  int(fp),
        "tn":                  int(tn),
        "fn":                  int(fn),
        "inference_ms_per_txn": round(inference_ms_per_txn, 4),
    }


def measure_inference(model, X: np.ndarray, model_type: str) -> tuple:
    """Return (scores, ms_per_transaction)."""
    t0 = time.time()
    if model_type == "isolation_forest":
        scores = -model.decision_function(X)
    else:
        scores = model.predict_proba(X)[:, 1]
    elapsed_ms = (time.time() - t0) * 1000
    ms_per_txn = elapsed_ms / len(X)
    return scores, ms_per_txn


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    logger.info(f"Loading features from {FEATURE_DATA_PATH} ...")
    df = pd.read_parquet(FEATURE_DATA_PATH)

    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(df)
    del df

    logger.info(f"Test set: {len(X_test):,} rows  fraud={int(y_test.sum()):,}")

    results = []

    # ── Model A — Isolation Forest ────────────────────────────────────
    logger.info("\n── Isolation Forest ──────────────────────────────────")
    if_model = joblib.load(IF_MODEL_PATH)

    # Threshold on validation set
    val_scores_if, _ = measure_inference(if_model, X_val, "isolation_forest")
    threshold_if = select_threshold_f1(y_val, val_scores_if)

    # Evaluate on test set
    test_scores_if, ms_if = measure_inference(if_model, X_test, "isolation_forest")
    metrics_if = compute_metrics(y_test, test_scores_if, threshold_if,
                                  "Isolation Forest", ms_if)
    results.append(metrics_if)
    logger.info(f"  Test PR-AUC={metrics_if['pr_auc']}  "
                f"Recall={metrics_if['recall']}  F1={metrics_if['f1']}")

    # ── Model B — Random Forest ───────────────────────────────────────
    logger.info("\n── Random Forest ─────────────────────────────────────")
    rf_model = joblib.load(RF_MODEL_PATH)

    val_scores_rf, _ = measure_inference(rf_model, X_val, "random_forest")
    threshold_rf = select_threshold_f1(y_val, val_scores_rf)

    test_scores_rf, ms_rf = measure_inference(rf_model, X_test, "random_forest")
    metrics_rf = compute_metrics(y_test, test_scores_rf, threshold_rf,
                                  "Random Forest", ms_rf)
    results.append(metrics_rf)
    logger.info(f"  Test PR-AUC={metrics_rf['pr_auc']}  "
                f"Recall={metrics_rf['recall']}  F1={metrics_rf['f1']}")

    # ── Model C — XGBoost ─────────────────────────────────────────────
    logger.info("\n── XGBoost ───────────────────────────────────────────")
    xgb_model = joblib.load(XGB_MODEL_PATH)

    val_scores_xgb, _ = measure_inference(xgb_model, X_val, "xgboost")
    threshold_xgb = select_threshold_f1(y_val, val_scores_xgb)

    test_scores_xgb, ms_xgb = measure_inference(xgb_model, X_test, "xgboost")
    metrics_xgb = compute_metrics(y_test, test_scores_xgb, threshold_xgb,
                                   "XGBoost", ms_xgb)
    results.append(metrics_xgb)
    logger.info(f"  Test PR-AUC={metrics_xgb['pr_auc']}  "
                f"Recall={metrics_xgb['recall']}  F1={metrics_xgb['f1']}")

    # ── Save thresholds ───────────────────────────────────────────────
    thresholds = {
        "isolation_forest": threshold_if,
        "random_forest":    threshold_rf,
        "xgboost":          threshold_xgb,
    }
    threshold_path = MODEL_DIR / "thresholds.json"
    with open(threshold_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info(f"\nThresholds saved → {threshold_path}")

    # ── Save results table ────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    csv_path = METRICS_DIR / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved → {csv_path}")

    # ── Print comparison table ────────────────────────────────────────
    logger.info("\n" + "=" * 90)
    logger.info("MODEL COMPARISON — TEST SET RESULTS")
    logger.info("=" * 90)
    cols = ["model", "pr_auc", "roc_auc", "precision", "recall",
            "f1", "fpr", "fnr", "tp", "fp", "fn", "inference_ms_per_txn"]
    print(results_df[cols].to_string(index=False))
    logger.info("=" * 90)
    logger.info("Run src.train_ensemble next to build the IF+XGBoost hybrid.")


if __name__ == "__main__":
    main()