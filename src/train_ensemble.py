"""
train_ensemble.py
-----------------
Builds and evaluates the proposed IF + XGBoost hybrid model.

Architecture:
  1. Isolation Forest generates an anomaly score for each transaction.
  2. The anomaly score is appended to the original 20 features.
  3. XGBoost trains on the 21-feature augmented set.
  4. SHAP explains XGBoost predictions (including the IF score contribution).

Also adds two lightweight comparison baselines:
  - Logistic Regression (simple supervised baseline)
  - Decision Tree (interpretable single-tree baseline)

Final comparison table includes all 5 models:
  Isolation Forest | Random Forest | XGBoost | IF+XGBoost | Logistic Regression | Decision Tree

Run:
    python -m src.train_ensemble
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import (
    FEATURE_DATA_PATH,
    IF_MODEL_PATH,
    METRICS_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    TRAIN_MAX_STEP,
    VAL_MAX_STEP,
    XGB_PARAMS,
)
from .features import MODEL_FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

HYBRID_MODEL_PATH = MODEL_DIR / "if_xgboost_hybrid.joblib"
HYBRID_IF_PATH    = MODEL_DIR / "if_xgboost_hybrid_if_component.joblib"
SCALER_PATH       = MODEL_DIR / "logreg_scaler.joblib"
LR_MODEL_PATH     = MODEL_DIR / "logistic_regression.joblib"
DT_MODEL_PATH     = MODEL_DIR / "decision_tree.joblib"


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

    logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    logger.info(f"Train fraud: {int(y_train.sum()):,}  "
                f"Val fraud: {int(y_val.sum()):,}  "
                f"Test fraud: {int(y_test.sum()):,}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def select_threshold_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Select threshold maximising F1 on validation set only."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precision + recall) == 0,
            0,
            2 * precision * recall / (precision + recall),
        )
    best_idx = int(np.argmax(f1_scores[:-1]))
    threshold = float(thresholds[best_idx])
    logger.info(f"  Val threshold (max F1): {threshold:.4f}  "
                f"F1={f1_scores[best_idx]:.4f}")
    return threshold


def compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    train_time_s: float,
    inference_ms: float,
) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pr_auc    = average_precision_score(y_true, scores)
    roc_auc   = roc_auc_score(y_true, scores)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "model":                model_name,
        "pr_auc":               round(pr_auc, 4),
        "roc_auc":              round(roc_auc, 4),
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "f1":                   round(f1, 4),
        "fpr":                  round(fpr, 4),
        "fnr":                  round(fnr, 4),
        "tp":                   int(tp),
        "fp":                   int(fp),
        "tn":                   int(tn),
        "fn":                   int(fn),
        "threshold":            round(threshold, 4),
        "train_time_s":         round(train_time_s, 1),
        "inference_ms_per_txn": round(inference_ms, 4),
    }


def add_if_score(
    if_model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """Append IF anomaly score as an extra feature column."""
    # decision_function: more negative = more anomalous
    # Invert so higher = more anomalous (consistent with fraud convention)
    anomaly_score = -if_model.decision_function(X).reshape(-1, 1)
    return np.hstack([X, anomaly_score])


def train_logistic_regression(X_train, y_train, X_val, y_val):
    logger.info("\n── Logistic Regression (baseline) ────────────────────")
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)

    t0 = time.time()
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        solver="saga",
    )
    model.fit(X_tr_s, y_train)
    elapsed = time.time() - t0

    proba     = model.predict_proba(X_va_s)[:, 1]
    threshold = select_threshold_f1(y_val, proba)
    logger.info(f"  Train time: {elapsed:.1f}s  Val PR-AUC: "
                f"{average_precision_score(y_val, proba):.4f}")

    joblib.dump(model,  LR_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler, threshold, elapsed


def train_decision_tree(X_train, y_train, X_val, y_val):
    logger.info("\n── Decision Tree (baseline) ───────────────────────────")

    t0 = time.time()
    model = DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    proba     = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold_f1(y_val, proba)
    logger.info(f"  Train time: {elapsed:.1f}s  Val PR-AUC: "
                f"{average_precision_score(y_val, proba):.4f}")

    joblib.dump(model, DT_MODEL_PATH)
    return model, threshold, elapsed


def train_if_xgboost_hybrid(X_train, y_train, X_val, y_val):
    """
    The proposed hybrid model:
      Step 1 — Train Isolation Forest on training data only.
      Step 2 — Append IF anomaly score to feature matrix.
      Step 3 — Train XGBoost on the 21-feature augmented set.

    This follows the XGBOD principle: supervised learning
    augmented with unsupervised anomaly information.
    """
    logger.info("\n── IF + XGBoost Hybrid (proposed model) ──────────────")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos
    logger.info(f"  scale_pos_weight = {scale_pos_weight:.1f}")

    # Step 1 — Fit Isolation Forest on training features only
    logger.info("  Fitting Isolation Forest component ...")
    t0 = time.time()
    if_component = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if_component.fit(X_train)
    if_time = time.time() - t0
    logger.info(f"  IF component trained in {if_time:.1f}s")

    # Step 2 — Augment feature matrices
    X_train_aug = add_if_score(if_component, X_train)
    X_val_aug   = add_if_score(if_component, X_val)

    logger.info(f"  Augmented feature shape: {X_train_aug.shape} "
                f"(+1 IF anomaly score column)")

    # Step 3 — Train XGBoost on augmented features
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    logger.info("  Training XGBoost on augmented features ...")
    t1 = time.time()
    xgb_component = XGBClassifier(**params)
    xgb_component.fit(
        X_train_aug, y_train,
        eval_set=[(X_val_aug, y_val)],
        verbose=False,
    )
    xgb_time = time.time() - t1
    total_time = time.time() - t0

    proba     = xgb_component.predict_proba(X_val_aug)[:, 1]
    threshold = select_threshold_f1(y_val, proba)
    val_pr    = average_precision_score(y_val, proba)
    logger.info(f"  XGBoost component trained in {xgb_time:.1f}s")
    logger.info(f"  Total hybrid training time: {total_time:.1f}s")
    logger.info(f"  Val PR-AUC: {val_pr:.4f}")

    # Save both components
    joblib.dump(if_component,  HYBRID_IF_PATH)
    joblib.dump(xgb_component, HYBRID_MODEL_PATH)
    logger.info(f"  Saved IF component  → {HYBRID_IF_PATH}")
    logger.info(f"  Saved XGBoost component → {HYBRID_MODEL_PATH}")

    return if_component, xgb_component, threshold, total_time


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────
    logger.info(f"Loading {FEATURE_DATA_PATH} ...")
    df = pd.read_parquet(FEATURE_DATA_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(df)
    del df

    all_results = []

    # ── Load previously evaluated models from baseline run ────────────
    # Load their saved thresholds so comparison is consistent
    threshold_path = MODEL_DIR / "thresholds.json"
    with open(threshold_path) as f:
        saved_thresholds = json.load(f)

    # Load and re-score IF, RF, XGB on test set for unified table
    logger.info("\n── Re-scoring baseline models on test set ────────────")

    if_model = joblib.load(IF_MODEL_PATH)
    if_scores_test = -if_model.decision_function(X_test)
    t0 = time.time(); _ = -if_model.decision_function(X_test[:1000]); inf_if = (time.time()-t0)/1000*1000
    all_results.append(compute_metrics(
        "Isolation Forest", y_test, if_scores_test,
        saved_thresholds["isolation_forest"], 12.2, inf_if,
    ))

    rf_model = joblib.load(MODEL_DIR / "random_forest.joblib")
    rf_scores_test = rf_model.predict_proba(X_test)[:, 1]
    t0 = time.time(); _ = rf_model.predict_proba(X_test[:1000]); inf_rf = (time.time()-t0)/1000*1000
    all_results.append(compute_metrics(
        "Random Forest", y_test, rf_scores_test,
        saved_thresholds["random_forest"], 157.8, inf_rf,
    ))

    xgb_model = joblib.load(MODEL_DIR / "xgboost.joblib")
    xgb_scores_test = xgb_model.predict_proba(X_test)[:, 1]
    t0 = time.time(); _ = xgb_model.predict_proba(X_test[:1000]); inf_xgb = (time.time()-t0)/1000*1000
    all_results.append(compute_metrics(
        "XGBoost", y_test, xgb_scores_test,
        saved_thresholds["xgboost"], 318.3, inf_xgb,
    ))

    # ── Logistic Regression ───────────────────────────────────────────
    lr_model, scaler, lr_threshold, lr_time = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )
    X_test_s = scaler.transform(X_test)
    lr_scores_test = lr_model.predict_proba(X_test_s)[:, 1]
    t0 = time.time(); _ = lr_model.predict_proba(X_test_s[:1000]); inf_lr = (time.time()-t0)/1000*1000
    all_results.append(compute_metrics(
        "Logistic Regression", y_test, lr_scores_test,
        lr_threshold, lr_time, inf_lr,
    ))

    # ── Decision Tree ─────────────────────────────────────────────────
    dt_model, dt_threshold, dt_time = train_decision_tree(
        X_train, y_train, X_val, y_val
    )
    dt_scores_test = dt_model.predict_proba(X_test)[:, 1]
    t0 = time.time(); _ = dt_model.predict_proba(X_test[:1000]); inf_dt = (time.time()-t0)/1000*1000
    all_results.append(compute_metrics(
        "Decision Tree", y_test, dt_scores_test,
        dt_threshold, dt_time, inf_dt,
    ))

    # ── IF + XGBoost Hybrid ───────────────────────────────────────────
    if_comp, xgb_comp, hybrid_threshold, hybrid_time = train_if_xgboost_hybrid(
        X_train, y_train, X_val, y_val
    )

    X_test_aug         = add_if_score(if_comp, X_test)
    hybrid_scores_test = xgb_comp.predict_proba(X_test_aug)[:, 1]
    t0 = time.time(); _ = xgb_comp.predict_proba(add_if_score(if_comp, X_test[:1000])); inf_h = (time.time()-t0)/1000*1000
    all_results.append(compute_metrics(
        "IF + XGBoost (Proposed)", y_test, hybrid_scores_test,
        hybrid_threshold, hybrid_time, inf_h,
    ))

    # ── Save updated thresholds ───────────────────────────────────────
    saved_thresholds["if_xgboost_hybrid"]     = hybrid_threshold
    saved_thresholds["logistic_regression"]   = lr_threshold
    saved_thresholds["decision_tree"]         = dt_threshold
    with open(threshold_path, "w") as f:
        json.dump(saved_thresholds, f, indent=2)

    # ── Save full comparison table ────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    csv_path = METRICS_DIR / "full_model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nFull comparison saved → {csv_path}")

    # ── Print final table ─────────────────────────────────────────────
    display_cols = [
        "model", "pr_auc", "roc_auc", "precision", "recall",
        "f1", "fpr", "fnr", "tp", "fp", "fn",
        "train_time_s", "inference_ms_per_txn"
    ]

    print("\n" + "=" * 110)
    print("FULL MODEL COMPARISON — TEST SET")
    print("=" * 110)
    print(results_df[display_cols].to_string(index=False))
    print("=" * 110)

    # Identify best by PR-AUC
    best = results_df.loc[results_df["pr_auc"].idxmax(), "model"]
    logger.info(f"\nBest model by PR-AUC: {best}")
    logger.info("Next step: run src.explain to generate SHAP explanations.")


if __name__ == "__main__":
    main()