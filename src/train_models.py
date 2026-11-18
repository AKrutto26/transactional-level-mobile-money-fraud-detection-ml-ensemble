"""
train_models.py
---------------
Trains three baseline models on the PaySim feature dataset:
  - Model A: Isolation Forest (unsupervised anomaly detection)
  - Model B: Random Forest (supervised classification)
  - Model C: XGBoost (supervised classification)

Uses temporal train/validation/test split from split_data.py.
Saves all trained models and metadata to models/.

Run:
    python -m src.train_models
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from .config import (
    FEATURE_DATA_PATH,
    IF_MODEL_PATH,
    MODEL_DIR,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    META_PATH,
    FEATURE_COLS_PATH,
    RANDOM_STATE,
    IF_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
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
    """Temporal train / validation / test split by step."""
    train = df[df["step"] <= TRAIN_MAX_STEP]
    val   = df[(df["step"] > TRAIN_MAX_STEP) & (df["step"] <= VAL_MAX_STEP)]
    test  = df[df["step"] > VAL_MAX_STEP]

    logger.info(f"Train : {len(train):>9,} rows  fraud={int(train['isFraud'].sum()):,}")
    logger.info(f"Val   : {len(val):>9,} rows  fraud={int(val['isFraud'].sum()):,}")
    logger.info(f"Test  : {len(test):>9,} rows  fraud={int(test['isFraud'].sum()):,}")

    X_train = train[MODEL_FEATURE_COLUMNS].values.astype("float32")
    y_train = train["isFraud"].values.astype("int8")
    X_val   = val[MODEL_FEATURE_COLUMNS].values.astype("float32")
    y_val   = val["isFraud"].values.astype("int8")
    X_test  = test[MODEL_FEATURE_COLUMNS].values.astype("float32")
    y_test  = test["isFraud"].values.astype("int8")

    return X_train, y_train, X_val, y_val, X_test, y_test


def peak_memory_mb() -> float:
    """Current process RSS memory in MB."""
    return psutil.Process().memory_info().rss / 1e6


def train_isolation_forest(X_train, y_val, X_val):
    """
    Train Isolation Forest — unsupervised, no labels used.
    Evaluate using anomaly scores on validation set.
    """
    logger.info("=" * 55)
    logger.info("MODEL A — Isolation Forest")
    logger.info("=" * 55)

    mem_before = peak_memory_mb()
    t0 = time.time()

    model = IsolationForest(**IF_PARAMS)
    model.fit(X_train)

    elapsed = time.time() - t0
    mem_after = peak_memory_mb()

    # Anomaly score: higher = more anomalous (invert IF convention)
    # IF returns negative scores where more negative = more anomalous
    raw_scores  = model.decision_function(X_val)        # negative = anomalous
    anomaly_scores = -raw_scores                         # flip: higher = more fraud-like

    pr_auc = average_precision_score(y_val, anomaly_scores)

    logger.info(f"  Training time : {elapsed:.1f}s")
    logger.info(f"  Memory delta  : {mem_after - mem_before:.0f} MB")
    logger.info(f"  Val PR-AUC    : {pr_auc:.4f}")

    return model, {
        "model":          "isolation_forest",
        "training_time_s": round(elapsed, 2),
        "memory_delta_mb": round(mem_after - mem_before, 1),
        "val_pr_auc":     round(pr_auc, 4),
        "params":         IF_PARAMS,
    }


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest with class balancing."""
    logger.info("=" * 55)
    logger.info("MODEL B — Random Forest")
    logger.info("=" * 55)

    # Compute scale weight for reference
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    logger.info(f"  Train class balance: {neg:,} legit / {pos:,} fraud  ratio={neg//pos}:1")

    mem_before = peak_memory_mb()
    t0 = time.time()

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    elapsed = time.time() - t0
    mem_after = peak_memory_mb()

    proba   = model.predict_proba(X_val)[:, 1]
    pr_auc  = average_precision_score(y_val, proba)

    logger.info(f"  Training time : {elapsed:.1f}s")
    logger.info(f"  Memory delta  : {mem_after - mem_before:.0f} MB")
    logger.info(f"  Val PR-AUC    : {pr_auc:.4f}")

    return model, {
        "model":           "random_forest",
        "training_time_s": round(elapsed, 2),
        "memory_delta_mb": round(mem_after - mem_before, 1),
        "val_pr_auc":      round(pr_auc, 4),
        "train_neg":       neg,
        "train_pos":       pos,
        "params":          RF_PARAMS,
    }


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with scale_pos_weight for class imbalance."""
    logger.info("=" * 55)
    logger.info("MODEL C — XGBoost")
    logger.info("=" * 55)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos
    logger.info(f"  scale_pos_weight = {scale_pos_weight:.1f}")

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    mem_before = peak_memory_mb()
    t0 = time.time()

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    elapsed = time.time() - t0
    mem_after = peak_memory_mb()

    proba  = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, proba)

    logger.info(f"  Training time : {elapsed:.1f}s")
    logger.info(f"  Memory delta  : {mem_after - mem_before:.0f} MB")
    logger.info(f"  Val PR-AUC    : {pr_auc:.4f}")

    return model, {
        "model":             "xgboost",
        "training_time_s":   round(elapsed, 2),
        "memory_delta_mb":   round(mem_after - mem_before, 1),
        "val_pr_auc":        round(pr_auc, 4),
        "scale_pos_weight":  round(scale_pos_weight, 1),
        "params":            params,
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load feature dataset ──────────────────────────────────────────
    logger.info(f"Loading features from {FEATURE_DATA_PATH} ...")
    df = pd.read_parquet(FEATURE_DATA_PATH)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Split ─────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(df)
    del df  # free memory

    # ── Save feature column list ──────────────────────────────────────
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(MODEL_FEATURE_COLUMNS, f, indent=2)
    logger.info(f"Feature columns saved → {FEATURE_COLS_PATH}")

    # ── Train models ──────────────────────────────────────────────────
    all_meta = []

    # Model A — Isolation Forest
    if_model, if_meta = train_isolation_forest(X_train, y_val, X_val)
    joblib.dump(if_model, IF_MODEL_PATH)
    logger.info(f"  Saved → {IF_MODEL_PATH}")
    all_meta.append(if_meta)

    # Model B — Random Forest
    rf_model, rf_meta = train_random_forest(X_train, y_train, X_val, y_val)
    joblib.dump(rf_model, RF_MODEL_PATH)
    logger.info(f"  Saved → {RF_MODEL_PATH}")
    all_meta.append(rf_meta)

    # Model C — XGBoost
    xgb_model, xgb_meta = train_xgboost(X_train, y_train, X_val, y_val)
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    logger.info(f"  Saved → {XGB_MODEL_PATH}")
    all_meta.append(xgb_meta)

    # ── Save metadata ─────────────────────────────────────────────────
    with open(META_PATH, "w") as f:
        json.dump(all_meta, f, indent=2)
    logger.info(f"Model metadata saved → {META_PATH}")

    # ── Quick summary ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 55)
    logger.info(f"{'Model':<22} {'Val PR-AUC':>12} {'Train Time':>12}")
    logger.info("-" * 48)
    for m in all_meta:
        logger.info(
            f"{m['model']:<22} {m['val_pr_auc']:>12.4f} {m['training_time_s']:>10.1f}s"
        )
    logger.info("=" * 55)
    logger.info("Training complete. Run src.evaluate_models next.")


if __name__ == "__main__":
    main()