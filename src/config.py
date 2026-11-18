"""
config.py
---------
Central configuration — all paths, constants, and parameters.
Never hard-code these values in any other file.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────────────────────────
DATA_DIR      = ROOT / "data"
RAW_DATA_DIR  = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATA_PATH     = RAW_DATA_DIR / "PS_20174392719_1491204439457_log.csv"
FEATURE_DATA_PATH = PROCESSED_DIR / "features.parquet"

# ── Model paths ────────────────────────────────────────────────────────────
MODEL_DIR = ROOT / "models"

IF_MODEL_PATH     = MODEL_DIR / "isolation_forest.joblib"
RF_MODEL_PATH     = MODEL_DIR / "random_forest.joblib"
XGB_MODEL_PATH    = MODEL_DIR / "xgboost.joblib"
THRESHOLD_PATH    = MODEL_DIR / "thresholds.json"
META_PATH         = MODEL_DIR / "model_metadata.json"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.json"

# ── Results paths ──────────────────────────────────────────────────────────
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
SHAP_DIR    = RESULTS_DIR / "shap"

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Temporal split boundaries (by step) ───────────────────────────────────
TRAIN_MAX_STEP = 520
VAL_MAX_STEP   = 631

# ── Class imbalance ────────────────────────────────────────────────────────
SCALE_POS_WEIGHT = 577

# ── Risk engine thresholds ─────────────────────────────────────────────────
RISK_LOW_MAX    = 0.30
RISK_MEDIUM_MAX = 0.60
RISK_HIGH_MAX   = 0.85

# ── Rules engine ──────────────────────────────────────────────────────────
LARGE_TRANSFER_THRESHOLD = 200_000.0
RAPID_TX_WINDOW_STEPS    = 1
RAPID_TX_COUNT_LIMIT     = 5

# ── Model hyperparameters ──────────────────────────────────────────────────
IF_PARAMS = {
    "n_estimators": 100,
    "contamination": "auto",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "class_weight": "balanced_subsample",
    "max_samples": 50_000,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

# ── SHAP ──────────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 5_000
SHAP_TOP_N       = 5