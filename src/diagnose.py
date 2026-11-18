"""
diagnose.py
-----------
Verifies three suspected weaknesses before the v2 experiments:
  1. Is `step` dominating the model (spurious time-index signal)?
  2. Are origin-history features near-empty (nameOrig ~unique)?
  3. Which transaction types actually contain fraud?

Run:
    python -m src.diagnose
"""

import joblib
import pandas as pd

from .config import FEATURE_DATA_PATH, MODEL_DIR, RAW_DATA_PATH, TRAIN_MAX_STEP, VAL_MAX_STEP
from .features import MODEL_FEATURE_COLUMNS

raw = pd.read_csv(RAW_DATA_PATH, usecols=["step", "type", "nameOrig", "nameDest", "isFraud"])

print("=" * 60)
print("1. ACCOUNT ID CARDINALITY")
print("=" * 60)
print(f"Rows:             {len(raw):,}")
print(f"Unique nameOrig:  {raw['nameOrig'].nunique():,}")
print(f"Unique nameDest:  {raw['nameDest'].nunique():,}")

print("\n" + "=" * 60)
print("2. FRAUD BY TRANSACTION TYPE")
print("=" * 60)
print(raw.groupby("type")["isFraud"].agg(total="size", fraud="sum").to_string())

print("\n" + "=" * 60)
print("3. VOLUME vs FRAUD ACROSS TIME (legit volume collapse check)")
print("=" * 60)
bins = pd.cut(raw["step"], bins=[0, 200, 400, TRAIN_MAX_STEP, VAL_MAX_STEP, 800])
print(raw.groupby(bins, observed=True)["isFraud"].agg(rows="size", fraud="sum").to_string())
del raw

print("\n" + "=" * 60)
print("4. % OF ROWS WHERE EACH FEATURE IS NON-ZERO")
print("=" * 60)
feat = pd.read_parquet(FEATURE_DATA_PATH, columns=MODEL_FEATURE_COLUMNS)
print((feat != 0).mean().mul(100).round(3).sort_values().to_string())
del feat

print("\n" + "=" * 60)
print("5. XGBOOST FEATURE IMPORTANCE (gain) — is `step` dominating?")
print("=" * 60)
xgb = joblib.load(MODEL_DIR / "xgboost.joblib")
imp = pd.Series(xgb.feature_importances_, index=MODEL_FEATURE_COLUMNS)
print(imp.sort_values(ascending=False).round(4).to_string())