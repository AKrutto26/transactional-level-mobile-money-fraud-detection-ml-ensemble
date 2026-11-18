"""
build_features_v2.py
--------------------
Version 2 feature set, informed by the diagnostic findings:

  1. nameOrig is ~99.85% unique -> origin-history features are near-empty
     and are REMOVED.
  2. Fraud occurs only in TRANSFER and CASH_OUT -> modelling is restricted
     to these two types (other types are routed as low-risk by the rules
     engine).
  3. `step` / `day_index` are raw time indices that cannot generalise to
     future periods -> REMOVED as features (step is kept only for splitting).

Destination history is computed on the FULL dataset (all types) before
filtering, so every destination's history includes all prior incoming
transactions.

Two feature sets are produced:
  FS2_COLUMNS               -> primary leakage-safe set
  BALANCE_ABLATION_COLUMNS  -> pre-transaction balance features, used ONLY
                               in the separately labelled ablation (FS3)

Run:
    python -m src.build_features_v2 --nrows 200000   # quick test
    python -m src.build_features_v2                  # full dataset
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_DATA_PATH
from .features import _cumstats_by_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURES_V2_PATH = PROCESSED_DIR / "features_v2.parquet"
MODELLED_TYPES = ["TRANSFER", "CASH_OUT"]

FS2_COLUMNS = [
    "amount",
    "log_amount",
    "hour_of_day",
    "type_TRANSFER",               # CASH_OUT is the complement
    "dest_txn_count_before",
    "dest_total_amount_before",
    "dest_avg_amount_before",
    "dest_time_since_last_step",
    "dest_is_new",
    "dest_amount_to_avg_ratio",
]

BALANCE_ABLATION_COLUMNS = [
    "orig_balance_before",         # oldbalanceOrg: known at authorisation time
    "amount_to_orig_balance",      # share of the balance being moved (-1 if balance is 0)
    "empties_orig_account",        # 1 if the transaction drains the whole balance
]

USECOLS = ["step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "isFraud"]
DTYPES = {
    "step": "int32", "type": "category", "amount": "float64",
    "nameOrig": "object", "nameDest": "object",
    "oldbalanceOrg": "float64", "isFraud": "int8",
}


def build(nrows: int | None = None) -> pd.DataFrame:
    logger.info(f"Loading raw CSV (nrows={nrows or 'all'}) ...")
    df = pd.read_csv(RAW_DATA_PATH, usecols=USECOLS, dtype=DTYPES, nrows=nrows)
    logger.info(f"Loaded {len(df):,} rows")

    # ── Row-level features ────────────────────────────────────────────
    amount = df["amount"].to_numpy()
    df["log_amount"] = np.log1p(np.clip(amount, 0, None)).astype("float32")
    df["hour_of_day"] = (df["step"] % 24).astype("int8")
    type_str = df["type"].astype(str)
    df["type_TRANSFER"] = (type_str == "TRANSFER").astype("int8")

    # ── Destination history on the FULL dataset (strictly prior rows) ──
    logger.info("Computing destination history on full dataset ...")
    stats = _cumstats_by_group(df["nameDest"].to_numpy(), df["step"].to_numpy(), amount)
    ri = stats["restore_idx"]
    df["dest_txn_count_before"] = stats["txn_count_before"][ri]
    df["dest_total_amount_before"] = stats["total_amount_before"][ri]
    df["dest_avg_amount_before"] = stats["avg_amount_before"][ri]
    df["dest_time_since_last_step"] = stats["time_since_last"][ri]
    df["dest_is_new"] = (df["dest_txn_count_before"] == 0).astype("int8")

    avg = df["dest_avg_amount_before"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["dest_amount_to_avg_ratio"] = np.where(avg > 0, amount / avg, 0.0).astype("float32")

    # ── Balance ablation features (NOT used in the primary model) ──────
    bal = df["oldbalanceOrg"].to_numpy()
    df["orig_balance_before"] = bal.astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["amount_to_orig_balance"] = np.where(bal > 0, amount / bal, -1.0).astype("float32")
    df["empties_orig_account"] = ((bal > 0) & (amount >= bal)).astype("int8")

    df["amount"] = df["amount"].astype("float32")

    # ── Restrict to the transaction types where fraud occurs ───────────
    before = len(df)
    df = df[type_str.isin(MODELLED_TYPES).to_numpy()].copy()
    logger.info(f"Filtered to {MODELLED_TYPES}: {len(df):,} of {before:,} rows kept")

    keep = ["step"] + FS2_COLUMNS + BALANCE_ABLATION_COLUMNS + ["isFraud"]
    df = df[keep].reset_index(drop=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    df = build(args.nrows)
    fraud = int(df["isFraud"].sum())
    logger.info(f"Final shape: {df.shape}  fraud={fraud:,}  rate={fraud / len(df) * 100:.4f}%")
    logger.info(f"Nulls: {int(df.isnull().sum().sum())}")
    logger.info("% non-zero per feature:\n"
                + (df[FS2_COLUMNS + BALANCE_ABLATION_COLUMNS] != 0).mean().mul(100).round(2).to_string())

    out = FEATURES_V2_PATH if args.nrows is None else PROCESSED_DIR / "features_v2_test.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Saved -> {out}  ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
