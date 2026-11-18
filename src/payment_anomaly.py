"""
payment_anomaly.py
------------------
Advisory anomaly screening for PAYMENT transactions (merchant, till and paybill payments).

Why a separate, unsupervised model:
  PaySim contains no fraud labels for PAYMENT, so the supervised ensemble cannot be
  trained or validated on it. Payments to impersonated merchants are nevertheless a
  common social engineering route in Kenya (Section 5.9.5). This module screens each
  payment with an Isolation Forest trained on normal payment behaviour, plus four
  transparent merchant rules, and marks unusual payments "REVIEW_SUGGESTED".

Why it is ADVISORY (never holds or declines a payment):
  Without labels its precision cannot be measured. The thesis applies the same rule
  to the rules engine: nothing may obstruct a customer unless its precision has been
  shown on validation data. A suggested review goes to the analyst payment queue;
  the customer's payment is approved as before.

Features (from the same feature store as the main model, history BEFORE the payment):
  amount, log_amount, hour_of_day, and the merchant's history: count, total and
  average received, time since its last payment received, whether it is new, and the
  payment relative to its average. (type_TRANSFER is not used: every row is a PAYMENT.)

Signals (shown to the analyst as plain-language reasons):
  NEW_MERCHANT_LARGE    first payment ever received by this merchant, and the amount
                        is in the top 5% of training payments
  NIGHT_LARGE           made between 00:00 and 05:59, amount in the top 5%
  AMOUNT_SPIKE          merchant has >= 3 earlier payments and this one is >= 10x its average
  DRAINS_BALANCE        payment is >= 70% of the payer's balance (only when the balance is supplied)

  REVIEW_SUGGESTED if the anomaly score is at or above the 99.5th percentile of
  validation payments, OR two or more signals fire.

Leakage safety: the Isolation Forest is fitted on training-period payments (steps
1-520); the review threshold and percentile scale come from validation payments
(521-631); test payments (632-743) are only used for the report.

Train (writes models/payment/):
    python -m src.payment_anomaly --train
Score one example (the user acceptance test case):
    python -m src.payment_anomaly --example
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import MODEL_DIR, RAW_DATA_PATH, VAL_MAX_STEP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TRAIN_MAX_STEP = 520
PAYMENT_DIR = MODEL_DIR / "payment"
MODEL_PATH = PAYMENT_DIR / "payment_iforest.joblib"
CONFIG_PATH = PAYMENT_DIR / "payment_config.json"

PAY_FEATURES = [
    "amount", "log_amount", "hour_of_day",
    "dest_txn_count_before", "dest_total_amount_before", "dest_avg_amount_before",
    "dest_time_since_last_step", "dest_is_new", "dest_amount_to_avg_ratio",
]
REVIEW_PERCENTILE = 99.5
FIT_SAMPLE = 400_000
SEED = 42

SIGNAL_TEXT = {
    "NEW_MERCHANT_LARGE": "First payment ever received by this merchant, and the amount is unusually large",
    "NIGHT_LARGE": "Large payment made between 00:00 and 05:59",
    "AMOUNT_SPIKE": "Payment is at least 10 times this merchant's usual amount",
    "DRAINS_BALANCE": "Payment uses at least 70% of the payer's balance",
}


# ── Feature engineering for training (mirrors FeatureStore.compute_features) ──
def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Destination history BEFORE each row, over all transaction types, in file order."""
    g = df.groupby("nameDest", sort=False)
    count = g.cumcount()
    running = g["amount"].cumsum()
    total = running.groupby(df["nameDest"], sort=False).shift(1).fillna(0.0)
    last = g["step"].shift(1)
    total32 = total.astype("float32")
    avg32 = np.where(count > 0, total32 / count.clip(lower=1), 0.0).astype("float32")
    amount = df["amount"].astype("float64")
    out = pd.DataFrame({
        "amount": amount.astype("float32"),
        "log_amount": np.log1p(amount.clip(lower=0)).astype("float32"),
        "hour_of_day": (df["step"] % 24).astype("int16"),
        "dest_txn_count_before": count.astype("int32"),
        "dest_total_amount_before": total32,
        "dest_avg_amount_before": avg32,
        "dest_time_since_last_step": np.where(count > 0, df["step"] - last.fillna(0), 0).astype("int32"),
        "dest_is_new": (count == 0).astype("int8"),
        "dest_amount_to_avg_ratio": np.where(avg32 > 0, amount / np.where(avg32 > 0, avg32, 1.0), 0.0)
                                     .astype("float32"),
    }, index=df.index)
    return out


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (negated score_samples)."""
    return -model.score_samples(X)


def train() -> None:
    t0 = time.time()
    logger.info(f"Reading {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH, usecols=["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest"],
                     dtype={"step": "int32", "type": "category", "amount": "float64",
                            "nameOrig": "object", "oldbalanceOrg": "float64", "nameDest": "object"})
    feats = history_features(df)
    pay = df["type"] == "PAYMENT"
    feats, meta = feats[pay], df.loc[pay, ["step", "amount", "oldbalanceOrg"]]
    del df
    tr, va, te = meta["step"] <= TRAIN_MAX_STEP, meta["step"].between(TRAIN_MAX_STEP + 1, VAL_MAX_STEP), \
        meta["step"] > VAL_MAX_STEP
    logger.info(f"PAYMENT rows: train {tr.sum():,}, validation {va.sum():,}, test {te.sum():,}")

    X_tr = feats.loc[tr, PAY_FEATURES].to_numpy("float32")
    if len(X_tr) > FIT_SAMPLE:
        X_fit = X_tr[np.random.default_rng(SEED).choice(len(X_tr), FIT_SAMPLE, replace=False)]
    else:
        X_fit = X_tr
    model = IsolationForest(n_estimators=200, max_samples=256, contamination="auto",
                            random_state=SEED, n_jobs=-1).fit(X_fit)

    s_va = anomaly_scores(model, feats.loc[va, PAY_FEATURES].to_numpy("float32"))
    s_te = anomaly_scores(model, feats.loc[te, PAY_FEATURES].to_numpy("float32"))
    threshold = float(np.percentile(s_va, REVIEW_PERCENTILE))
    grid = np.percentile(s_va, np.arange(0, 100.5, 0.5)).tolist()

    amt_tr = meta.loc[tr, "amount"]
    config = {
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "features": PAY_FEATURES,
        "trained_on_steps": [1, TRAIN_MAX_STEP],
        "threshold_from_steps": [TRAIN_MAX_STEP + 1, VAL_MAX_STEP],
        "review_percentile": REVIEW_PERCENTILE,
        "review_threshold": threshold,
        "percentile_grid": grid,
        "amount_p95": float(amt_tr.quantile(0.95)),
        "amount_p99": float(amt_tr.quantile(0.99)),
        "amount_median": float(amt_tr.median()),
        "n_estimators": 200,
        "max_samples": 256,
    }
    PAYMENT_DIR.mkdir(parents=True, exist_ok=True)
    model.set_params(n_jobs=1)
    joblib.dump(model, MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    # ── Report on the test period (labels do not exist; this is a behaviour report) ──
    scorer = PaymentScorer()
    te_feats = feats.loc[te].copy()
    te_meta = meta.loc[te]
    flags = []
    for row_f, row_m, s in zip(te_feats[PAY_FEATURES].to_dict("records"), te_meta.to_dict("records"), s_te):
        sig = scorer.signals(row_f, row_m["oldbalanceOrg"])
        flags.append((s >= threshold) or len(sig) >= 2)
    flags = np.array(flags)
    logger.info(f"Isolation Forest fitted on {len(X_fit):,} training payments in {time.time() - t0:.0f}s")
    print("\n" + "=" * 78)
    print("PAYMENT ANOMALY SCREENING (advisory; PaySim has no PAYMENT fraud labels)")
    print("=" * 78)
    print(f"Review threshold (validation {REVIEW_PERCENTILE}th percentile): {threshold:.4f}")
    print(f"Training payment amounts: median {config['amount_median']:,.0f}, "
          f"95th pct {config['amount_p95']:,.0f}, 99th pct {config['amount_p99']:,.0f}")
    print(f"Test payments: {len(s_te):,}; review suggested for {flags.sum():,} "
          f"({100 * flags.mean():.2f}%)")
    print(f"  of which by anomaly score: {(s_te >= threshold).sum():,}")
    print(f"Saved {MODEL_PATH} and {CONFIG_PATH}")
    print("=" * 78)


# ── Live scoring ─────────────────────────────────────────────────────────────
class PaymentScorer:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.config = json.loads(CONFIG_PATH.read_text())
        self.grid = np.array(self.config["percentile_grid"])
        self.version = f"payment-iforest ({self.config['created']})"

    @staticmethod
    def available() -> bool:
        return MODEL_PATH.exists() and CONFIG_PATH.exists()

    def signals(self, f: dict, old_balance: float | None) -> list[str]:
        amount, p95 = float(f["amount"]), self.config["amount_p95"]
        out = []
        if int(f["dest_is_new"]) == 1 and amount >= p95:
            out.append("NEW_MERCHANT_LARGE")
        if int(f["hour_of_day"]) <= 5 and amount >= p95:
            out.append("NIGHT_LARGE")
        if int(f["dest_txn_count_before"]) >= 3 and float(f["dest_amount_to_avg_ratio"]) >= 10:
            out.append("AMOUNT_SPIKE")
        if old_balance is not None and not pd.isna(old_balance) and float(old_balance) > 0 \
                and amount >= 0.7 * float(old_balance):
            out.append("DRAINS_BALANCE")
        return out

    def assess(self, features: dict, old_balance: float | None = None) -> dict:
        t = time.perf_counter()
        X = np.array([[features[c] for c in PAY_FEATURES]], dtype="float32")
        score = float(anomaly_scores(self.model, X)[0])
        pct = float(np.interp(score, self.grid, np.arange(0, 100.5, 0.5)))
        sig = self.signals(features, old_balance)
        by_score = score >= self.config["review_threshold"]
        review = by_score or len(sig) >= 2
        reasons = [{"reason": s, "description": SIGNAL_TEXT[s]} for s in sig]
        if by_score:
            reasons.insert(0, {"reason": "ANOMALY_SCORE",
                               "description": f"Payment is more unusual than {pct:.1f}% of normal payments"})
        return {
            "review_suggested": bool(review),
            "flag": "REVIEW_SUGGESTED" if review else "NONE",
            "anomaly_score": round(score, 4),
            "anomaly_percentile": round(pct, 1),
            "signals": sig,
            "reasons": reasons,
            "advisory_only": True,
            "payment_model_version": self.version,
            "payment_ms": round((time.perf_counter() - t) * 1000, 3),
        }


def example() -> None:
    """The user acceptance case: 1,500,000 from a 2,000,000 balance to a new merchant."""
    scorer = PaymentScorer()
    amount = 1_500_000.0
    feats = {"amount": amount, "log_amount": float(np.log1p(amount)), "hour_of_day": 12,
             "dest_txn_count_before": 0, "dest_total_amount_before": 0.0, "dest_avg_amount_before": 0.0,
             "dest_time_since_last_step": 0, "dest_is_new": 1, "dest_amount_to_avg_ratio": 0.0}
    print(json.dumps(scorer.assess(feats, old_balance=2_000_000.0), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--example", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    if args.example:
        example()
    if not (args.train or args.example):
        parser.print_help()


if __name__ == "__main__":
    main()
