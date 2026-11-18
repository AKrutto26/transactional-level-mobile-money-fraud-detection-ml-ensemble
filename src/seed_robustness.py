"""
seed_robustness.py
------------------
Confirms whether the Isolation Forest anomaly score genuinely improves
XGBoost, or whether the V2-07 vs V2-10 gain was luck of random seed 42.

The paired bootstrap in run_experiments.py measures uncertainty from the
test sample only. This script measures uncertainty from TRAINING
randomness (XGBoost row/column subsampling and Isolation Forest tree
construction) by retraining both models across five seeds.

Compared (both on FS2_clean, both sqrt class weighting):
  Hybrid : IF + XGBoost  (the V2-07 configuration)
  Plain  : XGBoost only  (the V2-10 configuration)

Decision rule (fixed BEFORE running):
  CONFIRMED     -> hybrid wins in >= 4 of 5 seeds AND mean delta > 0
  NOT CONFIRMED -> otherwise; plain XGBoost is preferred on simplicity

Run:
    python -m src.seed_robustness
"""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import METRICS_DIR
from .run_experiments import IFXGBHybrid, XGBScorer, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 7, 123, 2024, 31337]
FEATURE_SET = "FS2_clean"
OUTPUT_PATH = METRICS_DIR / "seed_robustness.csv"
MIN_WINS = 4


def train_and_score(model, X, y) -> tuple[float, float]:
    t0 = time.time()
    model.fit(X["train"], y["train"], X["val"], y["val"])
    elapsed = time.time() - t0
    ap = average_precision_score(y["test"], model.score(X["test"]))
    return ap, elapsed


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    X_cache, y = load_splits()
    X = X_cache[FEATURE_SET]

    rows = []
    for seed in SEEDS:
        logger.info("=" * 60)
        logger.info(f"Seed {seed}")
        plain_ap, plain_t = train_and_score(XGBScorer("sqrt", seed), X, y)
        hybrid_ap, hybrid_t = train_and_score(IFXGBHybrid("sqrt", seed), X, y)
        delta = hybrid_ap - plain_ap
        rows.append({
            "seed": seed,
            "plain_xgb_pr_auc": round(plain_ap, 4),
            "hybrid_pr_auc": round(hybrid_ap, 4),
            "delta": round(delta, 4),
            "hybrid_wins": delta > 0,
            "plain_train_s": round(plain_t, 1),
            "hybrid_train_s": round(hybrid_t, 1),
        })
        logger.info(f"  Plain XGBoost {plain_ap:.4f} | Hybrid {hybrid_ap:.4f} | delta {delta:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    wins = int(df["hybrid_wins"].sum())
    mean_delta = df["delta"].mean()
    std_delta = df["delta"].std(ddof=1)
    confirmed = wins >= MIN_WINS and mean_delta > 0

    print("\n" + "=" * 90)
    print("SEED ROBUSTNESS — IF + XGBoost (sqrt) vs XGBoost (sqrt), test PR-AUC")
    print("=" * 90)
    print(df.to_string(index=False))
    print("-" * 90)
    print(f"Plain XGBoost mean PR-AUC : {df['plain_xgb_pr_auc'].mean():.4f} "
          f"(sd {df['plain_xgb_pr_auc'].std(ddof=1):.4f})")
    print(f"Hybrid mean PR-AUC        : {df['hybrid_pr_auc'].mean():.4f} "
          f"(sd {df['hybrid_pr_auc'].std(ddof=1):.4f})")
    print(f"Mean delta                : {mean_delta:+.4f} (sd {std_delta:.4f})")
    print(f"Hybrid wins               : {wins} of {len(SEEDS)} seeds")
    print("-" * 90)
    if confirmed:
        print("VERDICT: CONFIRMED — the IF anomaly score improvement is robust to training "
              "randomness. Final model: IF + XGBoost (sqrt class weight).")
    else:
        print("VERDICT: NOT CONFIRMED — the gain is not stable across seeds. Final model: "
              "XGBoost (sqrt class weight), preferred on simplicity.")
    print("=" * 90)
    logger.info(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()