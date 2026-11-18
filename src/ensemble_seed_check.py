"""
ensemble_seed_check.py
----------------------
Robustness check for V2-11 (Random Forest + XGBoost soft voting).

V2-11 was significantly better than XGBoost (V2-10) on a single training run
(seed 42). The five-seed check showed seed 42 was XGBoost's weakest run, so
the gain may reflect an unlucky XGBoost model rather than a genuine benefit of
combining the two. Soft voting is therefore held to the same standard the
Isolation Forest hybrid had to meet.

For each seed, Random Forest and XGBoost (square-root class weighting) are
retrained on FS2_clean, and the test PR-AUC of XGBoost alone is compared with
the PR-AUC of the averaged probabilities. Random Forest alone is reported too.

Decision rule (fixed BEFORE running):
  CONFIRMED     -> soft voting beats XGBoost alone in >= 4 of 5 seeds
                   AND the mean difference is positive
  NOT CONFIRMED -> otherwise; XGBoost alone stays final (simpler, cheaper)

Run:
    python -m src.ensemble_seed_check
"""

import logging
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from .config import METRICS_DIR
from .run_experiments import XGBScorer, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 7, 123, 2024, 31337]          # same seeds as seed_robustness.py
FEATURE_SET = "FS2_clean"
OUTPUT_PATH = METRICS_DIR / "ensemble_seed_robustness.csv"
MIN_WINS = 4


def random_forest(seed: int) -> RandomForestClassifier:
    """Same configuration as V2-04, with a variable seed."""
    return RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        class_weight="balanced_subsample", max_samples=300_000,
        n_jobs=-1, random_state=seed,
    )


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    X_cache, y = load_splits()
    X = X_cache[FEATURE_SET]

    rows = []
    for seed in SEEDS:
        logger.info("=" * 60)
        logger.info(f"Seed {seed}")

        t0 = time.time()
        xgb = XGBScorer("sqrt", seed).fit(X["train"], y["train"], X["val"], y["val"])
        xgb_t = time.time() - t0

        t0 = time.time()
        rf = random_forest(seed).fit(X["train"], y["train"])
        rf_t = time.time() - t0

        p_xgb = xgb.score(X["test"])
        p_rf = rf.predict_proba(X["test"])[:, 1]
        p_vote = (p_xgb + p_rf) / 2

        ap_xgb = average_precision_score(y["test"], p_xgb)
        ap_rf = average_precision_score(y["test"], p_rf)
        ap_vote = average_precision_score(y["test"], p_vote)
        delta = ap_vote - ap_xgb
        rows.append({
            "seed": seed,
            "xgb_pr_auc": round(ap_xgb, 4),
            "rf_pr_auc": round(ap_rf, 4),
            "soft_voting_pr_auc": round(ap_vote, 4),
            "delta_vs_xgb": round(delta, 4),
            "soft_voting_wins": delta > 0,
            "xgb_train_s": round(xgb_t, 1),
            "rf_train_s": round(rf_t, 1),
        })
        logger.info(f"  XGBoost {ap_xgb:.4f} | RF {ap_rf:.4f} | Soft voting {ap_vote:.4f} | "
                    f"delta vs XGBoost {delta:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    wins = int(df["soft_voting_wins"].sum())
    mean_delta = df["delta_vs_xgb"].mean()
    confirmed = wins >= MIN_WINS and mean_delta > 0

    print("\n" + "=" * 100)
    print("SEED ROBUSTNESS — RF + XGBoost soft voting vs XGBoost alone (test PR-AUC)")
    print("=" * 100)
    print(df.to_string(index=False))
    print("-" * 100)
    for col, label in (("xgb_pr_auc", "XGBoost"), ("rf_pr_auc", "Random Forest"),
                       ("soft_voting_pr_auc", "Soft voting")):
        print(f"{label:<14} mean PR-AUC: {df[col].mean():.4f} (sd {df[col].std(ddof=1):.4f})")
    print(f"Mean delta (soft voting - XGBoost): {mean_delta:+.4f} (sd {df['delta_vs_xgb'].std(ddof=1):.4f})")
    print(f"Soft voting wins: {wins} of {len(SEEDS)} seeds")
    print("-" * 100)
    if confirmed:
        print("VERDICT: CONFIRMED — the soft-voting gain is robust to training randomness.\n"
              "Next: python -m src.calibrate_final --source soft_voting")
    else:
        print("VERDICT: NOT CONFIRMED — the gain is not stable across seeds. XGBoost alone stays final.\n"
              "The artefacts already in models/final/ are the correct ones.")
    print("=" * 100)
    logger.info(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
