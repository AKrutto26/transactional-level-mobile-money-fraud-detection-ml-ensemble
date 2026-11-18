"""
run_experiments.py
------------------
Version 2 experiment runner. Every run is logged to
results/metrics/experiment_log.csv for the Chapter 5 evidence table.

Protocol (identical for every experiment):
  - Temporal split: train <= step 520 < val <= step 631 < test
  - Early stopping (XGBoost) and threshold selection use VALIDATION only
  - Test set is scored once per experiment and never used for tuning
  - PR-AUC reported with a bootstrap 95% confidence interval
  - Paired bootstrap tests check whether key differences are real

Run all experiments:
    python -m src.run_experiments
Run selected experiments only:
    python -m src.run_experiments --only V2-05 V2-06
Recompute paired comparisons only (no training):
    python -m src.run_experiments --compare-only

Changelog:
  v2.1 — added V2-10 (XGBoost, sqrt weight, no IF) to isolate the IF
         contribution under matched weighting; paired tests now report a
         directional verdict (BETTER / WORSE / no significant difference).
  v2.2 — --only now fails loudly on unknown experiment IDs instead of
         silently training nothing.
  v2.3 — XGBScorer and IFXGBHybrid accept a `seed` argument (default 42,
         so all logged V2 results are unchanged) for the seed-robustness
         check in seed_robustness.py.
"""

import argparse
import logging
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .build_features_v2 import BALANCE_ABLATION_COLUMNS, FEATURES_V2_PATH, FS2_COLUMNS
from .config import METRICS_DIR, MODEL_DIR, RANDOM_STATE, TRAIN_MAX_STEP, VAL_MAX_STEP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

V2_MODEL_DIR = MODEL_DIR / "v2"
LOG_PATH = METRICS_DIR / "experiment_log.csv"
SCORES_PATH = METRICS_DIR / "v2_test_scores.parquet"
PAIRED_PATH = METRICS_DIR / "paired_comparisons.csv"
N_BOOTSTRAP = 500

FEATURE_SETS = {
    "FS2_clean": FS2_COLUMNS,
    "FS3_balance_ablation": FS2_COLUMNS + BALANCE_ABLATION_COLUMNS,
}

XGB_BASE = dict(
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)


# ── Model wrappers: common fit(X_tr, y_tr, X_va, y_va) / score(X) API ──────
class SklearnClassifier:
    def __init__(self, estimator):
        self.est = estimator

    def fit(self, X_tr, y_tr, X_va, y_va):
        self.est.fit(X_tr, y_tr)
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


class IsoForestScorer:
    """Unsupervised: labels are ignored during fitting."""
    def __init__(self):
        self.est = IsolationForest(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

    def fit(self, X_tr, y_tr, X_va, y_va):
        self.est.fit(X_tr)
        return self

    def score(self, X):
        return -self.est.decision_function(X)   # higher = more anomalous


class XGBScorer:
    def __init__(self, weight_mode: str = "ratio", seed: int = RANDOM_STATE):
        self.weight_mode = weight_mode
        self.seed = seed
        self.best_iteration = None

    def fit(self, X_tr, y_tr, X_va, y_va):
        ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        spw = ratio if self.weight_mode == "ratio" else np.sqrt(ratio)
        params = {**XGB_BASE, "random_state": self.seed}
        self.est = XGBClassifier(**params, scale_pos_weight=spw)
        self.est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        self.best_iteration = int(self.est.best_iteration)
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


class IFXGBHybrid:
    """Proposed model: IF anomaly score appended as a feature for XGBoost."""
    def __init__(self, weight_mode: str = "ratio", seed: int = RANDOM_STATE):
        self.iforest = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
        self.xgb = XGBScorer(weight_mode, seed)

    def _augment(self, X):
        anomaly = -self.iforest.decision_function(X).astype("float32")
        return np.hstack([X, anomaly.reshape(-1, 1)])

    def fit(self, X_tr, y_tr, X_va, y_va):
        self.iforest.fit(X_tr)                       # fitted on training data only
        self.xgb.fit(self._augment(X_tr), y_tr, self._augment(X_va), y_va)
        self.best_iteration = self.xgb.best_iteration
        return self

    def score(self, X):
        return self.xgb.score(self._augment(X))


# ── Experiment registry ────────────────────────────────────────────────────
EXPERIMENTS = [
    ("V2-01", "FS2_clean", "Logistic Regression",
     lambda: SklearnClassifier(make_pipeline(
         StandardScaler(),
         LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)))),
    ("V2-02", "FS2_clean", "Decision Tree",
     lambda: SklearnClassifier(DecisionTreeClassifier(
         max_depth=10, class_weight="balanced", random_state=RANDOM_STATE))),
    ("V2-03", "FS2_clean", "Isolation Forest", IsoForestScorer),
    ("V2-04", "FS2_clean", "Random Forest",
     lambda: SklearnClassifier(RandomForestClassifier(
         n_estimators=200, max_depth=20, min_samples_leaf=5,
         class_weight="balanced_subsample", max_samples=300_000,
         n_jobs=-1, random_state=RANDOM_STATE))),
    ("V2-05", "FS2_clean", "XGBoost", lambda: XGBScorer("ratio")),
    ("V2-06", "FS2_clean", "IF + XGBoost (Proposed)", lambda: IFXGBHybrid("ratio")),
    ("V2-07", "FS2_clean", "IF + XGBoost (sqrt class weight)", lambda: IFXGBHybrid("sqrt")),
    ("V2-08", "FS3_balance_ablation", "XGBoost [ablation]", lambda: XGBScorer("ratio")),
    ("V2-09", "FS3_balance_ablation", "IF + XGBoost [ablation]", lambda: IFXGBHybrid("ratio")),
    ("V2-10", "FS2_clean", "XGBoost (sqrt class weight)", lambda: XGBScorer("sqrt")),
]

# (candidate, reference, question answered)
PAIRED_TESTS = [
    ("V2-07", "V2-10", "Does the IF score help XGBoost (both sqrt-weighted)?"),
    ("V2-06", "V2-05", "Does the IF score help XGBoost (both ratio-weighted)?"),
    ("V2-10", "V2-05", "Does sqrt weighting help plain XGBoost?"),
    ("V2-10", "V2-04", "XGBoost (sqrt) vs Random Forest"),
    ("V2-07", "V2-04", "Hybrid (sqrt) vs Random Forest"),
    ("V2-09", "V2-06", "How much do balance features add (ablation)?"),
]


# ── Evaluation helpers ─────────────────────────────────────────────────────
def select_threshold(y_val, scores_val) -> float:
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
    return float(thresholds[int(np.argmax(f1[:-1]))])


def metrics_at(y, scores, threshold) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_indices(n, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    for _ in range(N_BOOTSTRAP):
        yield rng.integers(0, n, n)


def bootstrap_ap_ci(y, scores) -> tuple[float, float]:
    aps = [average_precision_score(y[i], scores[i]) for i in bootstrap_indices(len(y))]
    return float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))


def paired_bootstrap(y, s_a, s_b) -> dict:
    """Same resampled rows for both models, so the difference CI is paired."""
    diffs = np.array([
        average_precision_score(y[i], s_a[i]) - average_precision_score(y[i], s_b[i])
        for i in bootstrap_indices(len(y))
    ])
    return {
        "delta_pr_auc": float(average_precision_score(y, s_a) - average_precision_score(y, s_b)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value_one_sided": float((diffs <= 0).mean()),
    }


# ── Persistence helpers ────────────────────────────────────────────────────
def upsert_log(row: dict):
    log = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
    if not log.empty:
        log = log[log["exp_id"] != row["exp_id"]]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True).sort_values("exp_id")
    log.to_csv(LOG_PATH, index=False)


def save_scores(exp_id, y_test, scores):
    df = pd.read_parquet(SCORES_PATH) if SCORES_PATH.exists() else pd.DataFrame({"y_test": y_test})
    df[exp_id] = scores.astype("float32")
    df.to_parquet(SCORES_PATH, index=False)


# ── Main loop ──────────────────────────────────────────────────────────────
def load_splits():
    df = pd.read_parquet(FEATURES_V2_PATH)
    masks = {
        "train": (df["step"] <= TRAIN_MAX_STEP).to_numpy(),
        "val": ((df["step"] > TRAIN_MAX_STEP) & (df["step"] <= VAL_MAX_STEP)).to_numpy(),
        "test": (df["step"] > VAL_MAX_STEP).to_numpy(),
    }
    y = {k: df.loc[m, "isFraud"].to_numpy() for k, m in masks.items()}
    for k in masks:
        logger.info(f"{k:<5}: {masks[k].sum():>9,} rows  fraud={int(y[k].sum()):,}  "
                    f"rate={y[k].mean() * 100:.3f}%")
    logger.info(f"Random-classifier PR-AUC baseline on test = {y['test'].mean():.4f}")
    X_cache = {
        fs: {k: df.loc[m, cols].to_numpy(dtype="float32") for k, m in masks.items()}
        for fs, cols in FEATURE_SETS.items()
    }
    return X_cache, y


def run_experiment(exp_id, fs_name, model_name, factory, X_cache, y):
    logger.info("=" * 70)
    logger.info(f"{exp_id} | {model_name} | {fs_name}")
    X = X_cache[fs_name]

    model = factory()
    t0 = time.time()
    model.fit(X["train"], y["train"], X["val"], y["val"])
    train_time = time.time() - t0

    threshold = select_threshold(y["val"], model.score(X["val"]))

    t1 = time.time()
    test_scores = model.score(X["test"])
    inference_ms = (time.time() - t1) * 1000 / len(X["test"])

    m = metrics_at(y["test"], test_scores, threshold)
    ci_low, ci_high = bootstrap_ap_ci(y["test"], test_scores)

    path = V2_MODEL_DIR / f"{exp_id}.joblib"
    joblib.dump(model, path)

    row = {
        "exp_id": exp_id, "model": model_name, "feature_set": fs_name,
        "n_features": X["train"].shape[1],
        "pr_auc": round(m["pr_auc"], 4),
        "pr_auc_ci_low": round(ci_low, 4), "pr_auc_ci_high": round(ci_high, 4),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k != "pr_auc"},
        "threshold": round(threshold, 4),
        "train_time_s": round(train_time, 1),
        "inference_ms_per_txn": round(inference_ms, 5),
        "model_size_mb": round(path.stat().st_size / 1e6, 2),
        "best_iteration": getattr(model, "best_iteration", None),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    upsert_log(row)
    save_scores(exp_id, y["test"], test_scores)
    logger.info(f"  PR-AUC {row['pr_auc']} [{row['pr_auc_ci_low']}, {row['pr_auc_ci_high']}]  "
                f"Recall {row['recall']}  F1 {row['f1']}  FP {row['fp']}  "
                f"thr {row['threshold']}  train {row['train_time_s']}s")


def run_paired_tests():
    if not SCORES_PATH.exists():
        return
    scores = pd.read_parquet(SCORES_PATH)
    y = scores["y_test"].to_numpy()
    rows = []
    for a, b, question in PAIRED_TESTS:
        if a in scores and b in scores:
            res = paired_bootstrap(y, scores[a].to_numpy(), scores[b].to_numpy())
            if res["ci_low"] > 0:
                verdict = "candidate significantly BETTER"
            elif res["ci_high"] < 0:
                verdict = "candidate significantly WORSE"
            else:
                verdict = "no significant difference"
            rows.append({"candidate": a, "reference": b, "question": question,
                         **{k: round(v, 4) for k, v in res.items()},
                         "verdict": verdict})
    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(PAIRED_PATH, index=False)
        print("\n" + "=" * 100 + "\nPAIRED BOOTSTRAP COMPARISONS (test set)\n" + "=" * 100)
        print(out.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Experiment IDs to run, e.g. V2-05 V2-06")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    V2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if args.only:
        known = {exp_id for exp_id, *_ in EXPERIMENTS}
        unknown = [e for e in args.only if e not in known]
        if unknown:
            raise SystemExit(
                f"Unknown experiment ID(s): {unknown}. "
                f"Known IDs: {sorted(known)}. "
                "Check that src/run_experiments.py is the latest version."
            )

    if not args.compare_only:
        X_cache, y = load_splits()
        for exp_id, fs, name, factory in EXPERIMENTS:
            if args.only and exp_id not in args.only:
                continue
            run_experiment(exp_id, fs, name, factory, X_cache, y)

    log = pd.read_csv(LOG_PATH)
    cols = ["exp_id", "model", "feature_set", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
            "roc_auc", "precision", "recall", "f1", "fpr", "tp", "fp", "fn",
            "train_time_s", "inference_ms_per_txn"]
    print("\n" + "=" * 140 + "\nEXPERIMENT LOG — RANKED BY TEST PR-AUC\n" + "=" * 140)
    print(log.sort_values("pr_auc", ascending=False)[cols].to_string(index=False))
    run_paired_tests()


if __name__ == "__main__":
    main()