"""
calibrate_final.py
------------------
Turns the selected model (V2-10: XGBoost, square-root class weighting,
FS2_clean features) into a deployable, calibrated risk scorer.

Protocol:
  1. Validation is split in time:
       calibration half : steps 521-575  -> fit the calibrator
       tuning half      : steps 576-631  -> choose risk-band cut-offs
  2. Two calibrators are fitted (Platt and isotonic); the one with the lower
     Brier score on the tuning half is kept (rule fixed before running).
  3. Risk-band cut-offs on the tuning half (calibrated scores):
       MEDIUM   : highest cut-off that still catches >= 70% of fraud
       HIGH     : cut-off with the best F1 (the alert operating point)
       CRITICAL : lowest cut-off where >= 95% of flagged cases are fraud
                  (falls back to 90%; if neither is reachable, CRITICAL is
                  reserved for critical rule violations only)
  4. The test set is used for reporting only.

Outputs:
  models/final/final_xgboost.joblib   raw XGBClassifier for the API
  models/final/calibrator.joblib      fitted calibrator
  models/final/risk_config.json       features, cut-offs, band actions, metadata
  results/metrics/calibration_metrics.csv
  results/metrics/risk_bands_test.csv
  results/figures/calibration_curve.png

Run:
    python -m src.calibrate_final                        # XGBoost alone (V2-10), default
    python -m src.calibrate_final --source soft_voting   # RF + XGBoost average (V2-11)

Changelog:
  v1.1 — added --source soft_voting: calibrates the average of the V2-04
         Random Forest and V2-10 XGBoost probabilities and saves both
         component models. Default behaviour (XGBoost alone) is unchanged.
"""

import argparse
import json
import logging
from datetime import datetime

import __main__
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score,
)

from . import run_experiments as rx
from .build_features_v2 import FEATURES_V2_PATH, FS2_COLUMNS
from .config import FIGURES_DIR, METRICS_DIR, MODEL_DIR, TRAIN_MAX_STEP, VAL_MAX_STEP
from .run_experiments import V2_MODEL_DIR, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FINAL_DIR = MODEL_DIR / "final"
SOURCES = {
    "xgboost": {
        "experiment": "V2-10",
        "description": "XGBoost, square-root class weighting",
        "components": {"xgboost": "V2-10"},
    },
    "soft_voting": {
        "experiment": "V2-11",
        "description": "Soft voting: mean of Random Forest (V2-04) and XGBoost (V2-10) probabilities",
        "components": {"random_forest": "V2-04", "xgboost": "V2-10"},
    },
}
CALIB_SPLIT_STEP = 575           # validation steps 521-575 calibrate, 576-631 tune
MEDIUM_RECALL_TARGET = 0.70
CRITICAL_PRECISION_TARGETS = (0.95, 0.90)
EPS = 1e-6

BAND_ACTIONS = {
    "LOW": "APPROVE",
    "MEDIUM": "STEP_UP_VERIFICATION",
    "HIGH": "HOLD_AND_ALERT",
    "CRITICAL": "DECLINE_AND_ESCALATE",
}


# ── Calibrators ────────────────────────────────────────────────────────────
class PlattCalibrator:
    """Logistic regression on the log-odds of the raw score."""
    name = "platt"

    def fit(self, raw, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.lr.fit(self._logit(raw), y)
        return self

    def predict(self, raw):
        return self.lr.predict_proba(self._logit(raw))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)


class IsotonicCalibrator:
    name = "isotonic"

    def fit(self, raw, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(raw, y)
        return self

    def predict(self, raw):
        return self.iso.predict(raw)


# ── Helpers ────────────────────────────────────────────────────────────────
def expose_wrappers_for_unpickling():
    """V2 models were pickled under `__main__`; register the classes so they load."""
    for name in ("SklearnClassifier", "IsoForestScorer", "XGBScorer", "IFXGBHybrid"):
        setattr(__main__, name, getattr(rx, name))


def cutoff_for_recall(y, p, target):
    precision, recall, thr = precision_recall_curve(y, p)
    ok = np.where(recall[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def cutoff_max_f1(y, p):
    precision, recall, thr = precision_recall_curve(y, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
    return float(thr[int(np.argmax(f1[:-1]))])


def cutoff_for_precision(y, p, target):
    precision, recall, thr = precision_recall_curve(y, p)
    ok = np.where(precision[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else None


def assign_bands(p, cut):
    critical = cut["critical"] if cut["critical"] is not None else np.inf
    return np.select([p >= critical, p >= cut["high"], p >= cut["medium"]],
                     ["CRITICAL", "HIGH", "MEDIUM"], default="LOW")


def calib_metrics(y, p):
    return {
        "brier": round(brier_score_loss(y, p), 5),
        "log_loss": round(log_loss(y, np.clip(p, EPS, 1 - EPS)), 5),
        "mean_predicted": round(float(p.mean()), 5),
        "observed_rate": round(float(y.mean()), 5),
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), default="xgboost")
    args = parser.parse_args()
    source = SOURCES[args.source]
    logger.info(f"Calibrating source: {args.source} ({source['experiment']})")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    expose_wrappers_for_unpickling()

    X_cache, y = load_splits()
    X = X_cache["FS2_clean"]

    steps = pd.read_parquet(FEATURES_V2_PATH, columns=["step"])["step"].to_numpy()
    val_steps = steps[(steps > TRAIN_MAX_STEP) & (steps <= VAL_MAX_STEP)]
    cal_mask = val_steps <= CALIB_SPLIT_STEP
    tune_mask = ~cal_mask

    # Load raw estimators (plain XGBClassifier / RandomForestClassifier)
    components = {}
    for name, exp_id in source["components"].items():
        wrapper = joblib.load(V2_MODEL_DIR / f"{exp_id}.joblib")
        components[name] = wrapper.est

    def raw_score(X_):
        return np.mean([m.predict_proba(X_)[:, 1] for m in components.values()], axis=0)

    raw_val = raw_score(X["val"])
    raw_test = raw_score(X["test"])
    y_cal, y_tune = y["val"][cal_mask], y["val"][tune_mask]
    logger.info(f"Calibration half: {cal_mask.sum():,} rows, fraud={int(y_cal.sum()):,}")
    logger.info(f"Tuning half:      {tune_mask.sum():,} rows, fraud={int(y_tune.sum()):,}")

    # 1. Fit both calibrators, keep the lower Brier on the tuning half
    candidates = [PlattCalibrator().fit(raw_val[cal_mask], y_cal),
                  IsotonicCalibrator().fit(raw_val[cal_mask], y_cal)]
    rows = [{"method": "uncalibrated", "split": "tuning",
             **calib_metrics(y_tune, raw_val[tune_mask])},
            {"method": "uncalibrated", "split": "test", **calib_metrics(y["test"], raw_test)}]
    for c in candidates:
        rows.append({"method": c.name, "split": "tuning",
                     **calib_metrics(y_tune, c.predict(raw_val[tune_mask]))})
        rows.append({"method": c.name, "split": "test",
                     **calib_metrics(y["test"], c.predict(raw_test))})
    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(METRICS_DIR / "calibration_metrics.csv", index=False)

    tuning_brier = {c.name: brier_score_loss(y_tune, c.predict(raw_val[tune_mask])) for c in candidates}
    calibrator = min(candidates, key=lambda c: tuning_brier[c.name])
    logger.info(f"Selected calibrator: {calibrator.name} (tuning Brier {tuning_brier[calibrator.name]:.5f})")

    p_tune = calibrator.predict(raw_val[tune_mask])
    p_test = calibrator.predict(raw_test)

    # 2. Risk-band cut-offs on the tuning half
    cut = {"medium": cutoff_for_recall(y_tune, p_tune, MEDIUM_RECALL_TARGET),
           "high": cutoff_max_f1(y_tune, p_tune), "critical": None}
    critical_target_used = None
    for target in CRITICAL_PRECISION_TARGETS:
        c_thr = cutoff_for_precision(y_tune, p_tune, target)
        if c_thr is not None and c_thr > cut["high"]:
            cut["critical"], critical_target_used = c_thr, target
            break
    if cut["medium"] >= cut["high"]:
        logger.warning("MEDIUM cut-off was not below HIGH; MEDIUM set to half of HIGH.")
        cut["medium"] = cut["high"] / 2

    # 3. Test-set report (reporting only)
    bands = assign_bands(p_test, cut)
    total_fraud = int(y["test"].sum())
    band_rows = []
    for band in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        mask = bands == band
        n, fraud = int(mask.sum()), int(y["test"][mask].sum())
        band_rows.append({
            "band": band, "action": BAND_ACTIONS[band], "transactions": n,
            "share_of_transactions_pct": round(n / len(bands) * 100, 2),
            "fraud_cases": fraud,
            "fraud_rate_in_band_pct": round(fraud / n * 100, 2) if n else 0.0,
            "share_of_all_fraud_pct": round(fraud / total_fraud * 100, 2),
        })
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(METRICS_DIR / "risk_bands_test.csv", index=False)

    alert = (p_test >= cut["high"]).astype(int)
    alert_metrics = {
        "pr_auc": round(average_precision_score(y["test"], p_test), 4),
        "precision": round(precision_score(y["test"], alert, zero_division=0), 4),
        "recall": round(recall_score(y["test"], alert, zero_division=0), 4),
        "f1": round(f1_score(y["test"], alert, zero_division=0), 4),
    }

    # 4. Reliability plot (test set)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    for label, probs in (("Uncalibrated", raw_test), (f"Calibrated ({calibrator.name})", p_test)):
        frac, mean_pred = calibration_curve(y["test"], probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=label)
    ax.set_xlabel("Mean predicted fraud probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(f"Calibration curve, final model: {args.source} (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=200)
    plt.close(fig)

    # 5. Deployment artefacts
    for old in FINAL_DIR.glob("final_*.joblib"):
        old.unlink()                         # never leave a stale component behind
    component_files = {}
    for name, est in components.items():
        fname = f"final_{name}.joblib"
        joblib.dump(est, FINAL_DIR / fname)
        component_files[name] = fname
    joblib.dump(calibrator, FINAL_DIR / "calibrator.joblib")
    config = {
        "source": args.source,
        "source_experiment": source["experiment"],
        "model": source["description"],
        "component_files": component_files,
        "combination": "mean of component probabilities" if len(components) > 1 else "single model",
        "feature_set": "FS2_clean",
        "feature_columns": FS2_COLUMNS,
        "modelled_types": ["TRANSFER", "CASH_OUT"],
        "calibrator": calibrator.name,
        "cutoffs": cut,
        "critical_precision_target": critical_target_used,
        "medium_recall_target": MEDIUM_RECALL_TARGET,
        "band_actions": BAND_ACTIONS,
        "test_alert_metrics_at_high_cutoff": alert_metrics,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(FINAL_DIR / "risk_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 6. Summary
    print("\n" + "=" * 100)
    print("CALIBRATION (lower Brier and log-loss are better)")
    print("=" * 100)
    print(cal_df.to_string(index=False))
    print(f"\nSelected calibrator: {calibrator.name}")
    print("\n" + "=" * 100)
    print("RISK-BAND CUT-OFFS (chosen on the tuning half of validation)")
    print("=" * 100)
    print(f"MEDIUM   >= {cut['medium']:.4f}   (catches >= {MEDIUM_RECALL_TARGET:.0%} of fraud on tuning data)")
    print(f"HIGH     >= {cut['high']:.4f}   (best F1 on tuning data)")
    if cut["critical"] is not None:
        print(f"CRITICAL >= {cut['critical']:.4f}   (>= {critical_target_used:.0%} precision on tuning data)")
    else:
        print("CRITICAL : no score cut-off reached 90% precision; CRITICAL is reserved for critical rule violations")
    print("\n" + "=" * 100)
    print("RISK BANDS ON THE TEST SET (reporting only)")
    print("=" * 100)
    print(band_df.to_string(index=False))
    print(f"\nAlert operating point (HIGH and above) on test: {alert_metrics}")
    t = cal_df[(cal_df["method"] == calibrator.name) & (cal_df["split"] == "test")].iloc[0]
    print(f"\nTest calibration check: mean predicted {t['mean_predicted']:.4f} vs observed fraud rate "
          f"{t['observed_rate']:.4f}")
    print("A gap here reflects the higher fraud prevalence in the later test period.")
    print("=" * 100)
    logger.info(f"Artefacts saved to {FINAL_DIR}")


if __name__ == "__main__":
    main()
