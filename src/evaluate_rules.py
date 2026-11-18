"""
evaluate_rules.py
-----------------
Evaluates the rules engine alongside the final calibrated ensemble.

Protocol:
  1. VALIDATION: measure every candidate rule (flags, precision, recall, lift),
     assign roles with the fixed thresholds in rules.py, save
     models/final/rules_config.json.
  2. VALIDATION: check which rules flag the fraud the ML model leaves in LOW.
  3. TEST (reporting only): compare ML-only bands with rules + ML bands.

Caveat reported with the results: the account-drain signal mirrors PaySim's
fraud generator, which always empties the victim's account. Combined results
therefore show what a rules layer adds under PaySim's behavioural assumptions;
the ML-only results remain the headline.

Outputs:
  models/final/rules_config.json
  results/metrics/rules_validation.csv
  results/metrics/rules_test.csv
  results/metrics/rules_vs_ml_missed_fraud_validation.csv
  results/metrics/bands_ml_vs_combined_test.csv
  results/metrics/combined_summary_test.csv

Run:
    python -m src.evaluate_rules
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd

from .build_features_v2 import BALANCE_ABLATION_COLUMNS, FEATURES_V2_PATH, FS2_COLUMNS
from .config import METRICS_DIR, MODEL_DIR, TRAIN_MAX_STEP, VAL_MAX_STEP
from .explain import expose_calibrators_for_unpickling
from .rules import (
    BAND_ORDER, assign_roles, combine_bands, evaluate_rules, rule_bands,
    rule_statistics, save_rules_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FINAL_DIR = MODEL_DIR / "final"


def load_final_model():
    """Components, calibrator and cut-offs saved by calibrate_final.py."""
    expose_calibrators_for_unpickling()
    with open(FINAL_DIR / "risk_config.json") as f:
        config = json.load(f)
    components = [joblib.load(FINAL_DIR / fname) for fname in config["component_files"].values()]
    calibrator = joblib.load(FINAL_DIR / "calibrator.joblib")
    return config, components, calibrator


def ml_bands(X, components, calibrator, cutoffs):
    raw = np.mean([m.predict_proba(X)[:, 1] for m in components], axis=0)
    risk = calibrator.predict(raw)
    critical = cutoffs["critical"] if cutoffs["critical"] is not None else np.inf
    band = np.select([risk >= critical, risk >= cutoffs["high"], risk >= cutoffs["medium"]],
                     ["CRITICAL", "HIGH", "MEDIUM"], default="LOW")
    return risk, band.astype(object)


def band_table(band, y, label):
    total_fraud = int(y.sum())
    rows = []
    for b in BAND_ORDER:
        m = band == b
        n, tp = int(m.sum()), int(y[m].sum())
        rows.append({"system": label, "band": b, "transactions": n,
                     "share_of_transactions_pct": round(n / len(y) * 100, 2),
                     "fraud_cases": tp, "legitimate_cases": n - tp,
                     "fraud_rate_in_band_pct": round(tp / n * 100, 2) if n else 0.0,
                     "share_of_all_fraud_pct": round(tp / total_fraud * 100, 2)})
    return pd.DataFrame(rows)


def summary(band, y, label):
    friction = band != "LOW"
    held = np.isin(band, ["HIGH", "CRITICAL"])
    critical = band == "CRITICAL"
    legit = y == 0
    return {
        "system": label,
        "transactions_with_friction_pct": round(friction.mean() * 100, 2),
        "fraud_caught_medium_or_above_pct": round(y[friction].sum() / y.sum() * 100, 2),
        "fraud_caught_high_or_above_pct": round(y[held].sum() / y.sum() * 100, 2),
        "fraud_approved_in_low": int(y[~friction].sum()),
        "legitimate_customers_with_friction": int((friction & legit).sum()),
        "legitimate_customers_held_or_declined": int((held & legit).sum()),
        "legitimate_customers_declined": int((critical & legit).sum()),
    }


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["step"] + FS2_COLUMNS + BALANCE_ABLATION_COLUMNS + ["isFraud"]
    df = pd.read_parquet(FEATURES_V2_PATH, columns=cols)
    val = df[(df["step"] > TRAIN_MAX_STEP) & (df["step"] <= VAL_MAX_STEP)].reset_index(drop=True)
    test = df[df["step"] > VAL_MAX_STEP].reset_index(drop=True)
    del df
    y_val, y_test = val["isFraud"].to_numpy(), test["isFraud"].to_numpy()
    logger.info(f"Validation: {len(val):,} rows, fraud={int(y_val.sum()):,} | "
                f"Test: {len(test):,} rows, fraud={int(y_test.sum()):,}")

    config, components, calibrator = load_final_model()
    cutoffs = config["cutoffs"]
    logger.info(f"Final model: {config['model']}")

    # 1. Rule roles from validation
    val_flags = evaluate_rules(val)
    val_stats = rule_statistics(val_flags, y_val)
    roles = assign_roles(val_stats)
    val_stats["assigned_role"] = val_stats["rule"].map(roles)
    val_stats.to_csv(METRICS_DIR / "rules_validation.csv", index=False)
    save_rules_config(FINAL_DIR / "rules_config.json", roles, val_stats)

    # 2. Do the rules reach the fraud the model leaves in LOW? (validation)
    _, val_ml_band = ml_bands(val[FS2_COLUMNS].to_numpy(dtype="float32"), components, calibrator, cutoffs)
    missed = (val_ml_band == "LOW") & (y_val == 1)
    overlap = pd.DataFrame([{
        "rule": r, "assigned_role": roles[r],
        "ml_missed_fraud_flagged": int(val_flags[r].to_numpy()[missed].sum()),
        "share_of_ml_missed_fraud_pct": round(val_flags[r].to_numpy()[missed].mean() * 100, 2) if missed.sum() else 0.0,
    } for r in val_flags.columns])
    overlap.to_csv(METRICS_DIR / "rules_vs_ml_missed_fraud_validation.csv", index=False)

    # 3. Test: ML only vs rules + ML
    test_flags = evaluate_rules(test)
    test_stats = rule_statistics(test_flags, y_test)
    test_stats["assigned_role"] = test_stats["rule"].map(roles)
    test_stats.to_csv(METRICS_DIR / "rules_test.csv", index=False)

    _, ml_band = ml_bands(test[FS2_COLUMNS].to_numpy(dtype="float32"), components, calibrator, cutoffs)
    forced_band, triggered = rule_bands(test_flags, roles)
    combined = combine_bands(ml_band, forced_band)
    raised = np.array([BAND_ORDER.index(c) > BAND_ORDER.index(m) for c, m in zip(combined, ml_band)])

    bands = pd.concat([band_table(ml_band, y_test, "ML only"),
                       band_table(combined, y_test, "Rules + ML")], ignore_index=True)
    bands.to_csv(METRICS_DIR / "bands_ml_vs_combined_test.csv", index=False)
    summ = pd.DataFrame([summary(ml_band, y_test, "ML only"), summary(combined, y_test, "Rules + ML")])
    summ.to_csv(METRICS_DIR / "combined_summary_test.csv", index=False)

    # 4. Report
    pd.set_option("display.width", 200)
    show = ["rule", "flags", "flag_rate_pct", "fraud_flagged", "precision", "recall", "lift_over_base_rate"]
    print("\n" + "=" * 110)
    print(f"RULES ON VALIDATION (base fraud rate {y_val.mean() * 100:.2f}%) — roles assigned here")
    print("=" * 110)
    print(val_stats[show + ["assigned_role"]].to_string(index=False))
    print("\n" + "=" * 110)
    print(f"DO THE RULES REACH WHAT THE MODEL MISSES? ({int(missed.sum())} validation frauds left in LOW by ML)")
    print("=" * 110)
    print(overlap.to_string(index=False))
    print("\n" + "=" * 110)
    print(f"RULES ON TEST (reporting only, base fraud rate {y_test.mean() * 100:.2f}%)")
    print("=" * 110)
    print(test_stats[show + ["assigned_role"]].to_string(index=False))
    print("\n" + "=" * 110)
    print("RISK BANDS ON TEST — ML ONLY vs RULES + ML")
    print("=" * 110)
    print(bands.to_string(index=False))
    print("\n" + "=" * 110)
    print("SUMMARY ON TEST")
    print("=" * 110)
    print(summ.set_index("system").T.to_string())
    print(f"\nTransactions whose band was raised by a rule: {int(raised.sum()):,} "
          f"({int(y_test[raised].sum()):,} fraud, {int((raised & (y_test == 0)).sum()):,} legitimate)")
    print("-" * 110)
    print("CAVEAT: the account-drain signal mirrors PaySim's fraud generator, which always empties the\n"
          "victim's account. Rules + ML results show what a rules layer adds under PaySim's behavioural\n"
          "assumptions; the ML-only results remain the headline.")
    print("=" * 110)
    logger.info(f"Rule roles saved -> {FINAL_DIR / 'rules_config.json'}")


if __name__ == "__main__":
    main()
