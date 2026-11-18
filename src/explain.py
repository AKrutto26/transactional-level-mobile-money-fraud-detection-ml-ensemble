"""
explain.py
----------
SHAP explainability for the final deployed model (models/final/).

The final model averages the fraud probabilities of its components
(Random Forest and XGBoost for soft voting). Each component is explained
with SHAP TreeExplainer in PROBABILITY space (interventional, against a fixed
background sample from the training period). Because the ensemble output is
the mean of the component probabilities, the mean of the component SHAP
values is an exact additive explanation of the ensemble output. The script
verifies this numerically (additivity check).

Explanations cover the ensemble probability BEFORE calibration. Calibration is
monotonic, so it rescales the score without changing the ranking of reasons.

Outputs:
  results/shap/global_importance.csv          mean |SHAP| per feature (ensemble + components)
  results/shap/grouped_importance.csv         mean |SHAP| per feature group (related features summed)
  results/shap/transfer_effect.csv            how the TRANSFER effect varies with context
  results/shap/local_examples.json            reasons for the three example cases
  results/figures/shap_global_importance.png  ranked bar chart
  results/figures/shap_beeswarm.png           direction and spread of each feature's effect
  results/figures/shap_grouped_importance.png ranked bar chart of feature groups
  results/figures/shap_dependence_transfer.png TRANSFER effect against transaction amount
  results/figures/shap_waterfall_<case>.png   one waterfall plot per example case
  models/final/shap_background.npy            background sample reused by the API

Example cases (test set, reporting only):
  best_catch    -> true fraud with the highest risk score
  false_alarm   -> legitimate transaction with the highest risk score
  missed_fraud  -> fraud with the lowest risk score (a miss in the LOW band)

Run:
    python -m src.explain
    python -m src.explain --n-global 1000 --n-background 100   # faster

Changelog:
  v1.1 — XGBoost 3.x enables a categorical-data flag by default, which SHAP's
         interventional explainer rejects. The flag is switched off after
         loading (all features are numeric, so predictions are unchanged).
  v1.2 — (a) the full background sample is now used (SHAP previously
         subsampled it to 100 rows); (b) related features are grouped and
         their SHAP values summed, because correlated features split credit
         (e.g. four features all encode "new destination"); grouped reasons in
         plain language are returned to the API; (c) the TRANSFER effect is
         broken down by context to explain its counter-intuitive direction.
"""

import argparse
import json
import logging
import time

import __main__
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import calibrate_final as cf
from .config import FIGURES_DIR, MODEL_DIR, RANDOM_STATE, SHAP_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FINAL_DIR = MODEL_DIR / "final"
BACKGROUND_PATH = FINAL_DIR / "shap_background.npy"

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "log_amount": "Transaction amount (log scale)",
    "hour_of_day": "Hour of day",
    "type_TRANSFER": "Transaction type is TRANSFER",
    "dest_txn_count_before": "Destination's previous transactions",
    "dest_total_amount_before": "Destination's total amount received",
    "dest_avg_amount_before": "Destination's average amount received",
    "dest_time_since_last_step": "Hours since destination's last transaction",
    "dest_is_new": "New destination account",
    "dest_amount_to_avg_ratio": "Amount vs destination's usual amount",
}


FEATURE_GROUPS = {
    "Transaction amount": ["amount", "log_amount"],
    "Destination account history": ["dest_txn_count_before", "dest_total_amount_before",
                                    "dest_avg_amount_before", "dest_time_since_last_step",
                                    "dest_is_new", "dest_amount_to_avg_ratio"],
    "Transaction type": ["type_TRANSFER"],
    "Time of day": ["hour_of_day"],
}


def describe_group(group: str, x: dict) -> str:
    """Plain-language description of a feature group's values for one transaction."""
    if group == "Transaction amount":
        return f"Amount of {x['amount']:,.0f}"
    if group == "Destination account history":
        if x["dest_is_new"] >= 0.5:
            return "New destination account with no transaction history"
        ratio = x["dest_amount_to_avg_ratio"]
        ratio_text = f"{ratio:.1f}x" if ratio >= 0.1 else "under 0.1x"
        return (f"Destination has {int(x['dest_txn_count_before'])} earlier transactions, last one "
                f"{int(x['dest_time_since_last_step'])} h earlier; amount is {ratio_text} its usual")
    if group == "Transaction type":
        return "TRANSFER" if x["type_TRANSFER"] >= 0.5 else "CASH_OUT"
    if group == "Time of day":
        return f"Made at {int(x['hour_of_day']):02d}:00"
    return group


def expose_calibrators_for_unpickling():
    """The calibrator was pickled while running `python -m src.calibrate_final`,
    so its class was recorded under `__main__`; register it so it loads."""
    for name in ("PlattCalibrator", "IsotonicCalibrator"):
        setattr(__main__, name, getattr(cf, name))


def _positive_class(values, expected):
    """Normalise SHAP output to the positive (fraud) class."""
    values = np.asarray(values)
    if values.ndim == 3:                       # (n, features, classes)
        values = values[:, :, 1]
    expected = np.atleast_1d(np.asarray(expected, dtype=float))
    expected = float(expected[1] if expected.size == 2 else expected[0])
    return values, expected


def _disable_xgb_categorical_flag(model) -> None:
    """
    XGBoost 3.x sets enable_categorical=True by default on its scikit-learn
    wrapper. SHAP's interventional TreeExplainer refuses any XGBoost model with
    that flag, even when no categorical feature exists. Our features are all
    numeric (feature_types is None), so switching the flag off changes nothing
    about the trees or the predictions; it only unblocks SHAP.
    """
    params = model.get_params() if hasattr(model, "get_params") else {}
    if params.get("enable_categorical"):
        booster_types = model.get_booster().feature_types
        if booster_types is not None and "c" in booster_types:
            raise ValueError("Model contains categorical features; interventional SHAP unsupported.")
        model.set_params(enable_categorical=False)


class FraudExplainer:
    """Loads the deployed artefacts and explains transactions. Reused by the API."""

    def __init__(self, background: np.ndarray | None = None):
        expose_calibrators_for_unpickling()
        with open(FINAL_DIR / "risk_config.json") as f:
            self.config = json.load(f)
        self.features = self.config["feature_columns"]
        self.cutoffs = self.config["cutoffs"]
        self.components = {name: joblib.load(FINAL_DIR / fname)
                           for name, fname in self.config["component_files"].items()}
        for model in self.components.values():
            _disable_xgb_categorical_flag(model)
        self.calibrator = joblib.load(FINAL_DIR / "calibrator.joblib")
        if background is None:
            background = np.load(BACKGROUND_PATH)
        self.background = background.astype("float32")
        # Independent masker sized to the whole background, so no silent subsampling
        masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
        self.explainers = {
            name: shap.TreeExplainer(model, data=masker,
                                     feature_perturbation="interventional",
                                     model_output="probability")
            for name, model in self.components.items()
        }
        self.group_index = {g: [self.features.index(f) for f in fs if f in self.features]
                            for g, fs in FEATURE_GROUPS.items()}

    def raw_probability(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(X)[:, 1] for m in self.components.values()], axis=0)

    def risk(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(self.raw_probability(X))

    def band(self, risk: np.ndarray) -> np.ndarray:
        critical = self.cutoffs["critical"] if self.cutoffs["critical"] is not None else np.inf
        return np.select([risk >= critical, risk >= self.cutoffs["high"], risk >= self.cutoffs["medium"]],
                         ["CRITICAL", "HIGH", "MEDIUM"], default="LOW")

    def shap_values(self, X: np.ndarray):
        """Returns (ensemble values, ensemble base value, per-component values)."""
        per_component, bases = {}, []
        for name, explainer in self.explainers.items():
            vals, base = _positive_class(explainer.shap_values(X, check_additivity=False),
                                         explainer.expected_value)
            per_component[name] = vals
            bases.append(base)
        values = np.mean(list(per_component.values()), axis=0)
        return values, float(np.mean(bases)), per_component

    def top_reasons(self, shap_row: np.ndarray, x_row: np.ndarray, k: int = 3) -> list[dict]:
        order = np.argsort(-np.abs(shap_row))[:k]
        return [{
            "feature": self.features[i],
            "label": FEATURE_LABELS.get(self.features[i], self.features[i]),
            "value": float(x_row[i]),
            "impact_pct_points": round(float(shap_row[i]) * 100, 2),
            "direction": "increases risk" if shap_row[i] > 0 else "decreases risk",
        } for i in order]

    def group_values(self, values: np.ndarray) -> pd.DataFrame:
        """Sum SHAP values within each feature group (valid because SHAP is additive)."""
        return pd.DataFrame({g: values[:, idx].sum(axis=1) for g, idx in self.group_index.items()})

    def grouped_reasons(self, shap_row: np.ndarray, x_row: np.ndarray, k: int = 3) -> list[dict]:
        x = dict(zip(self.features, map(float, x_row)))
        g = self.group_values(shap_row.reshape(1, -1)).iloc[0]
        top = g.reindex(g.abs().sort_values(ascending=False).index)[:k]
        return [{
            "reason": group,
            "description": describe_group(group, x),
            "impact_pct_points": round(float(v) * 100, 2),
            "direction": "increases risk" if v > 0 else "decreases risk",
        } for group, v in top.items()]

    def explain_one(self, x_row: np.ndarray, k: int = 3) -> dict:
        X = x_row.reshape(1, -1).astype("float32")
        values, base, _ = self.shap_values(X)
        risk = float(self.risk(X)[0])
        band = str(self.band(np.array([risk]))[0])
        return {
            "risk_score": round(risk, 4),
            "risk_band": band,
            "recommended_action": self.config["band_actions"][band],
            "top_reasons": self.grouped_reasons(values[0], X[0], k),
            "feature_contributions": self.top_reasons(values[0], X[0], k=len(self.features)),
        }


def waterfall(values_row, base, x_row, features, title, path):
    exp = shap.Explanation(values=values_row, base_values=base, data=x_row,
                           feature_names=[FEATURE_LABELS.get(f, f) for f in features])
    plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.title(title, fontsize=10)
    plt.gcf().set_size_inches(9, 5.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-global", type=int, default=2000, help="test rows for global SHAP")
    parser.add_argument("--n-background", type=int, default=200, help="training rows as SHAP background")
    args = parser.parse_args()

    from .run_experiments import load_splits      # heavy import only when run as a script
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    X_cache, y = load_splits()
    X = X_cache["FS2_clean"]

    background = X["train"][rng.choice(len(X["train"]), args.n_background, replace=False)]
    np.save(BACKGROUND_PATH, background)
    explainer = FraudExplainer(background)
    features = explainer.features
    logger.info(f"Final model: {explainer.config['model']}")
    logger.info(f"Background: {args.n_background} training rows (saved for the API)")

    # ── Global explanations on a representative test sample ─────────────
    idx = rng.choice(len(X["test"]), min(args.n_global, len(X["test"])), replace=False)
    X_g = X["test"][idx]
    t0 = time.time()
    values, base, per_comp = explainer.shap_values(X_g)
    elapsed = time.time() - t0
    raw = explainer.raw_probability(X_g)
    additivity_err = float(np.max(np.abs(values.sum(axis=1) + base - raw)))
    logger.info(f"SHAP for {len(X_g):,} transactions in {elapsed:.1f}s "
                f"({elapsed / len(X_g) * 1000:.1f} ms per transaction)")
    logger.info(f"Additivity check: max |sum(SHAP) + base - model output| = {additivity_err:.2e}")

    imp = pd.DataFrame({
        "feature": features,
        "label": [FEATURE_LABELS.get(f, f) for f in features],
        "ensemble_mean_abs_shap": np.abs(values).mean(axis=0),
        **{f"{name}_mean_abs_shap": np.abs(v).mean(axis=0) for name, v in per_comp.items()},
    }).sort_values("ensemble_mean_abs_shap", ascending=False)
    imp["ensemble_rank"] = range(1, len(imp) + 1)
    for name in per_comp:
        imp[f"{name}_rank"] = imp[f"{name}_mean_abs_shap"].rank(ascending=False).astype(int)
    imp.round(6).to_csv(SHAP_DIR / "global_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = imp.iloc[::-1]
    ax.barh(ordered["label"], ordered["ensemble_mean_abs_shap"] * 100, color="#2E6F9E")
    ax.set_xlabel("Mean |SHAP value| (percentage points of fraud probability)")
    ax.set_title("Global feature importance, final ensemble (test sample)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_global_importance.png", dpi=200)
    plt.close(fig)

    plt.figure()
    shap.summary_plot(values, pd.DataFrame(X_g, columns=[FEATURE_LABELS.get(f, f) for f in features]),
                      show=False, max_display=len(features))
    plt.title("SHAP beeswarm, final ensemble (test sample)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    # ── Grouped importance (related features summed) ────────────────────
    grouped = explainer.group_values(values)
    g_imp = grouped.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_frame()
    g_imp["features"] = [", ".join(FEATURE_GROUPS[g]) for g in g_imp.index]
    g_imp.index.name = "group"
    g_imp.round(6).to_csv(SHAP_DIR / "grouped_importance.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ordered_g = g_imp.iloc[::-1]
    ax.barh(ordered_g.index, ordered_g["mean_abs_shap"] * 100, color="#2E6F9E")
    ax.set_xlabel("Mean |SHAP value| of the group (percentage points of fraud probability)")
    ax.set_title("Grouped feature importance, final ensemble (test sample)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_grouped_importance.png", dpi=200)
    plt.close(fig)

    # ── Why does TRANSFER mostly lower risk? Break its effect down by context ──
    f = {name: i for i, name in enumerate(features)}
    is_tr = X_g[:, f["type_TRANSFER"]] >= 0.5
    tr_shap = values[:, f["type_TRANSFER"]]
    amount_hi = X_g[:, f["amount"]] >= np.median(X_g[:, f["amount"]])
    dest_new = X_g[:, f["dest_is_new"]] >= 0.5
    rows = []
    for label, mask in (("TRANSFER, all", is_tr),
                        ("TRANSFER, amount above median", is_tr & amount_hi),
                        ("TRANSFER, amount below median", is_tr & ~amount_hi),
                        ("TRANSFER, new destination", is_tr & dest_new),
                        ("TRANSFER, established destination", is_tr & ~dest_new),
                        ("CASH_OUT, all", ~is_tr)):
        if mask.sum():
            rows.append({"context": label, "transactions": int(mask.sum()),
                         "mean_shap_pct_points": round(float(tr_shap[mask].mean()) * 100, 3),
                         "share_lowering_risk_pct": round(float((tr_shap[mask] < 0).mean()) * 100, 1)})
    transfer_df = pd.DataFrame(rows)
    transfer_df.to_csv(SHAP_DIR / "transfer_effect.csv", index=False)
    plt.figure()
    shap.dependence_plot("Transaction type is TRANSFER", values,
                         pd.DataFrame(X_g, columns=[FEATURE_LABELS.get(c, c) for c in features]),
                         interaction_index="Transaction amount (log scale)", show=False)
    plt.title("TRANSFER effect coloured by transaction amount (log scale)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_dependence_transfer.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    # ── Local explanations for three example cases ─────────────────────
    risk_all = explainer.risk(X["test"])
    raw_all = explainer.raw_probability(X["test"])
    bands_all = explainer.band(risk_all)
    fraud_idx = np.where(y["test"] == 1)[0]
    legit_idx = np.where(y["test"] == 0)[0]
    cases = {
        "best_catch": fraud_idx[np.argmax(raw_all[fraud_idx])],
        "false_alarm": legit_idx[np.argmax(raw_all[legit_idx])],
        "missed_fraud": fraud_idx[np.argmin(raw_all[fraud_idx])],
    }
    local = {}
    for case, i in cases.items():
        row = X["test"][i:i + 1]
        vals, b, _ = explainer.shap_values(row)
        truth = "fraud" if y["test"][i] == 1 else "legitimate"
        info = {
            "test_row_index": int(i),
            "actual_label": truth,
            "raw_ensemble_probability": round(float(raw_all[i]), 4),
            "calibrated_risk": round(float(risk_all[i]), 4),
            "risk_band": str(bands_all[i]),
            "recommended_action": explainer.config["band_actions"][str(bands_all[i])],
            "top_reasons": explainer.grouped_reasons(vals[0], row[0], k=4),
            "feature_contributions": explainer.top_reasons(vals[0], row[0], k=5),
        }
        local[case] = info
        title = (f"{case.replace('_', ' ').title()}: actual {truth}, calibrated risk "
                 f"{info['calibrated_risk']:.3f} ({info['risk_band']})")
        waterfall(vals[0], b, row[0], features, title, FIGURES_DIR / f"shap_waterfall_{case}.png")
    with open(SHAP_DIR / "local_examples.json", "w") as f:
        json.dump(local, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    comp_cols = [c for c in imp.columns if c.endswith("_rank") and c != "ensemble_rank"]
    print("\n" + "=" * 100)
    print("GLOBAL FEATURE IMPORTANCE — final ensemble (mean |SHAP|, percentage points of probability)")
    print("=" * 100)
    show = imp[["ensemble_rank", "label", "ensemble_mean_abs_shap"] + comp_cols].copy()
    show["ensemble_mean_abs_shap"] = (show["ensemble_mean_abs_shap"] * 100).round(2)
    print(show.to_string(index=False))
    print("\n" + "=" * 100)
    print("GROUPED IMPORTANCE — related features summed (percentage points of probability)")
    print("=" * 100)
    print((g_imp["mean_abs_shap"] * 100).round(2).to_string())
    print("\n" + "=" * 100)
    print("TRANSFER EFFECT BY CONTEXT (negative = TRANSFER lowers the fraud score)")
    print("=" * 100)
    print(transfer_df.to_string(index=False))
    print(f"\nAdditivity check (should be ~0): {additivity_err:.2e}")
    print(f"SHAP cost: {elapsed / len(X_g) * 1000:.1f} ms per transaction "
          f"(background {args.n_background} rows)")
    for case, info in local.items():
        print("\n" + "-" * 100)
        print(f"{case.upper()}: actual {info['actual_label']} | calibrated risk {info['calibrated_risk']} "
              f"| band {info['risk_band']} -> {info['recommended_action']}")
        for r in info["top_reasons"]:
            print(f"   {r['impact_pct_points']:+7.2f} pts  {r['reason']}: {r['description']}  ({r['direction']})")
    print("=" * 100)
    logger.info(f"Figures -> {FIGURES_DIR}   Tables -> {SHAP_DIR}")


if __name__ == "__main__":
    main()
