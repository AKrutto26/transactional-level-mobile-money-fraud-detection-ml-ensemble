"""
ch5_result_figures.py
---------------------
Thesis-ready figures for Chapter 5 built from the CSVs your pipeline already wrote to results/.
It does not retrain or rescore anything. Column names are matched loosely, and any figure whose
input is missing is skipped with a message, so the script never stops half-way.

Writes to FIGURES_DIR:
  ch5_model_comparison.png    test PR-AUC with 95% CI per configuration (Figure 5.15)
  ch5_recall_vs_fpr.png       recall against false positive rate (Figure 5.16)
  ch5_inference_cost.png      per-transaction inference time (Figure 5.17)
  ch5_seeds_ensemble.png      soft voting vs XGBoost across seeds (Figure 5.22)
  ch5_seeds_hybrid.png        IF hybrid vs XGBoost across seeds (Figure 5.21)
  ch5_risk_bands.png          share of transactions and fraud rate per band (Figure 5.23)
  ch5_latency.png             latency benchmark (Figure 5.61)

Run from the project root:
    python -m src.ch5_result_figures
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import FIGURES_DIR, METRICS_DIR  # noqa: E402

SELECTED = ("soft", "vot", "ensemble")
BLUE, GREY, RED, ORANGE, GREEN = "#2E5E9E", "#A7AFBA", "#C0392B", "#D9822B", "#2E8B57"
BAND_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
BAND_COL = {"LOW": "#3C8D5A", "MEDIUM": "#D4A72C", "HIGH": "#D9822B", "CRITICAL": "#C0392B"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
    "axes.titleweight": "bold", "savefig.dpi": 300, "savefig.bbox": "tight",
})


def find(name: str) -> Path | None:
    for d in (Path(METRICS_DIR), Path(FIGURES_DIR), ROOT / "results"):
        if (d / name).exists():
            return d / name
    hits = sorted((ROOT / "results").rglob(name)) if (ROOT / "results").exists() else []
    return hits[0] if hits else None


def load(name: str) -> pd.DataFrame | None:
    p = find(name)
    if p is None:
        print(f"  skip: {name} not found")
        return None
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def pick(df, *cands):
    cols = {norm(c): c for c in df.columns}
    for c in cands:
        if norm(c) in cols:
            return cols[norm(c)]
    for c in cands:
        for k, v in cols.items():
            if norm(c) in k:
                return v
    return None


def selected(label) -> bool:
    return any(h in str(label).lower() for h in SELECTED)


def save(fig, name):
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(FIGURES_DIR) / name)
    plt.close(fig)
    print(f"  saved {name}")


def model_figures():
    log = load("experiment_log.csv")
    if log is None:
        return
    m, metric = pick(log, "model", "model_name", "name"), pick(log, "pr_auc", "prauc", "average_precision")
    if not (m and metric):
        print(f"  skip model figures: no model/PR-AUC columns in {list(log.columns)}")
        return
    fs = pick(log, "feature_set")
    if fs:  # the balance ablation is reported in the text, not on the comparison charts
        log = log[~log[fs].astype(str).str.contains("ablation", case=False)].reset_index(drop=True)
    d = pd.DataFrame({"label": log[m].astype(str),
                      "pr": pd.to_numeric(log[metric], errors="coerce")})
    for k, c in {"lo": ("pr_auc_ci_low", "ci_low"), "hi": ("pr_auc_ci_high", "ci_high"), "rec": ("recall",),
                 "fpr": ("fpr", "false_positive_rate"), "inf": ("inference_ms_per_txn", "inference_ms")}.items():
        col = pick(log, *c)
        d[k] = pd.to_numeric(log[col], errors="coerce") if col else np.nan
    d = d.dropna(subset=["pr"]).sort_values("pr").drop_duplicates("label", keep="last")
    colours = [BLUE if selected(l) else GREY for l in d["label"]]

    fig, ax = plt.subplots(figsize=(7.5, .45 * len(d) + 1.2))
    ax.barh(d["label"], d["pr"], color=colours)
    if d["lo"].notna().any():
        ax.errorbar(d["pr"], d["label"], xerr=[(d["pr"] - d["lo"]).clip(lower=0), (d["hi"] - d["pr"]).clip(lower=0)],
                    fmt="none", ecolor="black", capsize=3, lw=1)
    for y, (v, hi) in enumerate(zip(d["pr"], d["hi"])):
        ax.text((hi if pd.notna(hi) else v) + .015, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test PR-AUC (bars: 95% bootstrap CI)")
    ax.axvline(0.033, color=RED, ls="--", lw=1)
    ax.text(0.04, len(d) - .45, "random baseline 0.033", color=RED, fontsize=8, va="center")
    ax.set_title("Model comparison on the chronological test set")
    save(fig, "ch5_model_comparison.png")

    r = d.dropna(subset=["rec", "fpr"])
    if not r.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for _, row in r.iterrows():
            ax.scatter(row["fpr"], row["rec"], s=70, color=BLUE if selected(row["label"]) else GREY, zorder=3)
            ax.annotate(row["label"], (row["fpr"], row["rec"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("Recall")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("Fraud caught against legitimate customers flagged")
        save(fig, "ch5_recall_vs_fpr.png")

    c = d.dropna(subset=["inf"])
    if not c.empty:
        fig, ax = plt.subplots(figsize=(7.5, .4 * len(c) + 1.2))
        ax.barh(c["label"], c["inf"], color=[BLUE if selected(l) else GREY for l in c["label"]])
        ax.set_xlabel("Inference time per transaction (ms)")
        ax.set_title("Computational cost of each configuration")
        save(fig, "ch5_inference_cost.png")


NICE = {"xgb": "XGBoost", "plain_xgb": "XGBoost", "rf": "Random Forest", "soft_voting": "Soft voting",
        "hybrid": "IF hybrid"}


def seed_figure(name: str, out: str, title: str):
    """Slope chart of test PR-AUC per seed: reference model on the left, challenger on the right."""
    df = load(name)
    if df is None:
        return
    seed = pick(df, "seed", "random_state", "run")
    cols = [c for c in df.columns if norm(c).endswith("prauc")]
    if seed is None or len(cols) < 2:
        print(f"  skip {out}: need a seed column and two *_pr_auc columns, found {list(df.columns)}")
        return
    ref = next((c for c in cols if "xgb" in c.lower()), cols[0])
    chal = next((c for c in cols if any(k in c.lower() for k in ("soft", "hybrid"))), cols[-1])
    order = [ref, chal]  # other members (e.g. Random Forest alone) are reported in the text
    labels = [NICE.get(c.lower().replace("_pr_auc", ""), c) for c in order]
    wide = df.set_index(seed)[order].apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(order))
    for s_, row in wide.iterrows():
        won = row[chal] > row[ref]
        ax.plot(x, row.values, color=GREEN if won else RED, lw=1.3, marker="o", ms=5, alpha=.85)
        ax.text(x[-1] + .06, row[chal], f"seed {s_}", fontsize=8, va="center", color="#555")
    ax.plot(x, wide.mean().values, color="black", lw=2.6, marker="D", ms=7, label="Mean of five seeds")
    ax.set_xticks(x, labels)
    ax.set_xlim(-.25, len(order) - .55)
    ax.set_ylabel("Test PR-AUC")
    diff = wide[chal] - wide[ref]
    ax.set_title(f"{title}\n{labels[-1]} better on {int((diff > 0).sum())} of {int(diff.notna().sum())} seeds, "
                 f"mean difference {diff.mean():+.4f}", fontsize=11)
    ax.plot([], [], color=GREEN, label=f"{labels[-1]} wins")
    ax.plot([], [], color=RED, label=f"{labels[-1]} loses")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    save(fig, out)


def band_figure():
    df = load("risk_bands_test.csv")
    if df is None:
        return
    band = pick(df, "band", "risk_band")
    if band is None:
        print(f"  skip risk bands: no band column in {list(df.columns)}")
        return
    n = pick(df, "n", "count", "transactions", "share")
    rate = pick(df, "fraud_rate", "precision", "fraud_share")
    df = df.set_index(df[band].astype(str).str.upper()).reindex([b for b in BAND_ORDER if b in set(df[band].astype(str).str.upper())])
    fig, ax1 = plt.subplots(figsize=(6.8, 3.8))
    if n:
        vals = pd.to_numeric(df[n], errors="coerce")
        ax1.bar(df.index, vals, color=[BAND_COL[b] for b in df.index])
        ax1.set_ylabel(n.replace("_", " ").capitalize())
        ax1.set_yscale("log")
    if rate:
        ax2 = ax1.twinx()
        ax2.plot(df.index, pd.to_numeric(df[rate], errors="coerce"), color="black", marker="o", lw=2)
        ax2.set_ylabel(rate.replace("_", " ").capitalize())
        ax2.grid(False)
    ax1.set_title("Risk bands on the test set")
    save(fig, "ch5_risk_bands.png")


def latency_figure():
    df = load("latency_benchmark.csv")
    if df is None:
        return
    text = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    ms = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "ms" in c.lower()
          and "max" not in c.lower()]
    if not ms:
        print(f"  skip latency: no *_ms columns in {list(df.columns)}")
        return
    label = df[text].astype(str).agg(" | ".join, axis=1) if text else df.index.astype(str)
    plot = df[ms].set_index(label)
    fig, ax = plt.subplots(figsize=(7.5, .55 * len(plot) + 1.4))
    plot.plot(kind="barh", ax=ax, color=[BLUE, ORANGE, RED][:len(ms)])
    ax.invert_yaxis()
    ax.set_xlabel("Milliseconds")
    ax.set_ylabel("")
    ax.set_title("Decision latency benchmark")
    ax.legend([c.replace("_ms", "").replace("p95", "95th percentile").capitalize() for c in ms], frameon=False)
    save(fig, "ch5_latency.png")


if __name__ == "__main__":
    print("Building Chapter 5 result figures ...")
    model_figures()
    seed_figure("ensemble_seed_robustness.csv", "ch5_seeds_ensemble.png", "Soft voting against XGBoost")
    seed_figure("seed_robustness.csv", "ch5_seeds_hybrid.png", "Isolation Forest hybrid against XGBoost")
    band_figure()
    latency_figure()
    print("Done.")
