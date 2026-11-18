"""
ch5_eda_figures.py
------------------
Exploratory data analysis figures and summary tables for Chapter 5 (Section 5.3).
Reads only the raw PaySim file and the chronological split defined in src/config.py.

Writes to FIGURES_DIR:
  eda_type_distribution.png   transactions and fraud by transaction type (Figure 5.5)
  eda_class_imbalance.png     legitimate vs fraudulent counts, overall and in scope (Figure 5.6)
  eda_amount_distribution.png amount distribution, fraud vs legitimate, TRANSFER and CASH_OUT (Figure 5.7)
  eda_fraud_by_hour.png       fraud rate by hour of day (Figure 5.8)
  eda_balance_drain.png       share of the sender's balance moved, fraud vs legitimate (Figure 5.9)
  eda_split_timeline.png      transactions per step with the train/validation/test split (Figure 5.10)
Writes to METRICS_DIR:
  eda_by_type.csv             fills Table 5.2
  eda_partitions.csv          fills Table 5.3

Run from the project root:
    python -m src.ch5_eda_figures
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import FIGURES_DIR, METRICS_DIR, RAW_DATA_PATH, VAL_MAX_STEP  # noqa: E402

TRAIN_MAX_STEP = 520
IN_SCOPE = ["TRANSFER", "CASH_OUT"]
LEGIT, FRAUD, MUTED = "#4A6FA5", "#C0392B", "#9AA3AE"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
    "axes.titleweight": "bold", "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(FIGURES_DIR) / name)
    plt.close(fig)
    print(f"  saved {name}")


def main() -> None:
    print(f"Reading {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH, usecols=["step", "type", "amount", "oldbalanceOrg", "isFraud"])
    print(f"  {len(df):,} transactions, {int(df['isFraud'].sum()):,} fraudulent, {df['step'].max()} steps")
    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)

    # Table 5.2 and Figure 5.3: by type
    by_type = (df.groupby("type").agg(transactions=("isFraud", "size"), fraudulent=("isFraud", "sum"))
               .sort_values("transactions", ascending=False))
    by_type["fraud_rate_pct"] = (100 * by_type["fraudulent"] / by_type["transactions"]).round(3)
    by_type.to_csv(Path(METRICS_DIR) / "eda_by_type.csv")
    print(by_type.to_string())

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(by_type))
    ax.bar(x - .2, by_type["transactions"], .4, label="All transactions", color=LEGIT)
    ax.bar(x + .2, by_type["fraudulent"].replace(0, np.nan), .4, label="Fraudulent", color=FRAUD)
    ax.set_yscale("log")
    ax.set_xticks(x, by_type.index)
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Transactions and fraud by transaction type")
    for i, f in enumerate(by_type["fraudulent"]):
        if f == 0:
            ax.text(i + .2, .03, "no fraud", ha="center", va="bottom", fontsize=8, color=MUTED, rotation=90,
                    transform=ax.get_xaxis_transform())
    ax.legend(frameon=False)
    save(fig, "eda_type_distribution.png")

    # Figure 5.4: class imbalance
    scope = df[df["type"].isin(IN_SCOPE)]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, d, title in ((axes[0], df, "All types"), (axes[1], scope, "TRANSFER and CASH_OUT")):
        counts = d["isFraud"].value_counts().reindex([0, 1]).fillna(0)
        ax.bar(["Legitimate", "Fraudulent"], counts, color=[LEGIT, FRAUD])
        ax.set_yscale("log")
        ax.set_title(f"{title}\nfraud rate {100 * counts[1] / counts.sum():.2f}%")
        for i, v in enumerate(counts):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Transactions (log scale)")
    save(fig, "eda_class_imbalance.png")

    # Amount distribution: histogram of log10(amount), so every bin has the same width
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    la = np.log10(scope["amount"].clip(lower=1))
    bins = np.linspace(0, np.ceil(la.max()), 61)
    ax.hist(la[scope["isFraud"] == 0], bins=bins, density=True, alpha=.6, color=LEGIT, label="Legitimate")
    ax.hist(la[scope["isFraud"] == 1], bins=bins, density=True, alpha=.7, color=FRAUD, label="Fraudulent")
    ticks = np.arange(0, np.ceil(la.max()) + 1)
    ax.set_xticks(ticks, [f"{10 ** t:,.0f}" for t in ticks], fontsize=8)
    ax.set_xlabel("Amount (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount, TRANSFER and CASH_OUT")
    ax.legend(frameon=False)
    save(fig, "eda_amount_distribution.png")

    # Figure 5.6: fraud rate by hour
    scope = scope.assign(hour=scope["step"] % 24)
    hourly = scope.groupby("hour").agg(n=("isFraud", "size"), fraud=("isFraud", "sum"))
    hourly["rate"] = 100 * hourly["fraud"] / hourly["n"]
    fig, ax1 = plt.subplots(figsize=(7.5, 3.6))
    ax1.bar(hourly.index, hourly["n"], color=MUTED, alpha=.5, label="Transactions")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Transactions")
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly["rate"], color=FRAUD, marker="o", lw=2, label="Fraud rate")
    ax2.set_ylabel("Fraud rate (%)", color=FRAUD)
    ax2.grid(False)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_title("Transaction volume and fraud rate by hour of day")
    save(fig, "eda_fraud_by_hour.png")

    # Balance drain: share of each class by how much of the sender's balance was moved
    s = scope[scope["oldbalanceOrg"] > 0].copy()
    ratio = s["amount"] / s["oldbalanceOrg"]
    cats = pd.cut(ratio, [0, .5, .999, 1.001, np.inf],
                  labels=["Under half", "Half to almost all", "Whole balance", "More than balance"])
    tab = (pd.crosstab(cats, s["isFraud"], normalize="columns") * 100).reindex(columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(tab))
    ax.bar(x - .2, tab[0], .4, color=LEGIT, label="Legitimate")
    ax.bar(x + .2, tab[1], .4, color=FRAUD, label="Fraudulent")
    for i in range(len(tab)):
        ax.text(i - .2, tab[0].iloc[i] + 1, f"{tab[0].iloc[i]:.0f}%", ha="center", fontsize=8)
        ax.text(i + .2, tab[1].iloc[i] + 1, f"{tab[1].iloc[i]:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x, tab.index)
    ax.set_ylabel("Share of transactions (%)")
    ax.set_title("How much of the sender's balance each transaction moved")
    ax.legend(frameon=False)
    save(fig, "eda_balance_drain.png")
    print(f"  share of frauds moving the whole balance: {tab.loc['Whole balance', 1]:.1f}%")

    # Table 5.3 and Figure 5.8: chronological split
    def part(step):
        return "Training" if step <= TRAIN_MAX_STEP else "Validation" if step <= VAL_MAX_STEP else "Test"
    sc = df[df["type"].isin(IN_SCOPE)]
    parts = sc.assign(partition=sc["step"].map(part)).groupby("partition").agg(
        first_step=("step", "min"), last_step=("step", "max"), transactions=("isFraud", "size"),
        fraudulent=("isFraud", "sum")).reindex(["Training", "Validation", "Test"])
    parts["fraud_rate_pct"] = (100 * parts["fraudulent"] / parts["transactions"]).round(3)
    parts.to_csv(Path(METRICS_DIR) / "eda_partitions.csv")
    print(parts.to_string())

    per_step = sc.groupby("step").agg(n=("isFraud", "size"), fraud=("isFraud", "sum"))
    fig, ax1 = plt.subplots(figsize=(8, 3.6))
    ax1.fill_between(per_step.index, per_step["n"], color=LEGIT, alpha=.35, lw=0)
    ax1.set_yscale("symlog")
    ax1.set_ylabel("Transactions per step")
    ax1.set_xlabel("Simulation step (hour)")
    ax2 = ax1.twinx()
    ax2.plot(per_step.index, per_step["fraud"], color=FRAUD, lw=.9)
    ax2.set_ylabel("Fraudulent per step", color=FRAUD)
    ax2.grid(False)
    for x0, x1, label, c in ((0, TRAIN_MAX_STEP, "Training", "#E8EEF6"),
                             (TRAIN_MAX_STEP, VAL_MAX_STEP, "Validation", "#FFF4D6"),
                             (VAL_MAX_STEP, per_step.index.max(), "Test", "#F9E0DD")):
        ax1.axvspan(x0, x1, color=c, zorder=0)
        ax1.text((x0 + x1) / 2, ax1.get_ylim()[1] * .55, label, ha="center", fontsize=10, weight="bold")
    ax1.set_title("Chronological split of TRANSFER and CASH_OUT transactions")
    save(fig, "eda_split_timeline.png")
    print("Done.")


if __name__ == "__main__":
    main()
