"""
ch5_amount_hour.py
------------------
Figure 5.5: transaction amount against hour of day, fraudulent vs legitimate (TRANSFER and CASH_OUT).

Fixes two problems in the usual version of this chart:
  * fraud is drawn ON TOP of legitimate points, so fraud during the day is not hidden;
  * the legend shows the correct colour for each class.
Legitimate transactions are randomly sampled (fraud is never sampled) so the plot stays readable.

Writes:
  FIGURES_DIR/eda_amount_vs_hour.png   -> paste into Figure 5.5
  METRICS_DIR/eda_amount_by_hour.csv   -> fraud count, fraud rate and median amounts per hour (for the text)

Run from the project root:
    python -m src.ch5_amount_hour
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
from src.config import FIGURES_DIR, METRICS_DIR, RAW_DATA_PATH  # noqa: E402

IN_SCOPE = ["TRANSFER", "CASH_OUT"]
LEGIT, FRAUD = "#4A6FA5", "#C0392B"
LEGIT_SAMPLE = 60_000
SEED = 42

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
    "axes.titleweight": "bold", "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def main() -> None:
    print(f"Reading {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH, usecols=["step", "type", "amount", "isFraud"])
    df = df[df["type"].isin(IN_SCOPE) & (df["amount"] > 0)].copy()
    df["hour"] = df["step"] % 24

    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0].sample(n=min(LEGIT_SAMPLE, (df["isFraud"] == 0).sum()), random_state=SEED)
    print(f"  plotting {len(fraud):,} fraudulent and {len(legit):,} sampled legitimate transactions")

    rng = np.random.default_rng(SEED)
    jitter = lambda n: rng.uniform(-0.28, 0.28, n)  # spread points within each hour column

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(legit["hour"] + jitter(len(legit)), legit["amount"], s=6, alpha=.25, color=LEGIT,
               linewidths=0, label=f"Legitimate (random sample of {len(legit):,})", rasterized=True)
    ax.scatter(fraud["hour"] + jitter(len(fraud)), fraud["amount"], s=9, alpha=.75, color=FRAUD,
               linewidths=0, label=f"Fraudulent (all {len(fraud):,})", rasterized=True, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(24))
    ax.set_xlim(-0.6, 23.6)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Transaction amount (log scale)")
    ax.set_title("Transaction amount against hour of day (TRANSFER and CASH_OUT)")
    ax.axvspan(-0.6, 5.5, color=FRAUD, alpha=.05, zorder=0)
    ax.legend(loc="lower right", frameon=True, markerscale=2.5)

    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(FIGURES_DIR) / "eda_amount_vs_hour.png")
    plt.close(fig)
    print("  saved eda_amount_vs_hour.png")

    # Supporting numbers for the paragraph above Figure 5.5
    by_hour = df.groupby("hour").agg(
        transactions=("isFraud", "size"),
        fraudulent=("isFraud", "sum"),
        median_amount_legit=("amount", lambda a: a[df.loc[a.index, "isFraud"] == 0].median()),
        median_amount_fraud=("amount", lambda a: a[df.loc[a.index, "isFraud"] == 1].median()),
    )
    by_hour["fraud_rate_pct"] = (100 * by_hour["fraudulent"] / by_hour["transactions"]).round(2)
    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
    by_hour.to_csv(Path(METRICS_DIR) / "eda_amount_by_hour.csv")
    print("  saved eda_amount_by_hour.csv")
    print(f"\n  median amount: fraud {fraud['amount'].median():,.0f}, legitimate {df.loc[df.isFraud == 0, 'amount'].median():,.0f}")
    night = fraud["hour"].between(0, 5).mean()
    print(f"  share of all fraud between 00:00 and 05:59: {100 * night:.1f}%")


if __name__ == "__main__":
    main()
