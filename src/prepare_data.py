import json
import logging
from typing import Optional

import pandas as pd

from src.config import (
    DATA_FILE,
    DATA_DTYPES,
    EXPECTED_COLUMNS,
    METRICS_DIR,
    ensure_directories,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_schema(df: pd.DataFrame) -> None:
    """
    Verify that the PaySim dataset contains the expected columns.
    """

    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    if unexpected_columns:
        logger.warning(
            "Unexpected columns found: %s",
            unexpected_columns,
        )

    logger.info("Dataset schema validation passed.")


# ============================================================
# DATA LOADER
# ============================================================

def load_paysim(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the PaySim dataset using explicit memory-conscious
    data types.

    Parameters
    ----------
    nrows:
        Optional number of rows to load. If None, the complete
        dataset is loaded.

    Returns
    -------
    pandas.DataFrame
        Loaded PaySim transaction data.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"PaySim dataset not found at:\n{DATA_FILE}"
        )

    logger.info("Loading PaySim dataset from:")
    logger.info("%s", DATA_FILE)

    if nrows is None:
        logger.info("Loading COMPLETE dataset.")
    else:
        logger.info(
            "Loading development sample: %s rows",
            f"{nrows:,}",
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype=DATA_DTYPES,
        nrows=nrows,
    )

    validate_schema(df)

    # FIXED:
    # Python logging does not support "%,d"
    logger.info(
        "Loaded %s rows and %d columns.",
        f"{len(df):,}",
        df.shape[1],
    )

    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    logger.info(
        "DataFrame memory usage: %.2f MB",
        memory_mb,
    )

    return df


# ============================================================
# GENERAL DATASET AUDIT
# ============================================================

def audit_dataset(
    df: pd.DataFrame,
) -> dict:
    """
    Perform the initial descriptive and integrity audit
    of the PaySim dataset.
    """

    logger.info("Beginning dataset audit...")

    # --------------------------------------------------------
    # General information
    # --------------------------------------------------------

    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    missing_values = {
        column: int(value)
        for column, value
        in df.isna().sum().items()
    }

    duplicate_rows = int(
        df.duplicated().sum()
    )

    # --------------------------------------------------------
    # Fraud distribution
    # --------------------------------------------------------

    fraud_counts_series = (
        df["isFraud"]
        .value_counts()
        .sort_index()
    )

    fraud_counts = {
        str(index): int(value)
        for index, value
        in fraud_counts_series.items()
    }

    fraud_count = int(
        df["isFraud"].sum()
    )

    fraud_prevalence = float(
        df["isFraud"].mean()
    )

    # --------------------------------------------------------
    # Existing PaySim rule flag
    # --------------------------------------------------------

    flagged_counts_series = (
        df["isFlaggedFraud"]
        .value_counts()
        .sort_index()
    )

    flagged_counts = {
        str(index): int(value)
        for index, value
        in flagged_counts_series.items()
    }

    fraud_flag_overlap = int(
        (
            (df["isFraud"] == 1)
            &
            (df["isFlaggedFraud"] == 1)
        ).sum()
    )

    # --------------------------------------------------------
    # Transaction types
    # --------------------------------------------------------

    transaction_type_counts_series = (
        df["type"]
        .value_counts()
    )

    transaction_type_counts = {
        str(index): int(value)
        for index, value
        in transaction_type_counts_series.items()
    }

    fraud_by_type = (
        df.groupby(
            "type",
            observed=True,
        )["isFraud"]
        .agg(
            transactions="count",
            fraud_count="sum",
        )
    )

    fraud_by_type["fraud_rate"] = (
        fraud_by_type["fraud_count"]
        / fraud_by_type["transactions"]
    )

    fraud_by_type = (
        fraud_by_type
        .sort_values(
            "fraud_count",
            ascending=False,
        )
    )

    # --------------------------------------------------------
    # Transaction amount statistics
    # --------------------------------------------------------

    amount_stats = {
        "minimum": float(
            df["amount"].min()
        ),
        "median": float(
            df["amount"].median()
        ),
        "mean": float(
            df["amount"].mean()
        ),
        "maximum": float(
            df["amount"].max()
        ),
    }

    # --------------------------------------------------------
    # Time-step statistics
    # --------------------------------------------------------

    step_counts = (
        df.groupby(
            "step",
            observed=True,
        )
        .size()
    )

    transactions_per_step = {
        "minimum": int(
            step_counts.min()
        ),
        "median": float(
            step_counts.median()
        ),
        "mean": float(
            step_counts.mean()
        ),
        "maximum": int(
            step_counts.max()
        ),
    }

    # --------------------------------------------------------
    # Account statistics
    # --------------------------------------------------------

    unique_origin_accounts = int(
        df["nameOrig"].nunique()
    )

    unique_destination_accounts = int(
        df["nameDest"].nunique()
    )

    # --------------------------------------------------------
    # Merchant identifiers
    # --------------------------------------------------------

    origin_merchant_count = int(
        df["nameOrig"]
        .str.startswith("M")
        .sum()
    )

    destination_merchant_count = int(
        df["nameDest"]
        .str.startswith("M")
        .sum()
    )

    # --------------------------------------------------------
    # Compile audit
    # --------------------------------------------------------

    audit = {
        "dataset_path": str(DATA_FILE),

        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        "column_names": list(
            df.columns
        ),

        "dtypes": {
            column: str(dtype)
            for column, dtype
            in df.dtypes.items()
        },

        "memory_mb": float(
            memory_mb
        ),

        "missing_values": (
            missing_values
        ),

        "duplicate_rows": (
            duplicate_rows
        ),

        "unique_origin_accounts": (
            unique_origin_accounts
        ),

        "unique_destination_accounts": (
            unique_destination_accounts
        ),

        "minimum_step": int(
            df["step"].min()
        ),

        "maximum_step": int(
            df["step"].max()
        ),

        "unique_steps": int(
            df["step"].nunique()
        ),

        "amount_statistics": (
            amount_stats
        ),

        "fraud_counts": (
            fraud_counts
        ),

        "fraud_count": (
            fraud_count
        ),

        "fraud_prevalence": (
            fraud_prevalence
        ),

        "fraud_percentage": float(
            fraud_prevalence * 100
        ),

        "isFlaggedFraud_counts": (
            flagged_counts
        ),

        "fraud_flag_overlap": (
            fraud_flag_overlap
        ),

        "transaction_type_counts": (
            transaction_type_counts
        ),

        "transactions_per_step": (
            transactions_per_step
        ),

        "origin_merchant_count": (
            origin_merchant_count
        ),

        "destination_merchant_count": (
            destination_merchant_count
        ),
    }

    # --------------------------------------------------------
    # Save audit outputs
    # --------------------------------------------------------

    ensure_directories()

    json_path = (
        METRICS_DIR
        / "dataset_audit.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=4,
        )

    fraud_by_type_path = (
        METRICS_DIR
        / "fraud_by_transaction_type.csv"
    )

    fraud_by_type.to_csv(
        fraud_by_type_path
    )

    logger.info(
        "Audit JSON saved to %s",
        json_path,
    )

    logger.info(
        "Transaction-type audit saved to %s",
        fraud_by_type_path,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PAYSIM DATASET AUDIT")
    print("=" * 70)

    print(
        f"\nDataset shape: "
        f"{df.shape[0]:,} rows x "
        f"{df.shape[1]} columns"
    )

    print(
        f"Memory usage: "
        f"{memory_mb:,.2f} MB"
    )

    print("\nColumns:")

    for column in df.columns:
        print(
            f"  {column:<20} "
            f"{df[column].dtype}"
        )

    print("\nMissing values:")
    print(
        df.isna().sum()
    )

    print(
        f"\nDuplicate rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"\nUnique origin accounts: "
        f"{unique_origin_accounts:,}"
    )

    print(
        f"Unique destination accounts: "
        f"{unique_destination_accounts:,}"
    )

    print(
        f"\nSteps: "
        f"{df['step'].min()} "
        f"to "
        f"{df['step'].max()}"
    )

    print(
        f"Unique steps: "
        f"{df['step'].nunique():,}"
    )

    print("\nAmount statistics:")

    for key, value in amount_stats.items():
        print(
            f"  {key:<10}: "
            f"{value:,.2f}"
        )

    print("\nFraud class distribution:")
    print(
        fraud_counts_series
    )

    print(
        f"\nFraud prevalence: "
        f"{fraud_prevalence:.6%}"
    )

    print("\nisFlaggedFraud distribution:")
    print(
        flagged_counts_series
    )

    print(
        f"\nFraud + flagged overlap: "
        f"{fraud_flag_overlap:,}"
    )

    print("\nTransaction type distribution:")
    print(
        transaction_type_counts_series
    )

    print("\nFraud by transaction type:")
    print(
        fraud_by_type
    )

    print("\nTransactions per step:")
    print(
        pd.Series(
            transactions_per_step
        )
    )

    print(
        "\nDestination merchant IDs: "
        f"{destination_merchant_count:,}"
    )

    print(
        "Origin merchant IDs: "
        f"{origin_merchant_count:,}"
    )

    print("\n" + "=" * 70)

    return audit


# ============================================================
# DATA LEAKAGE AUDIT
# ============================================================

def leakage_audit(
    df: pd.DataFrame,
) -> None:
    """
    Examine PaySim fields that could introduce data leakage
    into the primary machine-learning experiment.
    """

    print("\n" + "=" * 70)
    print("DATA LEAKAGE AUDIT")
    print("=" * 70)

    suspicious_columns = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ]

    print(
        "\nColumns excluded from the "
        "primary ML experiment:"
    )

    for column in suspicious_columns:
        print(
            f"  - {column}"
        )

    # --------------------------------------------------------
    # Rule flag versus fraud label
    # --------------------------------------------------------

    print("\n1. isFlaggedFraud vs isFraud")

    flag_table = pd.crosstab(
        df["isFlaggedFraud"],
        df["isFraud"],
        margins=True,
    )

    print(
        flag_table
    )

    # --------------------------------------------------------
    # Balance statistics
    # --------------------------------------------------------

    balance_columns = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    print(
        "\n2. Balance statistics by fraud class"
    )

    balance_summary = (
        df.groupby(
            "isFraud"
        )[balance_columns]
        .agg(
            ["mean", "median"]
        )
    )

    print(
        balance_summary
    )

    # --------------------------------------------------------
    # Zero balance analysis
    # --------------------------------------------------------

    print(
        "\n3. Zero-balance proportions "
        "by fraud class"
    )

    for column in balance_columns:

        zero_indicator = (
            df[column] == 0
        )

        zero_summary = (
            zero_indicator
            .groupby(
                df["isFraud"]
            )
            .mean()
        )

        print(
            f"\n{column}"
        )

        print(
            zero_summary
        )

    # --------------------------------------------------------
    # Fraud transaction-type check
    # --------------------------------------------------------

    print(
        "\n4. Fraud transactions by type"
    )

    fraud_type_table = (
        df.loc[
            df["isFraud"] == 1,
            "type",
        ]
        .value_counts()
    )

    print(
        fraud_type_table
    )

    # --------------------------------------------------------
    # High-value rule behaviour
    # --------------------------------------------------------

    print(
        "\n5. Existing flag behaviour "
        "for transfers over 200,000"
    )

    large_transfers = df[
        (df["type"] == "TRANSFER")
        &
        (df["amount"] > 200_000)
    ]

    print(
        f"\nTransfers > 200,000: "
        f"{len(large_transfers):,}"
    )

    print(
        "\nisFlaggedFraud distribution "
        "for these transfers:"
    )

    print(
        large_transfers[
            "isFlaggedFraud"
        ].value_counts()
    )

    print(
        "\nFraud distribution for "
        "these transfers:"
    )

    print(
        large_transfers[
            "isFraud"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Modelling policy
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print("PRIMARY MODELLING POLICY")

    print(
        """
Allowed raw transaction information:
  - step
  - type
  - amount
  - nameOrig (history derivation only)
  - nameDest (history derivation only)

Target:
  - isFraud

Excluded from the primary ML feature matrix:
  - isFlaggedFraud
  - oldbalanceOrg
  - newbalanceOrig
  - oldbalanceDest
  - newbalanceDest

Account identifiers will not be directly one-hot encoded.
They will only be used to derive leakage-safe historical
behavioural features based on previous transactions.
"""
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the PaySim dataset audit and leakage assessment.
    """

    ensure_directories()

    df = load_paysim()

    audit_dataset(
        df
    )

    leakage_audit(
        df
    )


if __name__ == "__main__":
    main()