import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):
    """
    Split PaySim chronologically by step.

    Training:
        earliest 70% of time steps

    Validation:
        next 15%

    Test:
        final 15%

    The test set remains untouched during model development.
    """

    if not (
        len(df)
        == len(X)
        == len(y)
    ):
        raise ValueError(
            "df, X and y must contain the same number of rows."
        )

    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError(
            "Split ratios must be positive."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Training + validation ratio must be less than 1."
        )

    # --------------------------------------------------------
    # Obtain chronological PaySim time steps
    # --------------------------------------------------------

    unique_steps = np.sort(
        df["step"].unique()
    )

    number_of_steps = len(
        unique_steps
    )

    train_index = int(
        number_of_steps * train_ratio
    )

    validation_index = int(
        number_of_steps
        * (
            train_ratio
            + validation_ratio
        )
    )

    train_cutoff = unique_steps[
        train_index
    ]

    validation_cutoff = unique_steps[
        validation_index
    ]

    # --------------------------------------------------------
    # Chronological masks
    # --------------------------------------------------------

    train_mask = (
        df["step"]
        < train_cutoff
    )

    validation_mask = (
        (df["step"] >= train_cutoff)
        &
        (df["step"] < validation_cutoff)
    )

    test_mask = (
        df["step"]
        >= validation_cutoff
    )

    # --------------------------------------------------------
    # Create partitions
    # --------------------------------------------------------

    X_train = X.loc[
        train_mask
    ].copy()

    y_train = y.loc[
        train_mask
    ].copy()

    X_validation = X.loc[
        validation_mask
    ].copy()

    y_validation = y.loc[
        validation_mask
    ].copy()

    X_test = X.loc[
        test_mask
    ].copy()

    y_test = y.loc[
        test_mask
    ].copy()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if (
        len(X_train)
        + len(X_validation)
        + len(X_test)
        != len(X)
    ):
        raise RuntimeError(
            "Temporal split lost or duplicated rows."
        )

    logger.info(
        "Temporal split complete."
    )

    logger.info(
        "Training cutoff step: %s",
        train_cutoff,
    )

    logger.info(
        "Validation cutoff step: %s",
        validation_cutoff,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,

        "X_validation": X_validation,
        "y_validation": y_validation,

        "X_test": X_test,
        "y_test": y_test,

        "train_cutoff": int(
            train_cutoff
        ),

        "validation_cutoff": int(
            validation_cutoff
        ),
    }


def print_split_summary(
    df: pd.DataFrame,
    splits: dict,
) -> None:
    """
    Print class distributions for each temporal partition.
    """

    train_cutoff = (
        splits["train_cutoff"]
    )

    validation_cutoff = (
        splits["validation_cutoff"]
    )

    partitions = {
        "TRAIN": (
            splits["X_train"],
            splits["y_train"],
            df["step"] < train_cutoff,
        ),

        "VALIDATION": (
            splits["X_validation"],
            splits["y_validation"],
            (
                (df["step"] >= train_cutoff)
                &
                (
                    df["step"]
                    < validation_cutoff
                )
            ),
        ),

        "TEST": (
            splits["X_test"],
            splits["y_test"],
            df["step"] >= validation_cutoff,
        ),
    }

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEMPORAL TRAIN / VALIDATION / TEST SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTraining cutoff: step < {train_cutoff}"
    )

    print(
        "Validation: "
        f"{train_cutoff} <= step < "
        f"{validation_cutoff}"
    )

    print(
        f"Test: step >= {validation_cutoff}"
    )

    total_rows = 0
    total_fraud = 0

    for name, (
        X_part,
        y_part,
        mask,
    ) in partitions.items():

        rows = len(
            X_part
        )

        fraud = int(
            y_part.sum()
        )

        legitimate = (
            rows - fraud
        )

        fraud_rate = (
            fraud / rows
            if rows > 0
            else 0
        )

        step_min = int(
            df.loc[
                mask,
                "step"
            ].min()
        )

        step_max = int(
            df.loc[
                mask,
                "step"
            ].max()
        )

        print(
            f"\n{name}"
        )

        print(
            "-" * 40
        )

        print(
            f"Rows:        {rows:,}"
        )

        print(
            f"Steps:       {step_min} - {step_max}"
        )

        print(
            f"Legitimate:  {legitimate:,}"
        )

        print(
            f"Fraud:       {fraud:,}"
        )

        print(
            f"Fraud rate:  {fraud_rate:.6%}"
        )

        total_rows += rows
        total_fraud += fraud

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"Total rows:  {total_rows:,}"
    )

    print(
        f"Total fraud: {total_fraud:,}"
    )

    print(
        "=" * 70
    )