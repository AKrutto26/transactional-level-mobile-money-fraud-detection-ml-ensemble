"""
rules.py
--------
Deterministic rules engine for the fraud detection system.

Used in two places with identical logic:
  - evaluate_rules.py : measures each rule on validation data, assigns its role,
                        and evaluates rules + ML on the test set
  - the API           : applies the saved rule roles to live transactions

Every rule uses only information available at authorisation time.

Rule roles (assigned on VALIDATION data; thresholds fixed before running):
  CRITICAL_RULE -> forces at least CRITICAL (decline and escalate)
                   requires precision >= 95% on validation
  ESCALATE      -> forces at least HIGH (hold and alert)
                   requires precision >= 50%
  STEP_UP       -> forces at least MEDIUM (step-up verification)
                   requires precision >= 10%
  UNUSED        -> too imprecise to act on
  A rule must also fire at least MIN_FLAGS times on validation to receive a role.

The final band of a transaction is the HIGHER of its ML band and the band
forced by its triggered rules. Rules can raise a band; they never lower one.

Non-modelled transaction types (PAYMENT, CASH_IN, DEBIT) never contain fraud
in PaySim and are routed as LOW by route_transaction_type().
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BAND_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
BAND_RANK = {b: i for i, b in enumerate(BAND_ORDER)}
ROLE_TO_BAND = {"CRITICAL_RULE": "CRITICAL", "ESCALATE": "HIGH", "STEP_UP": "MEDIUM"}

ROLE_THRESHOLDS = [("CRITICAL_RULE", 0.95), ("ESCALATE", 0.50), ("STEP_UP", 0.10)]
MIN_FLAGS = 30
LARGE_AMOUNT = 200_000.0          # business threshold, implemented independently of isFlaggedFraud
MODELLED_TYPES = {"TRANSFER", "CASH_OUT"}

REQUIRED_COLUMNS = [
    "amount", "type_TRANSFER", "empties_orig_account", "dest_is_new",
    "dest_txn_count_before", "dest_time_since_last_step",
]

RULE_DESCRIPTIONS = {
    "ACCOUNT_DRAIN": "Transaction moves the entire origin balance",
    "DRAIN_TO_NEW_DEST": "Entire origin balance moved to a destination with no history",
    "DRAIN_BY_TRANSFER": "Entire origin balance moved by TRANSFER",
    "LARGE_AMOUNT": f"Amount of at least {LARGE_AMOUNT:,.0f}",
    "LARGE_DRAIN": f"Entire origin balance moved, amount at least {LARGE_AMOUNT:,.0f}",
    "RAPID_REPEAT_DEST": "Destination already received a transaction in the same hour",
}


def route_transaction_type(tx_type: str) -> str:
    """'MODEL' for types where fraud occurs; 'LOW' for types routed straight to approval."""
    return "MODEL" if tx_type.upper().replace("-", "_") in MODELLED_TYPES else "LOW"


def evaluate_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix: one column per rule, one row per transaction."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Rules engine is missing columns: {missing}")
    drain = df["empties_orig_account"].to_numpy() >= 0.5
    new_dest = df["dest_is_new"].to_numpy() >= 0.5
    transfer = df["type_TRANSFER"].to_numpy() >= 0.5
    large = df["amount"].to_numpy() >= LARGE_AMOUNT
    repeat = (df["dest_txn_count_before"].to_numpy() > 0) & (df["dest_time_since_last_step"].to_numpy() == 0)
    return pd.DataFrame({
        "ACCOUNT_DRAIN": drain,
        "DRAIN_TO_NEW_DEST": drain & new_dest,
        "DRAIN_BY_TRANSFER": drain & transfer,
        "LARGE_AMOUNT": large,
        "LARGE_DRAIN": drain & large,
        "RAPID_REPEAT_DEST": repeat,
    }, index=df.index)


def rule_statistics(flags: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Flag count, precision, recall and lift over the base fraud rate for each rule."""
    base = y.mean()
    rows = []
    for rule in flags.columns:
        f = flags[rule].to_numpy()
        n = int(f.sum())
        tp = int(y[f].sum())
        precision = tp / n if n else 0.0
        rows.append({
            "rule": rule,
            "description": RULE_DESCRIPTIONS[rule],
            "flags": n,
            "flag_rate_pct": round(n / len(y) * 100, 3),
            "fraud_flagged": tp,
            "precision": round(precision, 4),
            "recall": round(tp / y.sum(), 4) if y.sum() else 0.0,
            "lift_over_base_rate": round(precision / base, 2) if base else 0.0,
        })
    return pd.DataFrame(rows)


def assign_roles(stats: pd.DataFrame) -> dict:
    """Role for each rule from its validation precision, using the fixed thresholds."""
    roles = {}
    for _, r in stats.iterrows():
        role = "UNUSED"
        if r["flags"] >= MIN_FLAGS:
            for name, threshold in ROLE_THRESHOLDS:
                if r["precision"] >= threshold:
                    role = name
                    break
        roles[r["rule"]] = role
    return roles


def rule_bands(flags: pd.DataFrame, roles: dict) -> tuple[np.ndarray, list[list[str]]]:
    """Minimum band forced by rules, plus the list of acting rules per transaction."""
    rank = np.zeros(len(flags), dtype=int)
    triggered = [[] for _ in range(len(flags))]
    for rule, role in roles.items():
        if role == "UNUSED" or rule not in flags.columns:
            continue
        hit = flags[rule].to_numpy()
        rank = np.where(hit, np.maximum(rank, BAND_RANK[ROLE_TO_BAND[role]]), rank)
        for i in np.flatnonzero(hit):
            triggered[i].append(rule)
    return np.array(BAND_ORDER, dtype=object)[rank], triggered


def combine_bands(ml_band: np.ndarray, rule_band: np.ndarray) -> np.ndarray:
    """Final band = the higher of the ML band and the rule-forced band."""
    ml_rank = np.array([BAND_RANK[b] for b in ml_band])
    rule_rank = np.array([BAND_RANK[b] for b in rule_band])
    return np.array(BAND_ORDER, dtype=object)[np.maximum(ml_rank, rule_rank)]


def save_rules_config(path: Path, roles: dict, stats: pd.DataFrame) -> None:
    config = {
        "roles": roles,
        "role_thresholds": dict(ROLE_THRESHOLDS),
        "min_flags": MIN_FLAGS,
        "large_amount": LARGE_AMOUNT,
        "descriptions": RULE_DESCRIPTIONS,
        "validation_precision": dict(zip(stats["rule"], stats["precision"])),
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_rules_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
