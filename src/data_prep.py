"""Load, filter and clean Lending Club data; build the default flag and out-of-time split.

Design choices (explained in the README):
* 36-month loans only, so every loan has the same default horizon.
* Resolved outcomes only (Fully Paid / Charged Off / Default). 'Current' loans are dropped
  because we don't yet know whether they will default; keeping them as "good" would
  understate the bad rate.
* PD features are restricted to fields known at origination (no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Default definition ---------------------------------------------------------------
BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

# --- Columns ----------------------------------------------------------------------------
# Known at origination: the only fields allowed as PD inputs.
ORIGINATION_COLS = [
    "loan_amnt", "term", "int_rate", "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "fico_range_low",
    "fico_range_high", "earliest_cr_line", "mort_acc", "pub_rec_bankruptcies",
    "addr_state", "installment", "application_type",
    # Credit-bureau attributes pulled at application (populated from 2013 onwards).
]
# Bureau attributes captured at application. This Kaggle version has NO FICO columns, so these
# carry the "bureau" signal instead. They are snapshot-at-origination, so no leakage.
BUREAU_FEATURES = [
    "acc_open_past_24mths", "bc_util", "percent_bc_gt_75", "num_tl_op_past_12m",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_tl", "tot_hi_cred_lim", "avg_cur_bal",
    "total_rev_hi_lim", "bc_open_to_buy", "pct_tl_nvr_dlq", "mths_since_recent_inq",
    "mths_since_last_delinq",
]
ORIGINATION_COLS += BUREAU_FEATURES
# Post-origination fields: used ONLY to build LGD/EAD targets, never as PD features.
OUTCOME_COLS = ["loan_status", "issue_d", "funded_amnt", "total_rec_prncp", "recoveries",
                "collection_recovery_fee", "total_pymnt", "last_pymnt_d"]
LEAKAGE_COLS = {
    "total_pymnt", "total_pymnt_inv", "recoveries", "last_pymnt_d", "last_pymnt_amnt",
    "out_prncp", "out_prncp_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "last_fico_range_high", "last_fico_range_low", "collection_recovery_fee",
    "debt_settlement_flag", "loan_status", "next_pymnt_d", "last_credit_pull_d",
    "funded_amnt", "funded_amnt_inv",
}

# Candidate PD features after cleaning. Grade / sub_grade / int_rate are Lending Club's own
# pricing, so the main scorecard EXCLUDES them (see LC_PRICING_FEATURES).
LC_PRICING_FEATURES = ["grade", "sub_grade", "int_rate"]
NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "credit_hist_months", "mort_acc",
    "pub_rec_bankruptcies", "emp_length_yrs", "loan_to_income", "int_rate",
    *BUREAU_FEATURES,
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose", "grade",
                        "sub_grade", "addr_state"]


def load_raw(path: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read the Kaggle CSV, keeping only the columns we need (saves memory on 2.2M rows)."""
    wanted = set(ORIGINATION_COLS + OUTCOME_COLS) if usecols is None else set(usecols)
    # Read in chunks and keep only 36-month loans as we go, so the 1.2 GB file fits in memory.
    parts = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False,
                             chunksize=200_000):
        chunk = chunk[chunk["term"].astype(str).str.contains("36", na=False)]
        for c in chunk.columns:
            if chunk[c].dtype == object or str(chunk[c].dtype).startswith("str"):
                chunk[c] = chunk[c].astype("category")
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    for c in df.columns:  # categories differ across chunks -> concat gives object; re-compress
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
    # The Kaggle file ends with a couple of summary rows that have no loan data.
    return df.dropna(subset=["loan_status", "issue_d"])


def _pct_to_float(s: pd.Series) -> pd.Series:
    """'13.5%' -> 13.5. Some file versions already store floats, so handle both."""
    if s.dtype == object:
        s = s.str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _emp_length_to_years(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '10+ years' -> 10, missing -> NaN (kept as its own WoE bin later)."""
    s = s.astype("string").str.strip()
    s = s.replace({"< 1 year": "0", "10+ years": "10"})
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type conversions and derived origination features."""
    df = df.copy()
    for c in df.columns:   # back to plain strings for the text-cleaning steps below
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object)
    df["term"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0])
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["issue_year"] = df["issue_d"].dt.year
    df["int_rate"] = _pct_to_float(df["int_rate"])
    df["revol_util"] = _pct_to_float(df["revol_util"])
    df["emp_length_yrs"] = _emp_length_to_years(df["emp_length"])

    # Credit history length in months, measured at the loan's issue date (not today).
    ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_hist_months"] = ((df["issue_d"] - ecl).dt.days / 30.44).round()

    # Single FICO figure (midpoint of the band) -- only if this file version has FICO.
    if {"fico_range_low", "fico_range_high"} <= set(df.columns):
        df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    # Affordability ratio: loan size relative to income (origination-time info only).
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)

    # Group rare/odd home_ownership levels.
    df["home_ownership"] = df["home_ownership"].replace({"ANY": "OTHER", "NONE": "OTHER"})
    return df


def add_default_flag(df: pd.DataFrame) -> pd.DataFrame:
    """bad = 1 for Charged Off / Default; 0 for Fully Paid. Unresolved loans are dropped."""
    df = df[df["loan_status"].isin(BAD_STATUSES | GOOD_STATUSES)].copy()
    df["bad"] = df["loan_status"].isin(BAD_STATUSES).astype(int)
    return df


def scope(df: pd.DataFrame, term: int, years: list[int]) -> pd.DataFrame:
    """Keep individual (not joint) 36-month loans issued in the chosen years.

    Joint applications only appear from late 2015 and their income/DTI fields mean something
    different (combined household), so they are excluded for a like-for-like population.
    """
    m = (df["term"] == term) & (df["issue_year"].isin(years))
    if "application_type" in df:
        m &= df["application_type"].fillna("Individual").eq("Individual")
    return df[m].copy()


def oot_split(df: pd.DataFrame, train_years: list[int], oot_years: list[int]):
    """Out-of-time split by issue year (not random) -- mirrors how a model is used in practice."""
    train = df[df["issue_year"].isin(train_years)].copy()
    oot = df[df["issue_year"].isin(oot_years)].copy()
    return train, oot


def status_counts_by_year(df: pd.DataFrame, term: int) -> pd.DataFrame:
    """Outcome counts per issue year, including unresolved loans -- used to choose the split."""
    d = df[df["term"] == term]
    t = pd.crosstab(d["issue_year"], d["loan_status"])
    resolved = t[[c for c in t.columns if c in BAD_STATUSES | GOOD_STATUSES]].sum(axis=1)
    out = pd.DataFrame({"total": t.sum(axis=1), "resolved": resolved})
    out["resolved_share"] = out["resolved"] / out["total"]
    bads = t[[c for c in t.columns if c in BAD_STATUSES]].sum(axis=1)
    out["bad_rate_resolved"] = bads / out["resolved"]
    return out
