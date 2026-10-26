"""Design matrix for the LGD and EAD regressions (origination features only).

Unlike the PD scorecard, LGD/EAD use plain transformed features + dummy variables, because
they are continuous-outcome regressions on a much smaller (defaults-only) sample.
Imputation values and dummy levels are learned on the training sample and re-used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# int_rate is left out: it is almost a deterministic function of grade (collinear).
NUM = ["log_loan_amnt", "log_annual_inc", "dti", "revol_util", "bc_util", "acc_open_past_24mths",
       "emp_length_yrs", "credit_hist_months", "inq_last_6mths", "mort_acc", "open_acc"]
CAT = ["grade", "home_ownership", "purpose_grp", "verification_status"]
MAJOR_PURPOSES = {"debt_consolidation", "credit_card", "home_improvement", "other"}


class DesignMatrix:
    def fit(self, df: pd.DataFrame):
        d = self._base(df)
        self.medians = d[NUM].median()
        self.means = d[NUM].fillna(self.medians).mean()
        self.stds = d[NUM].fillna(self.medians).std().replace(0, 1)
        self.levels = {c: sorted(d[c].dropna().unique()) for c in CAT}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self._base(df)
        # Standardise numerics so coefficients are comparable ("per 1 s.d.").
        X = (d[NUM].fillna(self.medians) - self.means) / self.stds
        for c in CAT:
            for lvl in self.levels[c][1:]:          # first level = reference category
                X[f"{c}={lvl}"] = (d[c] == lvl).astype(float)
        X.insert(0, "const", 1.0)
        return X

    @staticmethod
    def _base(df: pd.DataFrame) -> pd.DataFrame:
        d = pd.DataFrame(index=df.index)
        d["log_loan_amnt"] = np.log1p(df["loan_amnt"])
        d["log_annual_inc"] = np.log1p(df["annual_inc"].clip(lower=0))
        for c in ["dti", "revol_util", "bc_util", "acc_open_past_24mths", "emp_length_yrs",
                  "credit_hist_months", "inq_last_6mths", "mort_acc", "open_acc"]:
            d[c] = df[c].astype(float)
        d["dti"] = d["dti"].clip(upper=d["dti"].quantile(0.999))
        d["grade"] = df["grade"]
        d["home_ownership"] = df["home_ownership"].where(
            df["home_ownership"].isin(["MORTGAGE", "OWN", "RENT"]), "OTHER")
        d["purpose_grp"] = df["purpose"].where(df["purpose"].isin(MAJOR_PURPOSES), "other")
        d["verification_status"] = df["verification_status"]
        return d
