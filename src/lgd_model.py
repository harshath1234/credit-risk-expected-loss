"""Two-stage LGD model on defaulted loans.

Target
    Exposure at default (EAD_actual) = funded_amnt - total_rec_prncp  (principal still owed)
    Recovery rate (RR)               = recoveries / EAD_actual, clipped to [0, 1]
    LGD                              = 1 - RR
Gross recoveries are used (collection fees ignored) and are not discounted -- see limitations.

Why two stages? Lending Club recoveries are zero for a large share of defaults and spread
out otherwise, so LGD is bimodal. One linear regression would predict the middle of two humps.
    Stage 1: P(RR > 0)              -- logistic regression (Statsmodels Logit)
    Stage 2: E[RR | RR > 0]         -- fractional logit (Binomial GLM, logit link) so 0 < RR < 1
    E[RR]  = Stage1 * Stage2;  LGD = 1 - E[RR]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .regression_features import DesignMatrix


def lgd_targets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ead_actual"] = (d["funded_amnt"] - d["total_rec_prncp"]).clip(lower=0)
    rr = d["recoveries"] / d["ead_actual"].replace(0, np.nan)
    d["rr"] = rr.clip(0, 1)
    d["lgd_actual"] = 1 - d["rr"]
    d["any_recovery"] = (d["recoveries"] > 0).astype(int)
    # Loans fully repaid in principal but flagged bad have no exposure -> excluded from LGD fit.
    return d[d["ead_actual"] > 0]


class TwoStageLGD:
    def fit(self, defaults: pd.DataFrame):
        self.dm = DesignMatrix().fit(defaults)
        X = self.dm.transform(defaults)
        self.stage1 = sm.Logit(defaults["any_recovery"].values, X).fit(disp=0, maxiter=200)
        pos = defaults["rr"] > 0
        self.stage2 = sm.GLM(defaults.loc[pos, "rr"].values, X[pos.values],
                             family=sm.families.Binomial()).fit()
        return self

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.dm.transform(df)
        p = self.stage1.predict(X)
        m = self.stage2.predict(X)
        return pd.DataFrame({"p_recovery": np.asarray(p), "rr_if_recovery": np.asarray(m),
                             "lgd_pred": 1 - np.asarray(p) * np.asarray(m)}, index=df.index)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_components(df)["lgd_pred"].values


class SegmentAverageLGD:
    """Naive benchmark: mean LGD by grade x purpose group (falls back to grade, then overall)."""

    def fit(self, defaults: pd.DataFrame):
        d = defaults.assign(purpose_grp=_purpose_grp(defaults))
        self.cell = d.groupby(["grade", "purpose_grp"])["lgd_actual"].mean()
        self.grade = d.groupby("grade")["lgd_actual"].mean()
        self.overall = d["lgd_actual"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        d = df.assign(purpose_grp=_purpose_grp(df))
        key = pd.MultiIndex.from_arrays([d["grade"], d["purpose_grp"]])
        out = pd.Series(self.cell.reindex(key).values, index=d.index)
        out = out.fillna(d["grade"].map(self.grade)).fillna(self.overall)
        return out.values


def _purpose_grp(df):
    from .regression_features import MAJOR_PURPOSES
    return df["purpose"].where(df["purpose"].isin(MAJOR_PURPOSES), "other")


def regression_metrics(actual, pred) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = pred - actual
    ss_res, ss_tot = np.sum(err ** 2), np.sum((actual - actual.mean()) ** 2)
    return {"n": int(len(actual)), "mean_actual": float(actual.mean()),
            "mean_pred": float(pred.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "r2": float(1 - ss_res / ss_tot)}


def by_group(df: pd.DataFrame, actual: str, pred: str, group) -> pd.DataFrame:
    g = df.groupby(group, observed=True)
    return pd.DataFrame({"n": g.size(), "actual_mean": g[actual].mean(),
                         "pred_mean": g[pred].mean()}).reset_index()
