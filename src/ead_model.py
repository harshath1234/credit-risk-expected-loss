"""EAD model for amortising term loans.

For an instalment loan there is no undrawn limit: exposure only goes DOWN as the borrower
repays. So EAD at default = principal still outstanding. We model the share of the original
amount still owed at default (a CCF-style ratio for term loans):

    ead_ratio = (funded_amnt - total_rec_prncp) / funded_amnt   in [0, 1]
    EAD       = ead_ratio * funded_amnt

(For revolving products -- cards, overdrafts -- a CCF is applied to the UNDRAWN limit
instead: EAD = drawn + CCF * (limit - drawn). Not applicable here.)

Model: fractional logit (Binomial GLM, logit link) so predictions stay inside [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .regression_features import DesignMatrix


def ead_targets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ead_ratio"] = ((d["funded_amnt"] - d["total_rec_prncp"]) / d["funded_amnt"]).clip(0, 1)
    d["ead_actual"] = d["ead_ratio"] * d["funded_amnt"]
    return d


class EADRatioModel:
    def fit(self, defaults: pd.DataFrame):
        self.dm = DesignMatrix().fit(defaults)
        X = self.dm.transform(defaults)
        self.res = sm.GLM(defaults["ead_ratio"].values, X,
                          family=sm.families.Binomial()).fit()
        return self

    def predict_ratio(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.res.predict(self.dm.transform(df)))

    def predict_ead(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_ratio(df) * df["funded_amnt"].values


class GradeAverageEAD:
    """Naive benchmark: average EAD ratio by grade."""

    def fit(self, defaults: pd.DataFrame):
        self.grade = defaults.groupby("grade")["ead_ratio"].mean()
        self.overall = defaults["ead_ratio"].mean()
        return self

    def predict_ratio(self, df):
        return df["grade"].map(self.grade).fillna(self.overall).values

    def predict_ead(self, df):
        return self.predict_ratio(df) * df["funded_amnt"].values
