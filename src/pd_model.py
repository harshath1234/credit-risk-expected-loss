"""PD scorecard: feature selection, Statsmodels logit, Scikit-learn refit, points scaling.

Sign convention: we model P(bad). With WoE = ln(%good/%bad), a higher WoE means safer,
so every coefficient on a WoE feature should be NEGATIVE. A positive sign means the
feature is behaving oddly in combination with others (usually collinearity) -> remove.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ----------------------------------------------------------------------------------------
# Feature selection
# ----------------------------------------------------------------------------------------
def select_by_iv(iv_table: pd.DataFrame, iv_min: float, iv_suspicious: float):
    keep = iv_table.loc[iv_table["iv_coarse"] >= iv_min, "feature"].tolist()
    flagged = iv_table.loc[iv_table["iv_coarse"] > iv_suspicious, "feature"].tolist()
    return keep, flagged


def drop_correlated(Xw: pd.DataFrame, ranked: list[str], corr_max: float):
    """Walk features from highest IV down; drop any too correlated with one already kept."""
    corr = Xw[ranked].corr().abs()
    kept, dropped = [], {}
    for f in ranked:
        clash = [k for k in kept if corr.loc[f, k] > corr_max]
        if clash:
            dropped[f] = f"|corr| {corr.loc[f, clash[0]]:.2f} with {clash[0]}"
        else:
            kept.append(f)
    return kept, dropped


def vif_table(Xw: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(Xw, has_constant="add")
    vifs = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return pd.DataFrame({"feature": Xw.columns, "vif": vifs}).sort_values("vif", ascending=False)


def drop_high_vif(Xw: pd.DataFrame, vif_max: float):
    cols, dropped = list(Xw.columns), {}
    while len(cols) > 1:
        v = vif_table(Xw[cols])
        top = v.iloc[0]
        if top["vif"] <= vif_max:
            break
        dropped[top["feature"]] = f"VIF {top['vif']:.1f}"
        cols.remove(top["feature"])
    return cols, dropped


# ----------------------------------------------------------------------------------------
# Statsmodels logit with sign / significance checks
# ----------------------------------------------------------------------------------------
def fit_statsmodels(Xw: pd.DataFrame, y: pd.Series):
    X = sm.add_constant(Xw, has_constant="add")
    return sm.Logit(y.values, X).fit(disp=0, maxiter=200)


def stepwise_sign_pvalue(Xw: pd.DataFrame, y: pd.Series, p_max: float):
    """Remove (one at a time, worst first) features with a positive sign or p-value > p_max."""
    cols, dropped = list(Xw.columns), {}
    while True:
        res = fit_statsmodels(Xw[cols], y)
        params, pvals = res.params.drop("const"), res.pvalues.drop("const")
        wrong_sign = params[params > 0]
        if len(wrong_sign):
            f = wrong_sign.idxmax()
            dropped[f] = f"positive coefficient {params[f]:+.3f}"
            cols.remove(f)
            continue
        insig = pvals[pvals > p_max]
        if len(insig):
            f = insig.idxmax()
            dropped[f] = f"p-value {pvals[f]:.3f}"
            cols.remove(f)
            continue
        return cols, res, dropped


def coef_table(res) -> pd.DataFrame:
    ci = res.conf_int()
    return pd.DataFrame({
        "coef": res.params, "std_err": res.bse, "z": res.tvalues, "p_value": res.pvalues,
        "ci_low": ci[0], "ci_high": ci[1],
    }).rename_axis("feature").reset_index()


def fit_sklearn(Xw: pd.DataFrame, y: pd.Series, seed: int) -> LogisticRegression:
    """Same model, unpenalised, refit in scikit-learn for the scoring pipeline."""
    # C = inf means no regularisation, so this is the same maximum-likelihood fit as Statsmodels.
    m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000, tol=1e-10,
                           random_state=seed)
    return m.fit(Xw.values, y.values)


# ----------------------------------------------------------------------------------------
# Scorecard scaling
# ----------------------------------------------------------------------------------------
def scaling_params(pdo: float, base_score: float, base_odds: float):
    """Score = Offset + Factor * ln(good:bad odds). PDO points double the odds."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def points_table(binner, features: list[str], intercept: float, coefs: dict,
                 factor: float, offset: float) -> pd.DataFrame:
    """Points per attribute so that the sum over features = the scaled score.

    ln(odds_good) = -(b0 + sum_i b_i * WoE_i)
    Score         = Offset + Factor * ln(odds_good)
                  = sum_i [ -Factor * (b_i * WoE_i + b0 / n) + Offset / n ]
    The intercept and offset are spread evenly across the n features (common convention).
    """
    n = len(features)
    rows = []
    for f in features:
        spec = binner.specs[f]
        for label, w in spec.woe.items():
            pts = -factor * (coefs[f] * w + intercept / n) + offset / n
            rows.append({"feature": f, "bin": label, "woe": w, "coef": coefs[f],
                         "points_exact": pts, "points": int(round(pts)),
                         "share_train": float(spec.table.loc[label, "share"]),
                         "bad_rate_train": float(spec.table.loc[label, "bad_rate"])})
    return pd.DataFrame(rows)


def score_from_points(binner, X: pd.DataFrame, pts: pd.DataFrame) -> pd.Series:
    """Integer scorecard score: look up each applicant's bin, add up the points."""
    total = pd.Series(0, index=X.index, dtype=float)
    for f, grp in pts.groupby("feature", sort=False):
        lookup = dict(zip(grp["bin"], grp["points"]))
        labels = binner.specs[f].labels(X[f])
        # A bin never seen in training gets the population-average points for that feature.
        mean_pts = float(np.average(grp["points"], weights=grp["share_train"]))
        total += labels.map(lookup).fillna(mean_pts).astype(float).values
    return total


def score_to_pd(score, factor: float, offset: float):
    """Invert the scaling: odds_good = exp((score - Offset)/Factor); PD = 1/(1+odds_good)."""
    return 1.0 / (1.0 + np.exp((np.asarray(score) - offset) / factor))
