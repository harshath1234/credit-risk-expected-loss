"""PD validation metrics: discrimination, calibration, stability, rank ordering.

How to read them:
* AUC / Gini  -- discrimination: does the model rank bads above goods? Gini = 2*AUC - 1.
* KS          -- max gap between cumulative bad and good score distributions.
* Calibration -- are predicted PDs the right *level*? (Brier, Hosmer-Lemeshow, binomial per band)
* PSI / CSI   -- has the population shifted between development and the new period?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def auc_gini(y, pd_hat) -> tuple[float, float]:
    auc = roc_auc_score(y, pd_hat)
    return float(auc), float(2 * auc - 1)


def ks_stat(y, pd_hat) -> float:
    """KS = max |CDF_bad - CDF_good| = max(TPR - FPR) along the ROC curve."""
    fpr, tpr, _ = roc_curve(y, pd_hat)
    return float(np.max(tpr - fpr))


def brier(y, pd_hat) -> float:
    return float(brier_score_loss(y, pd_hat))


def calibration_table(y, pd_hat, n_bands: int = 10) -> pd.DataFrame:
    """Predicted vs observed bad rate by PD decile, with an exact binomial test per band."""
    d = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_hat)})
    d["band"] = pd.qcut(d["pd"].rank(method="first"), n_bands, labels=False) + 1
    t = d.groupby("band").agg(n=("y", "size"), bads=("y", "sum"),
                              pd_mean=("pd", "mean"), pd_min=("pd", "min"), pd_max=("pd", "max"))
    t["observed_rate"] = t["bads"] / t["n"]
    t["ratio_obs_pred"] = t["observed_rate"] / t["pd_mean"]
    t["binom_p"] = [stats.binomtest(int(b), int(n), float(p)).pvalue
                    for b, n, p in zip(t["bads"], t["n"], t["pd_mean"])]
    return t.reset_index()


def hosmer_lemeshow(y, pd_hat, n_groups: int = 10) -> dict:
    """HL chi-square on PD deciles; df = groups - 2. Large samples reject easily -- see README."""
    t = calibration_table(y, pd_hat, n_groups)
    exp_b = t["pd_mean"] * t["n"]
    obs_b = t["bads"]
    stat = float((((obs_b - exp_b) ** 2) / (exp_b * (1 - t["pd_mean"]))).sum())
    df = n_groups - 2
    return {"hl_stat": stat, "df": df, "p_value": float(stats.chi2.sf(stat, df))}


def psi(expected, actual, n_bands: int = 10, edges=None) -> tuple[float, pd.DataFrame]:
    """Population Stability Index. Bands = deciles of the EXPECTED (development) distribution.

    PSI = sum (actual% - expected%) * ln(actual% / expected%)
    < 0.10 stable | 0.10-0.25 monitor | > 0.25 significant shift
    """
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    if edges is None:
        qs = np.unique(np.quantile(expected, np.linspace(0, 1, n_bands + 1)[1:-1]))
        edges = [-np.inf, *qs, np.inf]
    e = pd.Series(pd.cut(expected, edges, right=True)).value_counts(sort=False).sort_index()
    a = pd.Series(pd.cut(actual, edges, right=True)).value_counts(sort=False).sort_index()
    t = pd.DataFrame({"expected_n": e.values, "actual_n": a.values}, index=e.index.astype(str))
    eps = 1e-6
    t["expected_pct"] = t["expected_n"] / t["expected_n"].sum() + eps
    t["actual_pct"] = t["actual_n"] / t["actual_n"].sum() + eps
    t["psi_band"] = (t["actual_pct"] - t["expected_pct"]) * np.log(t["actual_pct"] / t["expected_pct"])
    return float(t["psi_band"].sum()), t


def csi_by_feature(binner, features, X_dev: pd.DataFrame, X_new: pd.DataFrame) -> pd.DataFrame:
    """Characteristic Stability Index: PSI computed on each feature's coarse bins."""
    rows = []
    eps = 1e-6
    for f in features:
        spec = binner.specs[f]
        e = spec.labels(X_dev[f]).value_counts(normalize=True)
        a = spec.labels(X_new[f]).value_counts(normalize=True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx, fill_value=0) + eps, a.reindex(idx, fill_value=0) + eps
        rows.append({"feature": f, "csi": float(((a - e) * np.log(a / e)).sum())})
    out = pd.DataFrame(rows).sort_values("csi", ascending=False)
    out["status"] = pd.cut(out["csi"], [-1, 0.1, 0.25, 99], labels=["stable", "monitor", "shift"])
    return out


def gains_table(y, score, n_bands: int = 10) -> pd.DataFrame:
    """Score bands from riskiest (1) to safest (10): bad rate should fall band by band."""
    d = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    d["band"] = pd.qcut(d["score"].rank(method="first"), n_bands, labels=False) + 1
    t = d.groupby("band").agg(n=("y", "size"), bads=("y", "sum"),
                              score_min=("score", "min"), score_max=("score", "max"))
    t["bad_rate"] = t["bads"] / t["n"]
    t["cum_bad_capture"] = t["bads"].cumsum() / t["bads"].sum()
    t["cum_pop"] = t["n"].cumsum() / t["n"].sum()
    t["lift"] = t["bad_rate"] / (t["bads"].sum() / t["n"].sum())
    return t.reset_index()


def rank_order_ok(gains: pd.DataFrame) -> bool:
    """True if the bad rate strictly falls from the riskiest to the safest score band."""
    return bool(np.all(np.diff(gains["bad_rate"].values) < 0))


def summarise(y, pd_hat) -> dict:
    auc, gini = auc_gini(y, pd_hat)
    return {"n": int(len(y)), "bad_rate": float(np.mean(y)), "mean_pd": float(np.mean(pd_hat)),
            "auc": auc, "gini": gini, "ks": ks_stat(y, pd_hat), "brier": brier(y, pd_hat)}
