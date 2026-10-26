"""Expected Loss = PD x LGD x EAD, portfolio views, back-test, stress test and Monte Carlo.

Note on horizon: PD here is the probability of default over the 36-month life of the loan,
so EL is a *lifetime* expected loss per loan (closest to IFRS 9 Stage 2 lifetime ECL,
without discounting or macro scenarios) -- not a Basel 12-month EL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def loan_level_el(pd_hat, lgd_hat, ead_hat) -> np.ndarray:
    return np.asarray(pd_hat) * np.asarray(lgd_hat) * np.asarray(ead_hat)


def portfolio_summary(df: pd.DataFrame) -> dict:
    """df needs: el, funded_amnt, pd, lgd_pred, ead_pred."""
    total_el = df["el"].sum()
    exposure = df["funded_amnt"].sum()
    top = df.nlargest(int(np.ceil(0.10 * len(df))), "pd")
    return {"n_loans": int(len(df)), "exposure": float(exposure),
            "total_el": float(total_el), "el_pct_of_exposure": float(total_el / exposure),
            "mean_pd": float(df["pd"].mean()), "mean_lgd": float(df["lgd_pred"].mean()),
            "mean_ead_ratio": float(df["ead_pred"].sum() / exposure),
            "top10pct_pd_share_of_el": float(top["el"].sum() / total_el)}


def el_by(df: pd.DataFrame, group) -> pd.DataFrame:
    g = df.groupby(group, observed=True)
    t = pd.DataFrame({"n": g.size(), "exposure": g["funded_amnt"].sum(), "el": g["el"].sum(),
                      "mean_pd": g["pd"].mean(), "mean_lgd": g["lgd_pred"].mean()})
    t["el_pct"] = t["el"] / t["exposure"]
    t["share_of_el"] = t["el"] / t["el"].sum()
    if "realised_loss" in df:
        t["realised_loss"] = g["realised_loss"].sum()
        t["realised_pct"] = t["realised_loss"] / t["exposure"]
    return t.reset_index()


def realised_loss(df: pd.DataFrame) -> np.ndarray:
    """Actual loss on loans that defaulted: outstanding principal minus recoveries (>= 0)."""
    ead_act = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0)
    loss = (ead_act - df["recoveries"]).clip(lower=0)
    return np.where(df["bad"] == 1, loss, 0.0)


def stress_test(df: pd.DataFrame, pd_mult: float, lgd_add: float) -> pd.DataFrame:
    base = df["el"].sum()
    pd_s = np.clip(df["pd"] * pd_mult, 0, 1)
    lgd_s = np.clip(df["lgd_pred"] + lgd_add, 0, 1)
    scen = {
        "Base": base,
        f"PD x{pd_mult}": (pd_s * df["lgd_pred"] * df["ead_pred"]).sum(),
        f"LGD +{int(lgd_add * 100)}pts": (df["pd"] * lgd_s * df["ead_pred"]).sum(),
        "Combined": (pd_s * lgd_s * df["ead_pred"]).sum(),
    }
    exp = df["funded_amnt"].sum()
    return pd.DataFrame([{"scenario": k, "el": v, "el_pct": v / exp, "vs_base": v / base - 1}
                         for k, v in scen.items()])


def monte_carlo(df: pd.DataFrame, n_sims: int, seed: int, rho: float = 0.0,
                batch: int = 50) -> dict:
    """Simulate portfolio loss = sum_i D_i * LGD_i * EAD_i with D_i ~ Bernoulli.

    rho = 0   -> independent defaults (diversifies away almost all risk).
    rho > 0   -> one-factor Gaussian copula (the Vasicek model behind Basel IRB): every loan
                 shares a systematic factor Z, so defaults cluster in bad years.
                 Loan i defaults if sqrt(rho)*Z + sqrt(1-rho)*e_i < Phi^-1(PD_i).
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    pdv = np.clip(df["pd"].values, 1e-9, 1 - 1e-9)
    sev = (df["lgd_pred"] * df["ead_pred"]).values
    thr = norm.ppf(pdv)
    losses = np.empty(n_sims)
    for s0 in range(0, n_sims, batch):
        k = min(batch, n_sims - s0)
        if rho == 0:
            u = rng.random((k, len(pdv)))
            d = u < pdv
        else:
            z = rng.standard_normal((k, 1))
            e = rng.standard_normal((k, len(pdv)))
            d = (np.sqrt(rho) * z + np.sqrt(1 - rho) * e) < thr
        losses[s0:s0 + k] = d @ sev
    el = float(sev @ pdv)
    q99, q999 = np.quantile(losses, [0.99, 0.999])
    return {"rho": rho, "n_sims": n_sims, "analytic_el": el, "sim_mean": float(losses.mean()),
            "q99": float(q99), "q999": float(q999), "ul_99": float(q99 - losses.mean()),
            "ul_999": float(q999 - losses.mean()), "losses": losses}


def cohort_default_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative default rate by months on book, per issue-year cohort.

    Default timing proxy: months from issue_d to last_pymnt_d (the last payment before
    charge-off). Lending Club charges off ~4-5 months after that, so true default timing
    is slightly later -- the curve shape, not the exact month, is what matters here.
    """
    d = df.copy()
    lp = pd.to_datetime(d["last_pymnt_d"], format="%b-%Y", errors="coerce")
    d["mob_default"] = ((lp - d["issue_d"]).dt.days / 30.44).round()
    d.loc[d["mob_default"].isna() & (d["bad"] == 1), "mob_default"] = 0  # never paid
    rows = []
    for yr, g in d.groupby("issue_year"):
        n = len(g)
        bad_mob = g.loc[g["bad"] == 1, "mob_default"].clip(0, 36)
        for m in range(0, 37):
            rows.append({"issue_year": yr, "mob": m, "cum_default_rate": (bad_mob <= m).sum() / n})
    return pd.DataFrame(rows)
