"""Reproduce every result and figure:  python run_all.py   (optionally --until 1..5)

Phases
  1  Data: load, scope, default flag, out-of-time split, EDA
  2  PD scorecard: WoE/IV binning, feature selection, Statsmodels logit, scaling
  3  PD validation: Gini/AUC, KS, calibration, PSI/CSI, gains, LightGBM benchmark
  4  LGD (two-stage) and EAD (ratio) models
  5  Expected Loss, back-test, stress test, Monte Carlo, cohort curves
All headline numbers are written to reports/results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import data_prep as dp  # noqa: E402
from src import expected_loss as el  # noqa: E402
from src import pd_model as pm  # noqa: E402
from src import plots  # noqa: E402
from src import validation as val  # noqa: E402
from src.ead_model import EADRatioModel, GradeAverageEAD, ead_targets  # noqa: E402
from src.lgd_model import (SegmentAverageLGD, TwoStageLGD, by_group, lgd_targets,  # noqa: E402
                           regression_metrics)
from src.woe_binning import WoEBinner  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
SEED = CFG["seed"]
REP = ROOT / CFG["paths"]["reports"]
FIG = ROOT / CFG["paths"]["figures"]
INTERIM = ROOT / "data" / "interim"
for p in (REP, FIG, INTERIM):
    p.mkdir(parents=True, exist_ok=True)

RESULTS: dict = {}


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def r(x, n=4):
    return None if x is None else round(float(x), n)


# =========================================================================================
# Phase 1 -- data
# =========================================================================================
def phase1():
    cache = INTERIM / "loans_clean.pkl"
    raw_path = ROOT / CFG["data"]["raw_path"]
    if not raw_path.exists():
        raw_path = ROOT / CFG["data"]["fallback_path"]
    if cache.exists() and cache.stat().st_mtime > raw_path.stat().st_mtime:
        log("Loading cleaned data from cache")
        allr, status = pickle.load(open(cache, "rb"))
    else:
        log(f"Reading {raw_path.name} ({raw_path.stat().st_size / 1e6:,.0f} MB)")
        raw = dp.load_raw(str(raw_path))
        log(f"Raw rows: {len(raw):,}")
        raw = dp.clean(raw)
        status = dp.status_counts_by_year(raw, CFG["data"]["term_months"])
        allr = dp.add_default_flag(raw)
        allr = allr[allr["term"] == CFG["data"]["term_months"]]
        pickle.dump((allr, status), open(cache, "wb"))
    status.to_csv(REP / "status_by_year_36m.csv")
    RESULTS["data_source"] = raw_path.name

    df = dp.scope(allr, CFG["data"]["term_months"], CFG["data"]["issue_years"])
    train, oot = dp.oot_split(df, CFG["data"]["train_years"], CFG["data"]["oot_years"])

    # --- EDA tables --------------------------------------------------------------------
    by_year = df.groupby("issue_year")["bad"].agg(n="size", bads="sum", bad_rate="mean")
    by_grade = df.groupby("grade")["bad"].agg(n="size", bads="sum", bad_rate="mean")
    by_year.to_csv(REP / "bad_rate_by_year.csv")
    by_grade.to_csv(REP / "bad_rate_by_grade.csv")
    feats = [c for c in dp.NUMERIC_FEATURES + dp.CATEGORICAL_FEATURES if c in df]
    miss = df[feats].isna().mean().sort_values(ascending=False).rename("missing_share")
    miss.to_csv(REP / "missing_values.csv")
    plots.bar(by_year.index, by_year["bad_rate"], FIG / "bad_rate_by_year.png",
              "Bad rate by issue year (36-month, resolved loans)", "Bad rate")
    plots.bar(by_grade.index, by_grade["bad_rate"], FIG / "bad_rate_by_grade.png",
              "Bad rate by Lending Club grade", "Bad rate")

    RESULTS["data"] = {
        "n_loans": int(len(df)), "bad_rate": r(df["bad"].mean()),
        "train_years": CFG["data"]["train_years"], "oot_years": CFG["data"]["oot_years"],
        "n_train": int(len(train)), "n_oot": int(len(oot)),
        "bad_rate_train": r(train["bad"].mean()), "bad_rate_oot": r(oot["bad"].mean()),
        "bad_rate_by_year": {int(k): r(v) for k, v in by_year["bad_rate"].items()},
        "bad_rate_by_grade": {k: r(v) for k, v in by_grade["bad_rate"].items()},
    }
    log(f"Scope: {len(df):,} loans | train {len(train):,} | OOT {len(oot):,}")
    print(status.round(4).to_string())
    print(by_year.round(4).to_string())
    print(by_grade.round(4).to_string())
    print(miss[miss > 0].round(4).to_string())
    return train, oot


# =========================================================================================
# Phase 2 -- PD scorecard
# =========================================================================================
MAIN_NUMERIC = [f for f in dp.NUMERIC_FEATURES if f not in dp.LC_PRICING_FEATURES]
MAIN_CATEG = [f for f in dp.CATEGORICAL_FEATURES if f not in dp.LC_PRICING_FEATURES]


def build_scorecard(train, numeric, categ, tag):
    pc = CFG["pd_model"]
    # Leakage guard: no post-origination field may enter the PD model.
    assert not (set(numeric + categ) & dp.LEAKAGE_COLS), "leakage column in PD features"
    binner = WoEBinner(pc["fine_bins"], pc["min_bin_share"]).fit(
        train, train["bad"], numeric, categ)
    iv = binner.iv_table()
    keep, flagged = pm.select_by_iv(iv, pc["iv_min"], pc["iv_suspicious"])
    Xw = binner.transform(train, keep)
    kept_c, drop_c = pm.drop_correlated(Xw, keep, pc["corr_max"])
    kept_v, drop_v = pm.drop_high_vif(Xw[kept_c], pc["vif_max"])
    final, res, drop_s = pm.stepwise_sign_pvalue(Xw[kept_v], train["bad"], pc["pvalue_max"])
    selection = {"iv_below_min": iv.loc[iv["iv_coarse"] < pc["iv_min"], "feature"].tolist(),
                 "iv_suspicious": flagged, "dropped_correlation": drop_c,
                 "dropped_vif": drop_v, "dropped_sign_or_pvalue": drop_s, "final": final}
    log(f"[{tag}] final features ({len(final)}): {final}")
    return binner, iv, res, final, selection


def phase2(train, oot):
    sc = CFG["scorecard"]
    factor, offset = pm.scaling_params(sc["pdo"], sc["base_score"], sc["base_odds"])

    binner, iv, res, final, selection = build_scorecard(train, MAIN_NUMERIC, MAIN_CATEG, "main")
    iv.to_csv(REP / "iv_table.csv", index=False)
    binner.bin_table().to_csv(REP / "woe_bins_all.csv", index=False)
    coefs = pm.coef_table(res)
    coefs.to_csv(REP / "coefficients_statsmodels.csv", index=False)
    print(iv.to_string())
    print(json.dumps(selection, indent=1, default=str))
    print(res.summary2().tables[1].round(4).to_string())

    # Scikit-learn refit: same unpenalised model -> coefficients should match Statsmodels.
    Xtr = binner.transform(train, final)
    sk = pm.fit_sklearn(Xtr, train["bad"], SEED)
    max_diff = float(np.max(np.abs(np.r_[sk.intercept_, sk.coef_[0]] - res.params.values)))
    log(f"Max |coef difference| statsmodels vs sklearn: {max_diff:.2e}")

    coef_map = dict(zip(final, sk.coef_[0]))
    pts = pm.points_table(binner, final, float(sk.intercept_[0]), coef_map, factor, offset)
    pts.to_csv(REP / "scorecard_points.csv", index=False)
    print(pts[["feature", "bin", "woe", "points"]].to_string())

    # Manual binning demonstration: analyst-chosen revolving-utilisation cut points
    # (round business thresholds) vs the automatic monotonic merge.
    manual = WoEBinner(min_share=CFG["pd_model"]["min_bin_share"],
                       manual_edges={"revol_util": [20, 40, 60, 80]}).fit(
        train, train["bad"], ["revol_util"], [])
    manual.specs["revol_util"].table.to_csv(REP / "manual_binning_revol_util.csv")
    binner.specs["revol_util"].table.to_csv(REP / "auto_binning_revol_util.csv")

    # Worked example: score the first OOT applicant by hand.
    ex = oot.iloc[[0]]
    rows = []
    for f in final:
        lab = binner.specs[f].labels(ex[f]).iloc[0]
        p = pts[(pts.feature == f) & (pts.bin == lab)]["points"].iloc[0]
        rows.append({"feature": f, "value": ex[f].iloc[0], "bin": lab, "points": int(p)})
    worked = pd.DataFrame(rows)
    worked.to_csv(REP / "worked_example.csv", index=False)
    ex_score = int(worked["points"].sum())
    print(worked.to_string(), "\nTOTAL", ex_score, "-> PD", pm.score_to_pd(ex_score, factor, offset))

    # Comparison scorecard INCLUDING Lending Club's own pricing (grade, sub_grade, int_rate).
    b2, iv2, res2, final2, sel2 = build_scorecard(
        train, MAIN_NUMERIC + ["int_rate"], MAIN_CATEG + ["grade", "sub_grade"], "with_LC_grade")
    iv2.to_csv(REP / "iv_table_with_grade.csv", index=False)
    p2_tr = res2.predict(sm_const(b2.transform(train, final2)))
    p2_oot = res2.predict(sm_const(b2.transform(oot, final2)))

    RESULTS["pd_scorecard"] = {
        "scaling": {"pdo": sc["pdo"], "base_score": sc["base_score"], "base_odds": sc["base_odds"],
                    "factor": r(factor), "offset": r(offset)},
        "n_candidates": len(MAIN_NUMERIC) + len(MAIN_CATEG),
        "n_features": len(final), "features": final,
        "top5_iv": iv.head(5)[["feature", "iv_coarse"]].round(4).to_dict("records"),
        "selection": selection, "statsmodels_sklearn_max_coef_diff": max_diff,
        "pseudo_r2_mcfadden": r(res.prsquared), "worked_example_score": ex_score,
        "worked_example_pd": r(pm.score_to_pd(ex_score, factor, offset)),
    }
    RESULTS["pd_with_lc_grade"] = {"features": final2, "n_features": len(final2),
                                   "train": val.summarise(train["bad"], p2_tr),
                                   "oot": val.summarise(oot["bad"], p2_oot)}
    return {"binner": binner, "final": final, "pts": pts, "factor": factor, "offset": offset,
            "sk": sk, "res": res, "p2": (p2_tr, p2_oot)}


def sm_const(X):
    import statsmodels.api as sm
    return sm.add_constant(X, has_constant="add")


# =========================================================================================
# Phase 3 -- PD validation
# =========================================================================================
def phase3(train, oot, m):
    b, final, pts, factor, offset = m["binner"], m["final"], m["pts"], m["factor"], m["offset"]
    for d in (train, oot):
        d["score"] = pm.score_from_points(b, d, pts)
        d["pd"] = pm.score_to_pd(d["score"], factor, offset)

    out = {}
    for name, d in (("train", train), ("oot", oot)):
        s = val.summarise(d["bad"], d["pd"])
        s["hosmer_lemeshow"] = val.hosmer_lemeshow(d["bad"], d["pd"])
        cal = val.calibration_table(d["bad"], d["pd"])
        cal.to_csv(REP / f"calibration_{name}.csv", index=False)
        gains = val.gains_table(d["bad"], d["score"])
        gains.to_csv(REP / f"gains_{name}.csv", index=False)
        s["rank_order_monotonic"] = val.rank_order_ok(gains)
        s["calibration_bands_binom_p_lt_0.05"] = int((cal["binom_p"] < 0.05).sum())
        out[name] = s
        print(name, json.dumps({k: v for k, v in s.items()}, indent=1, default=float))
        print(cal.round(4).to_string())
        print(gains.round(4).to_string())

    psi_val, psi_t = val.psi(train["score"], oot["score"], CFG["validation"]["psi_bands"])
    psi_t.to_csv(REP / "psi_score.csv")
    csi = val.csi_by_feature(b, final, train, oot)
    csi.to_csv(REP / "csi_features.csv", index=False)
    log(f"PSI score train->OOT: {psi_val:.4f}")
    print(csi.round(4).to_string())

    # Recalibration check: a one-parameter intercept shift that would match the OOT bad rate
    # (diagnostic only -- in practice you'd recalibrate on recent data, not the test set).
    from scipy.optimize import brentq
    lo = np.log(oot["pd"] / (1 - oot["pd"]))
    shift = brentq(lambda c: (1 / (1 + np.exp(-(lo + c)))).mean() - oot["bad"].mean(), -3, 3)

    # --- Benchmark: LightGBM on raw origination features (same no-LC-pricing restriction) ---
    import lightgbm as lgb
    raw_feats = MAIN_NUMERIC + MAIN_CATEG

    def prep(d):
        X = d[raw_feats].copy()
        for c in MAIN_CATEG:
            X[c] = X[c].astype("category")
        return X
    Xtr, Xoot = prep(train), prep(oot)
    for c in MAIN_CATEG:
        Xoot[c] = pd.Categorical(Xoot[c], categories=Xtr[c].cat.categories)
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=SEED, verbose=-1)
    gbm.fit(Xtr, train["bad"])
    g_tr, g_oot = gbm.predict_proba(Xtr)[:, 1], gbm.predict_proba(Xoot)[:, 1]
    bench = {"train": val.summarise(train["bad"], g_tr), "oot": val.summarise(oot["bad"], g_oot)}

    # --- Figures ------------------------------------------------------------------------
    plots.roc({"Scorecard - train": (train["bad"], train["pd"], out["train"]["gini"]),
               "Scorecard - OOT": (oot["bad"], oot["pd"], out["oot"]["gini"]),
               "LightGBM - OOT": (oot["bad"], g_oot, bench["oot"]["gini"]),
               "Scorecard + LC grade - OOT": (oot["bad"], m["p2"][1],
                                              RESULTS["pd_with_lc_grade"]["oot"]["gini"])},
              FIG / "roc.png")
    plots.ks_plot(oot["bad"], oot["score"], out["oot"]["ks"], FIG / "ks.png")
    tr_lab = f"Train ({min(CFG['data']['train_years'])}-{max(CFG['data']['train_years'])})"
    oot_lab = f"Out-of-time ({', '.join(map(str, CFG['data']['oot_years']))})"
    plots.calibration({tr_lab: pd.read_csv(REP / "calibration_train.csv"),
                       oot_lab: pd.read_csv(REP / "calibration_oot.csv")},
                      FIG / "calibration.png")
    plots.psi_bars(psi_t, psi_val, FIG / "psi.png", tr_lab, oot_lab)
    plots.bar(np.arange(1, 11), pd.read_csv(REP / "gains_oot.csv")["bad_rate"],
              FIG / "gains_oot.png", "OOT bad rate by score decile (1 = lowest score)", "Bad rate")

    RESULTS["pd_validation"] = {
        "train": out["train"], "oot": out["oot"], "psi_score": psi_val,
        "psi_status": "stable" if psi_val < 0.1 else ("monitor" if psi_val < 0.25 else "shift"),
        "csi": csi.set_index("feature")["csi"].round(4).to_dict(),
        "oot_intercept_shift_to_match_bad_rate": shift,
        "lightgbm_benchmark": bench,
    }
    return train, oot


# =========================================================================================
# Phase 4 -- LGD and EAD
# =========================================================================================
def phase4(train, oot):
    tr_def = lgd_targets(ead_targets(train[train["bad"] == 1]))
    oot_def = lgd_targets(ead_targets(oot[oot["bad"] == 1]))
    log(f"Defaults with exposure: train {len(tr_def):,} | OOT {len(oot_def):,}")

    # ---------------- LGD ----------------
    lgd = TwoStageLGD().fit(tr_def)
    seg = SegmentAverageLGD().fit(tr_def)
    pd.DataFrame({"stage1_coef": lgd.stage1.params, "stage1_p": lgd.stage1.pvalues,
                  "stage2_coef": lgd.stage2.params, "stage2_p": lgd.stage2.pvalues}
                 ).to_csv(REP / "lgd_coefficients.csv")
    lgd_res = {}
    for name, d in (("train", tr_def), ("oot", oot_def)):
        d["lgd_pred"] = lgd.predict(d)
        d["lgd_seg"] = seg.predict(d)
        d["lgd_overall"] = tr_def["lgd_actual"].mean()
        lgd_res[name] = {"two_stage": regression_metrics(d["lgd_actual"], d["lgd_pred"]),
                         "segment_avg": regression_metrics(d["lgd_actual"], d["lgd_seg"]),
                         "overall_mean": regression_metrics(d["lgd_actual"], d["lgd_overall"])}
    from sklearn.metrics import roc_auc_score
    s1_auc = roc_auc_score(oot_def["any_recovery"], lgd.predict_components(oot_def)["p_recovery"])
    by_group(oot_def, "lgd_actual", "lgd_pred", "grade").to_csv(REP / "lgd_by_grade_oot.csv", index=False)
    oot_def["lgd_decile"] = pd.qcut(oot_def["lgd_pred"].rank(method="first"), 10, labels=False) + 1
    by_group(oot_def, "lgd_actual", "lgd_pred", "lgd_decile").to_csv(REP / "lgd_by_decile_oot.csv", index=False)
    # Histogram counts (5-point LGD bins) for the web dashboard.
    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(tr_def["lgd_actual"], bins=edges)
    pd.DataFrame({"lgd_from": edges[:-1].round(2), "lgd_to": edges[1:].round(2),
                  "loans": counts}).to_csv(REP / "lgd_hist_train.csv", index=False)
    plots.hist(tr_def["lgd_actual"], FIG / "lgd_hist.png",
               "Realised LGD on defaulted loans (train)", "LGD = 1 - recoveries / principal outstanding")
    lg = by_group(oot_def, "lgd_actual", "lgd_pred", "grade")
    plots.grouped_bars(lg["grade"], {"Actual": lg["actual_mean"], "Predicted": lg["pred_mean"]},
                       FIG / "lgd_by_grade.png", "LGD by grade, out-of-time defaults", "Mean LGD")
    print(json.dumps(lgd_res, indent=1))
    print(lg.round(4).to_string())

    # ---------------- EAD ----------------
    ead = EADRatioModel().fit(tr_def)
    ead_b = GradeAverageEAD().fit(tr_def)
    pd.DataFrame({"coef": ead.res.params, "p": ead.res.pvalues}).to_csv(REP / "ead_coefficients.csv")
    ead_res = {}
    for name, d in (("train", tr_def), ("oot", oot_def)):
        d["ead_ratio_pred"] = ead.predict_ratio(d)
        d["ead_pred"] = ead.predict_ead(d)
        d["ead_grade"] = ead_b.predict_ead(d)
        ead_res[name] = {
            "ratio_model": regression_metrics(d["ead_ratio"], d["ead_ratio_pred"]),
            "ratio_grade_avg": regression_metrics(d["ead_ratio"], ead_b.predict_ratio(d)),
            "ead_model": regression_metrics(d["ead_actual"], d["ead_pred"]),
            "ead_grade_avg": regression_metrics(d["ead_actual"], d["ead_grade"]),
        }
    eg = by_group(oot_def, "ead_ratio", "ead_ratio_pred", "grade")
    eg.to_csv(REP / "ead_by_grade_oot.csv", index=False)
    oot_def["ead_decile"] = pd.qcut(oot_def["ead_pred"].rank(method="first"), 10, labels=False) + 1
    by_group(oot_def, "ead_actual", "ead_pred", "ead_decile").to_csv(REP / "ead_by_decile_oot.csv", index=False)
    plots.hist(tr_def["ead_ratio"], FIG / "ead_ratio_hist.png",
               "Share of loan still outstanding at default (train)", "EAD ratio")
    print(json.dumps(ead_res, indent=1))
    print(eg.round(4).to_string())

    RESULTS["lgd"] = {"definition": "LGD = 1 - recoveries / (funded_amnt - total_rec_prncp), clipped [0,1]",
                      "share_zero_recovery_train": r(1 - tr_def["any_recovery"].mean()),
                      "stage1_auc_oot": s1_auc, **lgd_res}
    RESULTS["ead"] = {"definition": "EAD ratio = (funded_amnt - total_rec_prncp) / funded_amnt",
                      **ead_res}
    return lgd, ead


# =========================================================================================
# Phase 5 -- Expected Loss
# =========================================================================================
def phase5(train, oot, lgd, ead):
    ec = CFG["el"]
    for d in (train, oot):
        d["lgd_pred"] = lgd.predict(d)
        d["ead_pred"] = ead.predict_ead(d)
        d["el"] = el.loan_level_el(d["pd"], d["lgd_pred"], d["ead_pred"])
        d["realised_loss"] = el.realised_loss(d)
        d["score_band"] = pd.qcut(d["score"].rank(method="first"), 10, labels=False) + 1

    out = {}
    for name, d in (("train", train), ("oot", oot)):
        s = el.portfolio_summary(d)
        s["realised_loss"] = float(d["realised_loss"].sum())
        s["realised_loss_pct"] = s["realised_loss"] / s["exposure"]
        s["el_to_realised_ratio"] = s["total_el"] / s["realised_loss"]
        # Where does any gap come from? Compare each component with what actually happened.
        dd = d[d["bad"] == 1]
        ead_act = (dd["funded_amnt"] - dd["total_rec_prncp"]).clip(lower=0)
        lgd_act = 1 - (dd["recoveries"] / ead_act.replace(0, np.nan)).clip(0, 1)
        s["decomposition"] = {
            "pd_ratio_pred_over_actual": float(d["pd"].mean() / d["bad"].mean()),
            "ead_ratio_pred_over_actual_defaults": float(dd["ead_pred"].sum() / ead_act.sum()),
            "lgd_ratio_pred_over_actual_defaults": float(
                np.average(dd["lgd_pred"], weights=ead_act) / np.average(lgd_act.fillna(0), weights=ead_act)),
        }
        out[name] = s
        el.el_by(d, "grade").to_csv(REP / f"el_by_grade_{name}.csv", index=False)
        el.el_by(d, "score_band").to_csv(REP / f"el_by_scoreband_{name}.csv", index=False)
    print(json.dumps(out, indent=1))
    g = el.el_by(oot, "grade")
    print(g.round(4).to_string())
    plots.grouped_bars(g["grade"], {"Predicted EL": g["el_pct"], "Realised loss": g["realised_pct"]},
                       FIG / "el_by_grade.png", "Expected vs realised loss by grade (OOT 2015)",
                       "% of funded amount")

    stress = el.stress_test(oot, ec["pd_stress_multiplier"], ec["lgd_stress_add"])
    stress.to_csv(REP / "stress_test_oot.csv", index=False)
    print(stress.round(4).to_string())

    log("Monte Carlo (independent and correlated defaults)...")
    mc = [el.monte_carlo(oot, ec["mc_simulations"], SEED, rho=0.0),
          el.monte_carlo(oot, ec["mc_simulations"], SEED, rho=0.10)]
    plots.loss_distribution(mc, FIG / "loss_distribution.png")
    mc_out = [{k: (r(v, 2) if isinstance(v, float) else v) for k, v in x.items() if k != "losses"}
              for x in mc]
    print(json.dumps(mc_out, indent=1))

    curves = el.cohort_default_curves(pd.concat([train, oot]))
    curves.to_csv(REP / "cohort_default_curves.csv", index=False)
    plots.cohort_curves(curves, FIG / "cohort_default_curves.png")

    RESULTS["expected_loss"] = {"currency": "USD (Lending Club is a US lender)", **out,
                                "stress_test_oot": stress.round(4).to_dict("records"),
                                "monte_carlo_oot": mc_out}


# =========================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", type=int, default=5)
    a = ap.parse_args()
    set_seeds()
    t0 = time.time()
    train, oot = phase1()
    if a.until >= 2:
        m = phase2(train, oot)
    if a.until >= 3:
        train, oot = phase3(train, oot, m)
    if a.until >= 4:
        lgd, ead = phase4(train, oot)
    if a.until >= 5:
        phase5(train, oot, lgd, ead)
    RESULTS["runtime_seconds"] = round(time.time() - t0, 1)
    with open(REP / "results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"Done in {RESULTS['runtime_seconds']}s -> reports/results.json")


if __name__ == "__main__":
    main()
