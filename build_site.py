"""Build the public dashboard (docs/index.html) from the outputs of run_all.py.

    python run_all.py        # writes reports/
    python build_site.py     # writes docs/index.html (served by GitHub Pages)

The page is one self-contained HTML file: every number is read from reports/ and embedded
as JSON, so the dashboard always matches results.json. d3 is the only external script.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REP = ROOT / "reports"
TEMPLATE = ROOT / "site" / "template.html"
OUT = ROOT / "docs" / "index.html"


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def collect() -> dict:
    r = json.load(open(REP / "results.json"))
    v, e = r["pd_validation"], r["expected_loss"]
    data = {}

    data["summary"] = [{
        "sample": s, "label": lab, "loans": v[s]["n"], "bad_rate": v[s]["bad_rate"],
        "mean_pd": v[s]["mean_pd"], "auc": v[s]["auc"], "gini": v[s]["gini"], "ks": v[s]["ks"],
        "brier": v[s]["brier"], "hl_p": v[s]["hosmer_lemeshow"]["p_value"],
        "exposure": e[s]["exposure"], "expected_loss": e[s]["total_el"],
        "el_pct": e[s]["el_pct_of_exposure"], "realised_loss": e[s]["realised_loss"],
        "realised_pct": e[s]["realised_loss_pct"], "el_to_realised": e[s]["el_to_realised_ratio"],
        "pd_ratio": e[s]["decomposition"]["pd_ratio_pred_over_actual"],
        "ead_ratio": e[s]["decomposition"]["ead_ratio_pred_over_actual_defaults"],
        "lgd_ratio": e[s]["decomposition"]["lgd_ratio_pred_over_actual_defaults"],
        "psi": v["psi_score"],
    } for s, lab in [("train", "Train 2013–14"), ("oot", "Out-of-time 2015")]]

    st = pd.read_csv(REP / "status_by_year_36m.csv")
    train_y, oot_y = r["data"]["train_years"], r["data"]["oot_years"]
    st["role"] = st["issue_year"].map(
        lambda y: "Train" if y in train_y else ("Out-of-time" if y in oot_y else "Excluded"))
    data["vintages"] = records(st)

    iv = pd.read_csv(REP / "iv_table.csv")
    sel, final = r["pd_scorecard"]["selection"], set(r["pd_scorecard"]["features"])

    def status(f):
        if f in final:
            return "In scorecard"
        if f in sel["iv_below_min"]:
            return "IV below 0.02"
        if f in sel["dropped_correlation"]:
            return "Correlated"
        if f in sel["dropped_sign_or_pvalue"]:
            return "Not significant"
        return "Dropped"
    iv["status"] = iv["feature"].map(status)
    data["iv"] = records(iv[["feature", "iv_coarse", "n_bins", "status"]].rename(columns={"iv_coarse": "iv"}))

    c = pd.read_csv(REP / "coefficients_statsmodels.csv")
    data["coefficients"] = records(c[c["feature"] != "const"][["feature", "coef", "std_err", "p_value"]])
    data["points"] = records(pd.read_csv(REP / "scorecard_points.csv")[
        ["feature", "bin", "woe", "points", "share_train", "bad_rate_train"]])

    cal = pd.concat([pd.read_csv(REP / "calibration_train.csv").assign(sample="train"),
                     pd.read_csv(REP / "calibration_oot.csv").assign(sample="oot")])
    data["calibration"] = records(cal[["sample", "band", "n", "bads", "pd_mean", "observed_rate", "binom_p"]])
    data["gains"] = records(pd.read_csv(REP / "gains_oot.csv")[
        ["band", "n", "bads", "score_min", "score_max", "bad_rate", "cum_bad_capture", "lift"]])
    p = pd.read_csv(REP / "psi_score.csv").reset_index(drop=True)
    p["band"] = p.index + 1
    data["psi"] = records(p[["band", "expected_pct", "actual_pct", "psi_band"]])
    data["csi"] = records(pd.read_csv(REP / "csi_features.csv")[["feature", "csi"]])

    data["benchmarks"] = [
        {"model": m, "sample": s, "gini": src[s]["gini"], "auc": src[s]["auc"], "ks": src[s]["ks"]}
        for m, src in [("Scorecard", v), ("LightGBM benchmark", v["lightgbm_benchmark"]),
                       ("Scorecard + LC sub-grade", r["pd_with_lc_grade"])]
        for s in ["train", "oot"]]

    models = []
    for comp, key, ms in [("LGD", "lgd", [("two_stage", "Two-stage model"),
                                          ("segment_avg", "Grade × purpose average"),
                                          ("overall_mean", "Overall mean")]),
                          ("EAD ($)", "ead", [("ead_model", "Ratio model"),
                                              ("ead_grade_avg", "Grade average")])]:
        for k, lab in ms:
            x = r[key]["oot"][k]
            models.append({"component": comp, "model": lab, **{m: x[m] for m in
                           ["mean_actual", "mean_pred", "rmse", "mae", "r2"]}})
    data["lgd_ead_models"] = models

    lg = pd.read_csv(REP / "lgd_by_grade_oot.csv").rename(columns={"actual_mean": "lgd_actual", "pred_mean": "lgd_pred"})
    eg = pd.read_csv(REP / "ead_by_grade_oot.csv").rename(columns={"actual_mean": "ead_ratio_actual", "pred_mean": "ead_ratio_pred"})
    data["lgd_ead_by_grade"] = records(lg.merge(eg[["grade", "ead_ratio_actual", "ead_ratio_pred"]], on="grade"))
    data["lgd_hist"] = records(pd.read_csv(REP / "lgd_hist_train.csv"))

    data["el_grade"] = records(pd.read_csv(REP / "el_by_grade_oot.csv")[
        ["grade", "n", "exposure", "el", "el_pct", "realised_loss", "realised_pct", "mean_pd", "share_of_el"]])
    data["el_band"] = records(pd.read_csv(REP / "el_by_scoreband_oot.csv")[
        ["score_band", "n", "exposure", "el", "el_pct", "realised_loss", "realised_pct", "mean_pd"]])
    data["stress"] = records(pd.read_csv(REP / "stress_test_oot.csv"))
    data["monte_carlo"] = [{
        "scenario": "Independent defaults" if x["rho"] == 0 else f"Correlated defaults (ρ = {x['rho']})",
        **{k: x[k] for k in ["rho", "n_sims", "analytic_el", "q99", "q999", "ul_99", "ul_999"]}}
        for x in e["monte_carlo_oot"]]
    data["cohorts"] = records(pd.read_csv(REP / "cohort_default_curves.csv"))
    data["meta"] = [{"n_loans": r["data"]["n_loans"], "source": r["data_source"],
                     "n_features": r["pd_scorecard"]["n_features"]}]
    return data


def main():
    data = collect()
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/{}", blob)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    (OUT.parent / ".nojekyll").touch()   # serve the file as-is on GitHub Pages
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:,.0f} KB, {len(data)} tables)")


if __name__ == "__main__":
    main()
