"""Build and execute the four walkthrough notebooks from the outputs of run_all.py.

The notebooks are a readable tour of the results (tables + figures + short explanations).
All computation lives in src/ and run_all.py; the notebooks only load what it saved.
Usage (from the repo root, after `python run_all.py`):  python notebooks/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
SETUP = """import json, pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)
R = '../reports/'; F = R + 'figures/'
res = json.load(open(R + 'results.json'))"""

NB = {
    "01_eda.ipynb": [
        ("md", "# 01 · Data and EDA\nScope: individual 36-month Lending Club loans with a resolved "
               "outcome. Train = 2013–14 issues, out-of-time (OOT) = 2015 issues."),
        ("code", SETUP),
        ("md", "## Why 2013–2015? Outcome counts by issue year (36-month loans)\n2016+ loans were "
               "still mostly unresolved at the data cut-off (Dec 2018), and the resolved ones are "
               "biased towards early defaults."),
        ("code", "pd.read_csv(R + 'status_by_year_36m.csv')"),
        ("code", "pd.Series({k: v for k, v in res['data'].items() if not k.startswith('bad_rate_by')}, dtype=object)"),
        ("code", "display(pd.read_csv(R + 'bad_rate_by_year.csv')); display(Image(F + 'bad_rate_by_year.png'))"),
        ("code", "display(pd.read_csv(R + 'bad_rate_by_grade.csv')); display(Image(F + 'bad_rate_by_grade.png'))"),
        ("md", "## Missing values (candidate features)"),
        ("code", "pd.read_csv(R + 'missing_values.csv').query('missing_share > 0')"),
    ],
    "02_pd_scorecard.ipynb": [
        ("md", "# 02 · PD scorecard and validation\nWoE/IV binning → feature selection → "
               "Statsmodels logit → points scaling (PDO 20, 600 points at 50:1)."),
        ("code", SETUP),
        ("md", "## Information Value ranking (train)\nIV < 0.02 dropped; IV > 0.5 would be flagged as "
               "possible leakage."),
        ("code", "pd.read_csv(R + 'iv_table.csv')"),
        ("code", "json.dumps(res['pd_scorecard']['selection'], indent=1)"),
        ("md", "## Manual vs automatic binning (revolving utilisation)\nManual: round business "
               "cut-points. Automatic: 20 quantile bins merged until WoE is monotonic and each bin "
               "holds ≥ 5% of loans."),
        ("code", "display(pd.read_csv(R + 'manual_binning_revol_util.csv')); "
                 "display(pd.read_csv(R + 'auto_binning_revol_util.csv'))"),
        ("md", "## Statsmodels coefficient table\nAll coefficients should be negative "
               "(higher WoE = safer = lower PD)."),
        ("code", "pd.read_csv(R + 'coefficients_statsmodels.csv')"),
        ("md", "## Points table and a worked example"),
        ("code", "pd.read_csv(R + 'scorecard_points.csv')[['feature','bin','woe','coef','points']]"),
        ("code", "w = pd.read_csv(R + 'worked_example.csv'); display(w); "
                 "print('Total score:', w.points.sum(), '-> PD', res['pd_scorecard']['worked_example_pd'])"),
        ("md", "## Validation: train vs out-of-time"),
        ("code", "v = res['pd_validation']; pd.DataFrame({k: {m: v[k][m] for m in "
                 "['n','bad_rate','mean_pd','auc','gini','ks','brier']} for k in ['train','oot']})"),
        ("code", "print('PSI (score):', round(v['psi_score'], 4), v['psi_status']); "
                 "pd.read_csv(R + 'csi_features.csv')"),
        ("code", "for f in ['roc','ks','calibration','psi','gains_oot']: display(Image(F + f + '.png'))"),
        ("code", "display(pd.read_csv(R + 'calibration_oot.csv')); display(pd.read_csv(R + 'gains_oot.csv'))"),
        ("md", "## Benchmarks: LightGBM, and a scorecard that includes Lending Club's own grade"),
        ("code", "pd.DataFrame({'scorecard': v['oot'], 'lightgbm': v['lightgbm_benchmark']['oot'], "
                 "'scorecard_with_LC_grade': res['pd_with_lc_grade']['oot']}).loc[['auc','gini','ks','brier']]"),
    ],
    "03_lgd_ead.ipynb": [
        ("md", "# 03 · LGD and EAD\nDefaulted loans only. LGD = 1 − recoveries / principal "
               "outstanding at default. EAD ratio = principal outstanding / funded amount."),
        ("code", SETUP),
        ("code", "display(Image(F + 'lgd_hist.png')); print('Share with zero recovery:', "
                 "res['lgd']['share_zero_recovery_train'])"),
        ("code", "pd.DataFrame({(s, m): res['lgd'][s][m] for s in ['train','oot'] for m in "
                 "['two_stage','segment_avg','overall_mean']}).T[['n','mean_actual','mean_pred','rmse','mae','r2']]"),
        ("code", "display(pd.read_csv(R + 'lgd_by_grade_oot.csv')); display(Image(F + 'lgd_by_grade.png'))"),
        ("code", "pd.read_csv(R + 'lgd_coefficients.csv')"),
        ("code", "display(Image(F + 'ead_ratio_hist.png')); pd.DataFrame({(s, m): res['ead'][s][m] "
                 "for s in ['train','oot'] for m in res['ead']['oot']}).T[['n','mean_actual','mean_pred','rmse','mae','r2']]"),
        ("code", "pd.read_csv(R + 'ead_by_grade_oot.csv')"),
    ],
    "04_expected_loss.ipynb": [
        ("md", "# 04 · Expected Loss\nEL = PD × LGD × EAD per loan (lifetime horizon for 36-month loans), "
               "back-tested against realised loss on the 2015 out-of-time cohort."),
        ("code", SETUP),
        ("code", "pd.DataFrame({k: {m: x for m, x in res['expected_loss'][k].items() if m != 'decomposition'} for k in ['train','oot']})"),
        ("code", "pd.DataFrame({k: res['expected_loss'][k]['decomposition'] for k in ['train','oot']})"),
        ("code", "display(pd.read_csv(R + 'el_by_grade_oot.csv')); display(Image(F + 'el_by_grade.png'))"),
        ("code", "pd.read_csv(R + 'el_by_scoreband_oot.csv')"),
        ("md", "## Stress test and Monte Carlo loss distribution"),
        ("code", "display(pd.read_csv(R + 'stress_test_oot.csv')); "
                 "display(pd.DataFrame(res['expected_loss']['monte_carlo_oot']))"),
        ("code", "display(Image(F + 'loss_distribution.png'))"),
        ("md", "## Cohort default curves"),
        ("code", "display(Image(F + 'cohort_default_curves.png'))"),
    ],
}

for name, cells in NB.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s)
                for t, s in cells]
    NotebookClient(nb, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, HERE / name)
    print("built", name)
