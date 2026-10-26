"""Light checks on the core maths. Run: pytest -q"""
import numpy as np
import pandas as pd

from src import data_prep as dp
from src import pd_model as pm
from src import validation as val
from src.woe_binning import WoEBinner, is_monotonic, woe_iv_table

rng = np.random.default_rng(42)


def _toy(n=20000):
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 0.8 * x)))).astype(int)
    return pd.DataFrame({"x": x, "c": rng.choice(list("abcd"), n)}), pd.Series(y)


def test_woe_sign_convention():
    t = woe_iv_table(pd.Series(["lo"] * 100 + ["hi"] * 100), pd.Series([0] * 95 + [1] * 5 + [0] * 70 + [1] * 30))
    assert t.loc["lo", "woe"] > 0 > t.loc["hi", "woe"]     # safer bin -> positive WoE


def test_coarse_bins_monotonic_and_large_enough():
    X, y = _toy()
    b = WoEBinner(fine_bins=20, min_share=0.05).fit(X, y, ["x"], ["c"])
    spec = b.specs["x"]
    assert is_monotonic(spec)
    assert (spec.table.drop(index="Missing", errors="ignore")["share"] >= 0.05).all()


def test_no_leakage_columns_in_pd_features():
    pd_feats = set(dp.NUMERIC_FEATURES + dp.CATEGORICAL_FEATURES)
    assert not (pd_feats & dp.LEAKAGE_COLS)


def test_psi_zero_for_identical_and_positive_for_shift():
    a = rng.normal(0, 1, 50000)
    assert val.psi(a, a)[0] < 1e-9
    assert val.psi(a, a + 0.5)[0] > 0.1


def test_scaling_anchor():
    factor, offset = pm.scaling_params(20, 600, 50)
    assert np.isclose(offset + factor * np.log(50), 600)       # 600 points at 50:1
    assert np.isclose(factor * np.log(100) - factor * np.log(50), 20)  # PDO = 20
    assert np.isclose(pm.score_to_pd(600, factor, offset), 1 / 51)


def test_gini_ks_perfect_model():
    y = np.r_[np.zeros(100), np.ones(100)]
    p = np.r_[np.linspace(0, .4, 100), np.linspace(.6, 1, 100)]
    auc, gini = val.auc_gini(y, p)
    assert gini == 1 and val.ks_stat(y, p) == 1
