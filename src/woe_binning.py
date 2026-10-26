"""Weight of Evidence (WoE) / Information Value (IV) binning -- implemented by hand.

Why WoE?
* Turns every feature (numeric or categorical) into one number per bin on the log-odds scale,
  which is exactly the scale logistic regression works on.
* Handles missing values and outliers naturally (they just get their own bin / fall in an end bin).
* Monotonic bins make the scorecard explainable: "higher utilisation -> fewer points".

Conventions (Section 10 of the transfer package):
    WoE_bin = ln( %good_bin / %bad_bin )     -> positive WoE = safer than average
    IV      = sum_bins (%good - %bad) * WoE
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SMOOTH = 0.5          # added to good/bad counts so empty cells don't give ln(0)
MISSING = "Missing"


# ----------------------------------------------------------------------------------------
# Core WoE / IV maths
# ----------------------------------------------------------------------------------------
def woe_iv_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Given a bin label per row and the 0/1 bad flag, return the WoE/IV table."""
    t = pd.crosstab(bins, y)
    t = t.reindex(columns=[0, 1], fill_value=0)
    t.columns = ["good", "bad"]
    tot_g, tot_b = t["good"].sum(), t["bad"].sum()
    t["n"] = t["good"] + t["bad"]
    t["share"] = t["n"] / t["n"].sum()
    t["bad_rate"] = t["bad"] / t["n"]
    t["pct_good"] = (t["good"] + SMOOTH) / (tot_g + SMOOTH * len(t))
    t["pct_bad"] = (t["bad"] + SMOOTH) / (tot_b + SMOOTH * len(t))
    t["woe"] = np.log(t["pct_good"] / t["pct_bad"])
    t["iv_bin"] = (t["pct_good"] - t["pct_bad"]) * t["woe"]
    return t


def information_value(bins: pd.Series, y: pd.Series) -> float:
    return float(woe_iv_table(bins, y)["iv_bin"].sum())


# ----------------------------------------------------------------------------------------
# Bin specifications
# ----------------------------------------------------------------------------------------
@dataclass
class NumericBins:
    """Right-closed intervals defined by `edges` (first = -inf, last = +inf) + a Missing bin."""
    name: str
    edges: list[float]
    woe: dict = field(default_factory=dict)        # label -> WoE
    table: pd.DataFrame | None = None

    def labels(self, x: pd.Series) -> pd.Series:
        idx = pd.cut(x, bins=self.edges, right=True, labels=False)
        lab = idx.map(lambda i: self._label(int(i)) if pd.notna(i) else MISSING)
        return lab.astype(object)

    def _label(self, i: int) -> str:
        lo, hi = self.edges[i], self.edges[i + 1]
        fmt = lambda v: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"   # 26,000 not 2.6e+04
        if np.isneginf(lo):
            return f"<= {fmt(hi)}"
        if np.isposinf(hi):
            return f"> {fmt(lo)}"
        return f"({fmt(lo)}, {fmt(hi)}]"

    def ordered_labels(self) -> list[str]:
        return [self._label(i) for i in range(len(self.edges) - 1)] + [MISSING]


@dataclass
class CategoricalBins:
    """Maps raw category -> group label. Unseen categories fall back to `default_group`."""
    name: str
    mapping: dict
    default_group: str
    woe: dict = field(default_factory=dict)
    table: pd.DataFrame | None = None

    def labels(self, x: pd.Series) -> pd.Series:
        x = x.astype(object).where(x.notna(), MISSING)
        return x.map(self.mapping).fillna(self.default_group).astype(object)

    def ordered_labels(self) -> list[str]:
        return list(dict.fromkeys(self.mapping.values()))


# ----------------------------------------------------------------------------------------
# Fine classing
# ----------------------------------------------------------------------------------------
def fine_edges(x: pd.Series, n_bins: int) -> list[float]:
    """Quantile cut points (duplicates removed -- discrete variables give fewer bins)."""
    qs = np.unique(np.nanquantile(x.dropna(), np.linspace(0, 1, n_bins + 1)[1:-1]))
    return [-np.inf, *qs.tolist(), np.inf]


# ----------------------------------------------------------------------------------------
# Coarse classing: merge adjacent bins until WoE is monotonic and every bin is big enough
# ----------------------------------------------------------------------------------------
def _counts_by_interval(x, y, edges):
    idx = pd.cut(x, bins=edges, right=True, labels=False)
    mask = idx.notna()
    g = pd.Series(1 - y[mask].values).groupby(idx[mask].values).sum()
    b = pd.Series(y[mask].values).groupby(idx[mask].values).sum()
    k = len(edges) - 1
    g = g.reindex(range(k), fill_value=0).values.astype(float)
    b = b.reindex(range(k), fill_value=0).values.astype(float)
    return g, b


def monotonic_merge(x: pd.Series, y: pd.Series, edges: list[float], min_share: float,
                    direction: int | None = None) -> tuple[list[float], int]:
    """Merge adjacent intervals so the bad rate is monotonic and each bin has >= min_share.

    direction: +1 = bad rate rises with x, -1 = falls with x, None = infer from the fine bins.
    Returns the new edges and the direction used.
    """
    edges = list(edges)
    g, b = _counts_by_interval(x, y, edges)
    # Drop empty intervals first.
    keep = [0] + [i + 1 for i in range(len(g)) if g[i] + b[i] > 0]
    edges = [edges[i] for i in sorted(set(keep))]
    if not np.isposinf(edges[-1]):
        edges[-1] = np.inf
    g, b = _counts_by_interval(x, y, edges)

    if direction is None:
        rate = b / np.maximum(g + b, 1)
        w = g + b
        # Weighted correlation between bin order and bad rate decides the trend.
        order = np.arange(len(rate))
        c = np.cov(order, rate, aweights=w)[0, 1] if len(rate) > 1 else 0
        direction = 1 if c >= 0 else -1

    def merge(i):  # merge interval i with i+1 by removing the edge between them
        del edges[i + 1]

    # 1) Pool adjacent violators until monotonic.
    while True:
        g, b = _counts_by_interval(x, y, edges)
        rate = b / np.maximum(g + b, 1)
        viol = [i for i in range(len(rate) - 1) if direction * (rate[i + 1] - rate[i]) <= 0]
        if not viol:
            break
        merge(viol[0])

    # 2) Merge undersized bins into the neighbour with the closest bad rate.
    total = len(x.dropna())
    while len(edges) > 2:
        g, b = _counts_by_interval(x, y, edges)
        n = g + b
        share = n / total
        i = int(np.argmin(share))
        if share[i] >= min_share:
            break
        rate = b / np.maximum(n, 1)
        if i == 0:
            merge(0)
        elif i == len(n) - 1:
            merge(i - 1)
        else:
            left, right = abs(rate[i] - rate[i - 1]), abs(rate[i] - rate[i + 1])
            merge(i - 1 if left <= right else i)
    return edges, direction


def coarse_categorical(x: pd.Series, y: pd.Series, min_share: float) -> CategoricalBins:
    """Sort categories by bad rate, then merge neighbours until each group has >= min_share."""
    x = x.astype(object).where(x.notna(), MISSING)
    t = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["sum", "count"])
    t["rate"] = t["sum"] / t["count"]
    t = t.sort_values("rate")
    groups = [[c] for c in t.index]
    counts = t["count"].tolist()
    bads = t["sum"].tolist()
    total = sum(counts)
    while len(groups) > 1:
        shares = [c / total for c in counts]
        i = int(np.argmin(shares))
        if shares[i] >= min_share:
            break
        rates = [bb / cc for bb, cc in zip(bads, counts)]
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if abs(rates[i] - rates[i - 1]) <= abs(rates[i] - rates[i + 1]) else i + 1
        a, z = min(i, j), max(i, j)
        groups[a] = groups[a] + groups[z]
        counts[a] += counts[z]
        bads[a] += bads[z]
        del groups[z], counts[z], bads[z]
    mapping = {}
    for grp in groups:
        label = grp[0] if len(grp) == 1 else " / ".join(map(str, grp[:4])) + (" ..." if len(grp) > 4 else "")
        for c in grp:
            mapping[c] = label
    # Unseen categories go to the largest group (most "average" behaviour).
    default = mapping[groups[int(np.argmax(counts))][0]]
    return CategoricalBins(name=x.name, mapping=mapping, default_group=default)


# ----------------------------------------------------------------------------------------
# Binner: fit on training data, transform any data to WoE
# ----------------------------------------------------------------------------------------
class WoEBinner:
    def __init__(self, fine_bins: int = 20, min_share: float = 0.05,
                 manual_edges: dict | None = None):
        self.fine_bins = fine_bins
        self.min_share = min_share
        self.manual_edges = manual_edges or {}
        self.specs: dict = {}
        self.fine_iv: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]):
        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)
        for col in numeric:
            x = X[col].astype(float)
            fe = fine_edges(x, self.fine_bins)
            self.fine_iv[col] = information_value(NumericBins(col, fe).labels(x), y)
            if col in self.manual_edges:          # analyst-chosen bins (shown for learning)
                edges = [-np.inf, *self.manual_edges[col], np.inf]
            else:
                edges, _ = monotonic_merge(x, y, fe, self.min_share)
            spec = NumericBins(col, edges)
            self._attach_woe(spec, X[col], y)
            self.specs[col] = spec
        for col in categorical:
            raw = X[col].astype(object).where(X[col].notna(), MISSING)
            self.fine_iv[col] = information_value(raw, y)
            spec = coarse_categorical(X[col], y, self.min_share)
            self._attach_woe(spec, X[col], y)
            self.specs[col] = spec
        return self

    @staticmethod
    def _attach_woe(spec, x, y):
        t = woe_iv_table(spec.labels(x), y)
        order = [l for l in spec.ordered_labels() if l in t.index]
        spec.table = t.loc[order]
        spec.woe = spec.table["woe"].to_dict()

    def transform(self, X: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        cols = cols or list(self.specs)
        out = {}
        for col in cols:
            spec = self.specs[col]
            # A bin never seen in training (e.g. Missing) gets WoE 0 = "average risk".
            out[col] = spec.labels(X[col]).map(spec.woe).fillna(0.0).astype(float).values
        return pd.DataFrame(out, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        rows = []
        for col, spec in self.specs.items():
            t = spec.table
            rows.append({
                "feature": col,
                "iv_fine": round(self.fine_iv[col], 4),
                "iv_coarse": round(float(t["iv_bin"].sum()), 4),
                "n_bins": len(t),
                "woe_monotonic": is_monotonic(spec),
            })
        df = pd.DataFrame(rows).sort_values("iv_coarse", ascending=False).reset_index(drop=True)
        df["strength"] = pd.cut(df["iv_coarse"], [-1, 0.02, 0.1, 0.3, 0.5, 99],
                                labels=["useless", "weak", "medium", "strong", "suspicious"])
        return df

    def bin_table(self) -> pd.DataFrame:
        frames = []
        for col, spec in self.specs.items():
            t = spec.table.reset_index().rename(columns={spec.table.index.name or "index": "bin"})
            t.columns = ["bin", *t.columns[1:]]
            t.insert(0, "feature", col)
            frames.append(t)
        return pd.concat(frames, ignore_index=True)


def is_monotonic(spec) -> bool:
    """WoE monotonic across the non-missing numeric bins (categoricals: not applicable -> True)."""
    if not isinstance(spec, NumericBins):
        return True
    w = spec.table.drop(index=MISSING, errors="ignore")["woe"].values
    d = np.diff(w)
    return bool(np.all(d >= -1e-12) or np.all(d <= 1e-12))
