"""Matplotlib figures for the README. One consistent, colour-blind-safe style."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"   # categorical slots, fixed order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False, "text.color": INK,
})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def bar(categories, values, path, title, ylabel, pct=True, labels=True):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(categories))
    ax.bar(x, values, color=C1, width=0.7)
    ax.set_xticks(x, [str(c) for c in categories])
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    if labels:
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.1%}" if pct else f"{v:,.0f}", (xi, v), ha="center",
                        va="bottom", fontsize=8, color=INK2, xytext=(0, 2),
                        textcoords="offset points")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def roc(curves: dict, path):
    """curves: label -> (y, pd_hat, gini)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for (lab, (y, p, g)), col in zip(curves.items(), [C1, C2, C3, C4]):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, label=f"{lab} (Gini {g:.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Random")
    ax.set(xlabel="False positive rate (goods flagged)", ylabel="True positive rate (bads caught)")
    ax.set_title("ROC curve", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def ks_plot(y, score, ks, path, title="KS plot (out-of-time)"):
    y, score = np.asarray(y), np.asarray(score)
    grid = np.sort(np.unique(score))
    cdf_b = np.searchsorted(np.sort(score[y == 1]), grid, side="right") / (y == 1).sum()
    cdf_g = np.searchsorted(np.sort(score[y == 0]), grid, side="right") / (y == 0).sum()
    i = int(np.argmax(cdf_b - cdf_g))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(grid, cdf_b, color=C2, label="Bads (cumulative)")
    ax.plot(grid, cdf_g, color=C1, label="Goods (cumulative)")
    ax.vlines(grid[i], cdf_g[i], cdf_b[i], color=INK, lw=1.2)
    ax.annotate(f"KS = {ks:.3f} at score {grid[i]:.0f}", (grid[i], (cdf_b[i] + cdf_g[i]) / 2),
                xytext=(8, 0), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Scorecard score", ylabel="Cumulative share")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def calibration(tables: dict, path):
    """tables: label -> calibration table (pd_mean, observed_rate)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    mx = 0
    for (lab, t), col in zip(tables.items(), [C1, C2]):
        ax.plot(t["pd_mean"], t["observed_rate"], marker="o", ms=5, color=col, label=lab)
        mx = max(mx, t["pd_mean"].max(), t["observed_rate"].max())
    ax.plot([0, mx * 1.05], [0, mx * 1.05], color=INK2, lw=1, ls="--", label="Perfect calibration")
    ax.set(xlabel="Mean predicted PD (decile)", ylabel="Observed bad rate")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Calibration by PD decile", loc="left")
    ax.legend(fontsize=8)
    _save(fig, path)


def psi_bars(t, psi_value, path, dev_label="Development", oot_label="Out-of-time"):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(t))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, t["expected_pct"], w, color=C1, label=dev_label)
    ax.bar(x + w / 2 + 0.01, t["actual_pct"], w, color=C2, label=oot_label)
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.set(xlabel="Score band (development deciles, 1 = riskiest)", ylabel="Share of loans")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"Score distribution stability (PSI = {psi_value:.3f})", loc="left")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def hist(values, path, title, xlabel, bins=40):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(values, bins=bins, color=C1, edgecolor="white", linewidth=0.5)
    ax.set(xlabel=xlabel, ylabel="Number of defaulted loans")
    ax.set_title(title, loc="left")
    _save(fig, path)


def grouped_bars(categories, series: dict, path, title, ylabel, pct=True):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(categories))
    k = len(series)
    w = 0.8 / k
    for j, ((lab, vals), col) in enumerate(zip(series.items(), [C1, C2, C3])):
        ax.bar(x + (j - (k - 1) / 2) * w, vals, w * 0.95, color=col, label=lab)
    ax.set_xticks(x, [str(c) for c in categories])
    ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def loss_distribution(results: list, path):
    """One panel per correlation assumption: their spreads differ by ~100x, so no shared axis."""
    fig, axes = plt.subplots(1, len(results), figsize=(9.6, 3.4))
    for ax, r, col in zip(np.atleast_1d(axes), results, [C1, C2]):
        lab = "Independent defaults" if r["rho"] == 0 else f"Correlated defaults (rho = {r['rho']})"
        ax.hist(r["losses"] / 1e6, bins=60, color=col, edgecolor="white", linewidth=0.3)
        ax.axvline(r["analytic_el"] / 1e6, color=INK, lw=1.2)
        ax.axvline(r["q99"] / 1e6, color=INK, lw=1.2, ls="--")
        ax.annotate(f"EL ${r['analytic_el'] / 1e6:,.0f}m", (r["analytic_el"] / 1e6, 0.97),
                    xycoords=("data", "axes fraction"), ha="right", va="top", fontsize=8,
                    xytext=(-4, 0), textcoords="offset points")
        ax.annotate(f"99th pct ${r['q99'] / 1e6:,.0f}m", (r["q99"] / 1e6, 0.87),
                    xycoords=("data", "axes fraction"), ha="left", va="top", fontsize=8,
                    xytext=(4, 0), textcoords="offset points")
        ax.set(xlabel="Portfolio loss ($m)", ylabel="Simulations")
        ax.set_title(lab, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Simulated 2015-cohort loss distribution (10,000 runs)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def cohort_curves(curves, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (yr, g), col in zip(curves.groupby("issue_year"), [C1, C2, C3, C4]):
        ax.plot(g["mob"], g["cum_default_rate"], color=col, label=str(yr))
    ax.set(xlabel="Months on book (to last payment before default)",
           ylabel="Cumulative default rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Cumulative default curves by issue-year cohort", loc="left")
    ax.legend(fontsize=8, title="Issue year")
    _save(fig, path)
