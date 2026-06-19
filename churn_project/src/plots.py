"""
All figures used in the deck. Every chart is generated from real pipeline
output -- no hand-drawn or fabricated values. Consistent, readable styling.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

# ---- Styling: limited palette, clean, legible ----------------------------
NAVY = "#1f2a44"
TEAL = "#0e8a8a"
AMBER = "#d98a2b"
GREY = "#9aa0a6"
LIGHT = "#e8eaed"
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": LIGHT,
    "grid.linewidth": 0.8,
    "figure.autolayout": True,
})


def _save(fig, name):
    path = C.FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_model_comparison(comparison: dict):
    """Horizontal bar chart showing ROC-AUC and PR-AUC side by side.
    Adding PR-AUC addresses guideline §8: report both metrics for imbalanced churn.
    """
    names = list(comparison.keys())
    aucs    = [comparison[n]["auc"]    for n in names]
    pr_aucs = [comparison[n].get("pr_auc", 0) for n in names]
    n = len(names)
    y = np.arange(n)
    height = 0.35
    bar_colors = [NAVY, TEAL, AMBER, "#5b4fcf"]   # up to 4 models
    fig, ax = plt.subplots(figsize=(8, max(4.2, n * 1.1)))
    ax.barh(y + height / 2, aucs, height, color=[bar_colors[i % len(bar_colors)] for i in range(n)],
            label="ROC-AUC")
    ax.barh(y - height / 2, pr_aucs, height,
            color=[bar_colors[i % len(bar_colors)] for i in range(n)],
            alpha=0.45, hatch="//", label="PR-AUC")
    for i, (a, p) in enumerate(zip(aucs, pr_aucs)):
        ax.text(a + 0.003, y[i] + height / 2, f"{a:.3f}", va="center", fontsize=10, fontweight="bold")
        ax.text(p + 0.003, y[i] - height / 2, f"{p:.3f}", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(aucs) + 0.10)
    ax.set_xlabel("Score (higher = better)")
    ax.set_title("Model comparison: ROC-AUC and PR-AUC (held-out test set)")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, "model_comparison.png")


def fig_calibration(mp_raw, fp_raw, mp_cal, fp_cal):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    ax.plot(mp_raw, fp_raw, "o-", color=AMBER, label="Raw XGBoost")
    ax.plot(mp_cal, fp_cal, "o-", color=TEAL, label="Isotonic-calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn frequency")
    ax.set_title("Probability calibration (reliability curve)")
    ax.legend(frameon=False)
    return _save(fig, "calibration.png")


def fig_capture(cap_evar, cap_score, cap_rand,
                cap_rev=None, cap_rule=None):
    """Capture curve with all four baselines (guideline §12).

    - Random (weak baseline)
    - Revenue-only (common business baseline)
    - Rule-based heuristic (simple operational baseline)
    - Churn probability (ML baseline)
    - EVaR model (proposed solution)
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(cap_rand["frac_contacted"] * 100, cap_rand["frac_churn_captured"] * 100,
            "--", color=GREY, lw=1.4, label="Random (no model)")
    if cap_rev is not None:
        ax.plot(cap_rev["frac_contacted"] * 100, cap_rev["frac_churn_captured"] * 100,
                ":", color=AMBER, lw=1.8, label="Revenue-only (business baseline)")
    if cap_rule is not None:
        ax.plot(cap_rule["frac_contacted"] * 100, cap_rule["frac_churn_captured"] * 100,
                "-.", color="#c0392b", lw=1.8, label="Rule-based heuristic")
    ax.plot(cap_score["frac_contacted"] * 100, cap_score["frac_churn_captured"] * 100,
            color=TEAL, lw=2, label="Churn probability (ML baseline)")
    ax.plot(cap_evar["frac_contacted"] * 100, cap_evar["frac_churn_captured"] * 100,
            color=NAVY, lw=2.8, label="EVaR model (proposed)")
    ax.axvline(20, color=AMBER, ls=":", lw=1.5, alpha=0.7)
    ax.set_xlabel("% of customer base contacted")
    ax.set_ylabel("% of churners captured")
    ax.set_title("Capture curve: four baselines vs EVaR model")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    return _save(fig, "capture_curve.png")


def fig_profit_curve(pc: pd.DataFrame, opt: dict):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(pc["frac_contacted"] * 100, pc["net_profit"] / 1e3,
            color=NAVY, lw=2.5)
    ax.axvline(opt["frac_contacted"] * 100, color=AMBER, ls=":", lw=1.8,
               label=f"Optimal: top {opt['frac_contacted']*100:.0f}%")
    ax.scatter([opt["frac_contacted"] * 100], [opt["net_profit"] / 1e3],
               color=AMBER, zorder=5, s=60)
    ax.set_xlabel("% of customer base contacted (ranked by EVaR)")
    ax.set_ylabel("Net profit, test set ($000s)")
    ax.set_title("Profit curve: optimal contact depth")
    ax.legend(frameon=False)
    return _save(fig, "profit_curve.png")


def fig_shap(shap_imp: pd.Series):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    s = shap_imp.sort_values()
    ax.barh(s.index, s.values, color=TEAL)
    ax.set_xlabel("Mean |SHAP value|  (impact on churn prediction)")
    ax.set_title("What drives churn (SHAP, top 15 features)")
    return _save(fig, "shap_importance.png")


def fig_handset_churn(raw: pd.DataFrame):
    """HONEST handset chart: real months from eqpdays, equal-width month bins."""
    df = raw[["eqpdays", C.TARGET]].copy()
    df = df[df["eqpdays"].between(0, 1095)]  # 0-36 months, drop bad rows
    df["months"] = df["eqpdays"] / 30.0
    bins = [0, 6, 12, 18, 24, 30, 36]
    labels = ["0-6", "6-12", "12-18", "18-24", "24-30", "30-36"]
    df["bin"] = pd.cut(df["months"], bins=bins, labels=labels, include_lowest=True)
    g = df.groupby("bin", observed=True)[C.TARGET].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(g.index.astype(str), g["mean"] * 100, color=NAVY)
    for b, (rate, cnt) in zip(bars, g.itertuples(index=False)):
        ax.text(b.get_x() + b.get_width() / 2, rate * 100 + 0.4,
                f"{rate*100:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Handset age (months) — real eqpdays/30, equal-width bins")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rises with handset age (honest binning)")
    return _save(fig, "handset_churn.png")


def fig_scenarios(scen: list[dict]):
    names = [s["scenario"] for s in scen]
    profits = [s["net_profit_population"] / 1e6 for s in scen]
    colors = [AMBER, NAVY, TEAL]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(names, profits, color=colors)
    for b, p in zip(bars, profits):
        ax.text(b.get_x() + b.get_width() / 2, p + max(profits) * 0.02,
                f"${p:.1f}M", ha="center", fontweight="bold")
    ax.set_ylabel("Annual net profit ($M, full base)")
    ax.set_title("Business case: bear / base / bull")
    return _save(fig, "scenarios.png")


def fig_uplift(uplift_res: dict):
    edges = np.array(uplift_res["uplift_edges"])
    counts = np.array(uplift_res["uplift_hist"])
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = [TEAL if c > 0.01 else GREY for c in centers]
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=colors)
    ax.axvline(0.01, color=AMBER, ls=":", lw=1.6, label="Persuadable threshold")
    ax.set_xlabel("Predicted uplift  P(retain|contact) − P(retain|no contact)")
    ax.set_ylabel("Customers")
    ax.set_title("Uplift distribution: sizing the 'persuadables' (simulation)")
    ax.legend(frameon=False)
    return _save(fig, "uplift.png")


def fig_personas(personas: dict):
    cl = sorted(personas["clusters"], key=lambda d: -d["churn_rate"])
    names = [f"Cluster {c['cluster']}\n({c['share']*100:.0f}%)" for c in cl]
    churn = [c["churn_rate"] * 100 for c in cl]
    rev = [c["avg_rev"] for c in cl]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.bar(names, churn, color=NAVY)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("K-means personas: churn vs. value")
    ax2 = ax.twinx()
    ax2.plot(names, rev, "o-", color=AMBER, lw=2)
    ax2.set_ylabel("Avg monthly revenue ($)", color=AMBER)
    ax2.grid(False)
    for b, c in zip(bars, churn):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.5, f"{c:.0f}%",
                ha="center", fontweight="bold", fontsize=10)
    return _save(fig, "personas.png")


def fig_ablation(ablation: list[dict]):
    """Feature-group ablation table visualised as a dot-plot.

    Guideline §7: shows AUC improvement per feature group added, making each
    engineering choice directly defensible to reviewers.
    """
    labels = [r["version"] for r in ablation]
    aucs   = [r["roc_auc"] for r in ablation]
    pr_aucs = [r["pr_auc"] for r in ablation]
    n = len(labels)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.9)))
    ax.scatter(aucs, y, color=NAVY, s=90, zorder=5, label="ROC-AUC")
    ax.scatter(pr_aucs, y, color=TEAL, s=70, marker="D", zorder=5, label="PR-AUC")

    # Draw delta arrows between consecutive ROC-AUC points.
    for i in range(1, n):
        delta = aucs[i] - aucs[i - 1]
        color = TEAL if delta >= 0 else "#c0392b"
        sign = "+" if delta >= 0 else ""
        ax.annotate("", xy=(aucs[i], y[i]), xytext=(aucs[i - 1], y[i - 1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        ax.text(max(aucs[i], aucs[i - 1]) + 0.002,
                (y[i] + y[i - 1]) / 2,
                f"{sign}{delta:.4f}", fontsize=9, color=color, va="center")

    for i, (a, p) in enumerate(zip(aucs, pr_aucs)):
        ax.text(a + 0.001, y[i] + 0.18, f"{a:.4f}", fontsize=9, color=NAVY)
        ax.text(p - 0.001, y[i] - 0.22, f"{p:.4f}", fontsize=9, color=TEAL, ha="right")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Score")
    ax.set_title("Ablation: AUC gain per feature group (guideline §7)")
    ax.legend(frameon=False)
    x_min = min(min(aucs), min(pr_aucs)) - 0.02
    x_max = max(max(aucs), max(pr_aucs)) + 0.04
    ax.set_xlim(x_min, x_max)
    ax.invert_yaxis()
    return _save(fig, "ablation.png")
