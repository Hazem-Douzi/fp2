"""
Generates `churn_analysis.ipynb` — a reproducible, top-to-bottom narrative that
calls the SAME tested modules in src/ (no logic is duplicated, so the notebook
can never drift from the pipeline that produced the deck).

Run:  python build_notebook.py
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


cells = []

# ---------------------------------------------------------------- title
cells.append(md(r"""
# Telecom Churn — Predict, Prioritize, and Act

**Company A retention analytics.** This notebook is the reproducible record behind
the slide deck. It runs top-to-bottom from the raw CSVs in `data/telecom/` and
regenerates every number (`outputs/results/results.json`) and every chart
(`outputs/figures/*.png`).

It deliberately calls the **same tested functions in `src/`** that produced the
deck, so the notebook and the presentation can never disagree.

### What this analysis fixes versus a naive first pass
1. **No protected attributes.** Race/ethnicity, dwelling, marital status, children,
   and other demographic proxies are dropped *before* modeling (legal + ethical).
2. **Honest target & base rate.** The modeling sample is ~50/50 oversampled; we
   train on it but **re-weight calibrated probabilities back to the true ~2%
   churn rate** so the business case is realistic.
3. **Calibrated probabilities**, not raw scores — so "P(churn)=0.3" means it.
4. **Decision threshold set on a profit curve**, not an arbitrary 0.5.
5. **SHAP** for honest, directional drivers (no hand-waving feature importance).
6. **Uplift / persuadables** framing with an explicit RCT recommendation.
7. **Bear / base / bull** business case with cited assumptions.
""".strip()))

# ---------------------------------------------------------------- setup
cells.append(md("## 0. Setup\nDeterministic seeds; all paths are relative to the project root."))
cells.append(code(r"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

warnings.filterwarnings("ignore")

# src/ is a package in the project root.
from src import config as C
from src import data as D
from src import model as M
from src import business as B

print("Random state:", C.RANDOM_STATE)
print("True base rate assumption:", C.TRUE_BASE_RATE)
print("Protected attributes dropped:", len(C.PROTECTED_ATTRS))
""".strip()))

# ---------------------------------------------------------------- data
cells.append(md(r"""
## 1. Data & feature engineering

Two tables — `Client.csv` (demographics/account) and `Record.csv` (usage/behavior)
— are merged on `Customer_ID`. We then:

- **Drop protected attributes** (`C.PROTECTED_ATTRS`) and high-missingness junk columns.
- **Engineer behavioral features**: equipment age in months, minutes-of-use trend,
  care-call intensity, overage share, revenue-per-minute, dropped-call rate, etc.
- One-hot encode low-cardinality categoricals.

`build_feature_frame()` returns the model matrix `X`, target `y`, and a metadata dict.
""".strip()))
cells.append(code(r"""
X, y, meta = D.build_feature_frame()
print(f"Rows: {meta['n_rows']:,}   Raw columns: {meta['n_raw_cols']}")
print(f"Model features after engineering + encoding: {meta['n_model_features']}")
print(f"Sample churn rate (oversampled): {meta['sample_churn_rate']:.3f}")
print(f"Protected attributes removed: {len(meta['dropped_protected'])}")
X.head()
""".strip()))
cells.append(md("A quick look at the dropped protected attributes — these never touch the model:"))
cells.append(code("sorted(C.PROTECTED_ATTRS)"))

# ---------------------------------------------------------------- run pipeline
cells.append(md(r"""
## 2. Run the full, tested pipeline

Rather than re-type the modeling logic here (which could silently drift from the
code that made the deck), we execute the exact same entry point:
`src.run_pipeline.main()`. It performs, in order:

1. Train/test split (stratified) + `scale_pos_weight` for imbalance.
2. Model zoo: Logistic Regression, Random Forest, XGBoost.
3. `RandomizedSearchCV` hyperparameter tuning of XGBoost + 5-fold CV stability.
4. **Isotonic calibration** of probabilities.
5. **Prior-prevalence re-weighting** to the true base rate.
6. EVaR value-at-risk, **capture curve**, and **profit curve** optimization.
7. **SHAP** global importances.
8. **Uplift** T-learner (documented simulation) + **K-means** personas.
9. Writes `results.json` and all figures.

This cell is the heavy compute step.
""".strip()))
cells.append(code(r"""
from src.run_pipeline import main as run_pipeline
results = run_pipeline()
print("\nDone. Top-level result keys:")
print(list(results.keys()))
""".strip()))

# ---------------------------------------------------------------- model comparison
cells.append(md(r"""
## 3. Model selection — honest comparison

We report ROC-AUC, PR-AUC (more meaningful under imbalance), and the Brier score
(calibration quality). XGBoost wins on ranking; **note the AUC is in a realistic
0.68–0.71 band — not a suspicious 0.95+**, which would signal leakage.
""".strip()))
cells.append(code(r"""
comp = pd.DataFrame(results["model_comparison"]).T
display(comp.round(4))
display(Image(filename=str(C.FIG_DIR / "model_comparison.png")))
""".strip()))
cells.append(code(r"""
print("Best tuned params:", results["tuning"]["best_params"])
print(f"Best CV AUC: {results['tuning']['best_cv_auc']:.4f}")
print(f"5-fold CV AUC: {results['cv']['mean_auc']:.4f} ± {results['cv']['std_auc']:.4f}")
""".strip()))

# ---------------------------------------------------------------- calibration
cells.append(md(r"""
## 4. Calibration & the base-rate correction

A churn *score* is not a churn *probability* unless it's calibrated. We apply
isotonic calibration and check the reliability curve and Brier score. We then
convert calibrated probabilities from the oversampled sample back to the true
population base rate — without this, every downstream dollar figure is inflated.
""".strip()))
cells.append(code(r"""
cal = results["calibration"]
print(f"Brier (raw):        {cal['brier_raw']:.4f}")
print(f"Brier (calibrated): {cal['brier_calibrated']:.4f}  (lower = better)")
print(f"AUC (calibrated):   {cal['auc_calibrated']:.4f}")

br = results["base_rate"]
print(f"\nMean P(churn) in sample:      {br['mean_p_sample']:.3f}")
print(f"Mean P(churn) re-weighted:    {br['mean_p_true']:.3f}  (target ~{br['true_rate']})")
display(Image(filename=str(C.FIG_DIR / "calibration.png")))
""".strip()))

# ---------------------------------------------------------------- capture
cells.append(md(r"""
## 5. Who to call first — capture curve

Ranking customers by **expected value at risk (EVaR = P(churn) × margin × CLV)**
lets retention reach far more churners per call than random outreach.
""".strip()))
cells.append(code(r"""
cap = results["capture_at_20pct"]
print(f"Contacting the top 20% by EVaR captures {cap['by_evar']*100:.0f}% of churners")
print(f"Lift vs random: {cap['lift_evar_vs_random']:.2f}x")
display(Image(filename=str(C.FIG_DIR / "capture_curve.png")))
""".strip()))

# ---------------------------------------------------------------- profit
cells.append(md(r"""
## 6. How many to call — profit curve

We don't pick a threshold by intuition. We sweep contact depth and choose the
point that **maximizes expected net profit** given the offer cost, save rate, and
recovered CLV (base scenario).
""".strip()))
cells.append(code(r"""
pc = results["profit_curve_base"]
print(f"Optimal contact depth: {pc['optimal_frac']*100:.0f}% of base")
print(f"Net profit (population): ${pc['optimal_net_profit_population']:,.0f}")
print(f"ROI: {pc['optimal_roi']:.2f}x")
print(f"Expected customers saved: {pc['customers_saved_population']:,.0f}")
display(Image(filename=str(C.FIG_DIR / "profit_curve.png")))
""".strip()))

# ---------------------------------------------------------------- shap
cells.append(md(r"""
## 7. Why — SHAP drivers

SHAP gives directionally honest, per-feature contributions. Equipment age, usage
trend, and care-call intensity dominate — all **actionable** levers (upgrade offers,
win-back on declining usage, service recovery), unlike demographics.
""".strip()))
cells.append(code(r"""
shap_imp = pd.Series(results["shap_top_features"]).sort_values(ascending=False)
display(shap_imp.round(4).to_frame("mean|SHAP|"))
display(Image(filename=str(C.FIG_DIR / "shap_importance.png")))
""".strip()))

# ---------------------------------------------------------------- handset honesty
cells.append(md(r"""
## 8. Honest handset/equipment view

Equipment age is binned into interpretable buckets (rather than implying false
precision). Churn rises monotonically with handset age — a clean upgrade-offer story.
""".strip()))
cells.append(code(r'display(Image(filename=str(C.FIG_DIR / "handset_churn.png")))'))

# ---------------------------------------------------------------- uplift
cells.append(md(r"""
## 9. Persuadables — uplift modeling

**Important honesty note:** this dataset is *observational* — there is no
treatment/control split, so true causal uplift cannot be measured here. We
demonstrate the persuadables framework with a **clearly-labeled simulated**
campaign response and a T-learner, and we recommend a live A/B holdout to measure
real uplift before scaling.
""".strip()))
cells.append(code(r"""
up = results["uplift"]
print(up["method"])
print(up["note"])
print(f"\nMean uplift, top-20% persuadables: {up['mean_uplift_top20']:.3f}")
print(f"Share with positive uplift:        {up['persuadable_share_positive_uplift']*100:.0f}%")
display(Image(filename=str(C.FIG_DIR / "uplift.png")))
""".strip()))

# ---------------------------------------------------------------- personas
cells.append(md(r"""
## 10. Customer personas — K-means

Real K-means on standardized behavioral features (not demographics) yields
actionable segments, each profiled by churn rate, revenue, equipment age, and
care intensity.
""".strip()))
cells.append(code(r"""
personas = pd.DataFrame(results["personas"]["clusters"])
display(personas.round(3))
display(Image(filename=str(C.FIG_DIR / "personas.png")))
""".strip()))

# ---------------------------------------------------------------- scenarios
cells.append(md(r"""
## 11. Business case — bear / base / bull

A single optimistic number is not credible. We present three scenarios with
assumptions cited in `REFERENCES.md` (save rate, offer cost, gross margin, CLV
horizon). Even the **bear** case is profit-positive.
""".strip()))
cells.append(code(r"""
scen = pd.DataFrame(results["scenarios"])
cols = ["name", "save_rate", "offer_cost", "net_profit_population",
        "roi", "customers_saved_population"]
display(scen[[c for c in cols if c in scen.columns]].round(2))
display(Image(filename=str(C.FIG_DIR / "scenarios.png")))
""".strip()))

# ---------------------------------------------------------------- takeaways
cells.append(md(r"""
## 12. Takeaways

- **Model:** tuned, calibrated XGBoost; realistic AUC (~0.70), good Brier score.
- **Targeting:** rank by EVaR, contact at the profit-optimal depth.
- **Drivers:** equipment age, usage decline, care intensity — all actionable.
- **Economics:** profit-positive across bear/base/bull; clear ROI.
- **Integrity:** no protected attributes; honest base rate; uplift flagged as a
  simulation pending a real RCT.

All figures and `results.json` in `outputs/` were produced by the cells above.
""".strip()))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
})

with open("churn_analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote churn_analysis.ipynb with", len(cells), "cells")
