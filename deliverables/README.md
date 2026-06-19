# Company A — Customer Retention Proof of Concept

**GCI World 2026 Final Assignment.** A data-driven proposal that turns ~100,000 telecom customer
records into a quantified, value-based churn-retention program — validated with a paid A/B pilot.

## Contents

| File | What it is |
|------|-----------|
| `Company_A_Business_Proposal.pdf` | The 15-slide business-proposal deck (submission deliverable). |
| `Company_A_Churn_Analysis.ipynb` | Reproducible notebook: EDA → feature engineering → 4 models + 5-fold CV → value-based (EVaR) targeting → personas → ROI + sensitivity → quotation. Already executed (outputs + charts embedded). |
| `REFERENCES.md` | Full cited reference list, AI-use disclosure, and reproducibility note. |
| `figs/` | All charts used in the deck (PNG), generated from the data. |

## The problem we identified

Company A operates in a saturated U.S. wireless market where ~95% of growth comes from poaching
competitors' customers and re-acquiring a lost customer costs 5–25x more than keeping one. The data
shows churn is concentrated and predictable — and its single strongest driver, **handset/equipment
age**, is a lever Company A directly controls.

## The solution

A **model-driven proactive retention engine**. We benchmarked four models (Logistic Regression, Random
Forest, XGBoost, LightGBM) **plus soft-voting and stacking ensembles**; **XGBoost wins (ROC-AUC 0.692,
5-fold CV 0.695 ± 0.003)** and the ensembles did not clear our +0.003 AUC promotion bar, so we keep the
simpler, more governable single model. The model is only **moderately predictive (AUC ≈ 0.69)** — and we
say so. The real differentiator is *what* we rank on: instead of churn probability alone we score every
customer by **Expected Value at Risk (EVaR = calibrated P(churn) × annual revenue)**. Targeting the top
20% by EVaR captures **43.7% of total revenue-at-risk vs 28.3%** for plain churn ranking — a **1.55x**
gain in dollars protected per dollar spent. We then route **four K-means-derived personas** to tailored
offers (a documented treatment playbook) and validate with a treatment-vs-holdout A/B pilot.

**Built to be defensible (not just impressive):**
- **Fairness by design** — ethnicity, marital status, income, and credit class are dropped; an A/B test
  shows removing them does **not** hurt performance (AUC 0.6903 → 0.6909, **+0.0006**), so fairness costs us nothing.
- **Genuinely calibrated** — the model is already well-calibrated (**ECE 0.7%, Brier 0.221**; isotonic
  confirms it), and probabilities are prior-shifted from the oversampled 50/50 sample to a realistic
  **22%/yr** operating churn rate, so EVaR dollars are believable, not inflated.
- **SHAP-explained** — the top drivers (handset age, tenure, usage trend, handset price) are all
  controllable; no protected attribute appears.
- **Honest economics** — a profit-curve sets the contact threshold, and we report **bear / base / bull**
  scenarios. Base ≈ **$10.1M net benefit per 1M subscribers/yr at 1.68× ROI**; bull ≈ $17.6M (3.52×);
  the **bear case is negative (−$0.4M)** — which is exactly why we de-risk with a paid pilot before scaling.
- **Forward roadmap** — today's risk×value targeting evolves toward **uplift (causal) modelling** once the
  pilot's A/B data exists, so spend shifts to the genuinely *persuadable*.

Our fee is framed as **8.3% of the base-case value created** — the client keeps the rest.

## Reproduce the analysis

```bash
pip install pandas numpy scikit-learn matplotlib xgboost lightgbm shap jupyter
# place Client.csv and Record.csv next to the notebook (or set TELECOM_DATA_DIR)
jupyter nbconvert --to notebook --execute --inplace Company_A_Churn_Analysis.ipynb
```

The notebook shares identical logic with `../analysis/pipeline.py` (single source of truth), uses a fixed
`SEED=42` and a 75/25 stratified split, so every number in the deck is regenerated deterministically.
