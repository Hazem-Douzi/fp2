# Company A — Customer Retention Proof of Concept

**GCI World 2026 Final Assignment.** A data-driven proposal that turns ~100,000 telecom customer
records into a quantified, ROI-positive churn-retention program.

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
Forest, XGBoost, LightGBM); **XGBoost wins (ROC-AUC 0.693, 5-fold CV 0.696 ± 0.004)**. The differentiator
is *what* we rank on: instead of churn probability alone we score every customer by **Expected Value at
Risk (EVaR = P(churn) × annual revenue)**. Targeting the top 20% by EVaR captures **44.8% of total
revenue-at-risk vs 27.9%** for plain churn ranking — a **1.61x** gain in dollars protected per dollar
spent. We then route four data-derived personas to tailored offers (device upgrade, bill-shock review,
loyalty/tenure rewards) and validate with a treatment-vs-holdout A/B test.

**Quantified impact:** ≈ **$9.8M net benefit per 1M subscribers per year at 1.64x ROI**, positive across
the entire save-rate × offer-cost sensitivity grid, scaling roughly linearly with the subscriber base. Our
fee is framed as **8.5% of the value created** — the client keeps the rest.

## Reproduce the analysis

```bash
pip install pandas numpy scikit-learn matplotlib xgboost lightgbm jupyter
# place Client.csv and Record.csv next to the notebook (or set TELECOM_DATA_DIR)
jupyter nbconvert --to notebook --execute --inplace Company_A_Churn_Analysis.ipynb
```

The notebook uses a fixed `random_state=42` and a 75/25 train/test split, so every number in the deck
is regenerated deterministically.
