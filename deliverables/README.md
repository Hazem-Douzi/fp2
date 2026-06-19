# Company A — Customer Retention Proof of Concept

**GCI World 2026 Final Assignment.** A data-driven proposal that turns ~100,000 telecom customer
records into a quantified, ROI-positive churn-retention program.

## Contents

| File | What it is |
|------|-----------|
| `Company_A_Business_Proposal.pdf` | The 15-slide business-proposal deck (submission deliverable). |
| `Company_A_Churn_Analysis.ipynb` | Reproducible notebook: EDA → feature engineering → 3 models → lift/gains → business case. Already executed (outputs + charts embedded). |
| `REFERENCES.md` | Full cited reference list, AI-use disclosure, and reproducibility note. |
| `figs/` | All charts used in the deck (PNG), generated from the data. |

## The problem we identified

Company A operates in a saturated U.S. wireless market where ~95% of growth comes from poaching
competitors' customers and re-acquiring a lost customer costs 5–25x more than keeping one. The data
shows churn is concentrated and predictable — and its single strongest driver, **handset/equipment
age**, is a lever Company A directly controls.

## The solution

A **model-driven proactive retention engine**: score every customer monthly with XGBoost
(ROC-AUC 0.693), target the top 30% risk segment (which contains ~42% of all churners), and trigger
proactive device-upgrade offers before competitors do — validated with a treatment-vs-holdout A/B test.

**Quantified impact:** ≈ **$22.6M net benefit per 1M subscribers per year at 2.57x ROI**
(churn 2.0% → 1.79%/mo), scaling roughly linearly with subscriber base.

## Reproduce the analysis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost jupyter
# place Client.csv and Record.csv next to the notebook (or set TELECOM_DATA_DIR)
jupyter nbconvert --to notebook --execute --inplace Company_A_Churn_Analysis.ipynb
```

The notebook uses a fixed `random_state=42` and a 75/25 train/test split, so every number in the deck
is regenerated deterministically.
