# References — Company A Customer Retention Proof of Concept

**GCI World 2026 — Final Assignment**
Prepared for the Company A Executive Team · June 2026

All figures cited in the slide deck and notebook trace to one of the sources below. Market and
retention-economics benchmarks are external (cited); all dataset statistics, model scores, lift, and
the financial business case are the authors' own analysis of the provided data, fully reproducible in
`Company_A_Churn_Analysis.ipynb`.

---

## Market analysis & retention economics

1. **IBISWorld (2025).** *Wireless Telecommunications Carriers in the US — Market Size & Industry Statistics.* IBISWorld Industry Report. — U.S. wireless market size (~$344B) and saturation. https://www.ibisworld.com/united-states/market-research-reports/wireless-telecommunications-carriers-industry/

2. **Verizon Communications (2025).** *Q4 2024 / Full-Year 2024 Earnings Report and Financial Statements.* — Postpaid phone churn (~0.9–1.0%/mo) and ARPU disclosures. https://www.verizon.com/about/investors

3. **T-Mobile US (2025).** *Q4 2024 Investor Factbook.* — Postpaid phone churn benchmark for a leading U.S. carrier. https://www.t-mobile.com/news/

4. **Reichheld, F. F., & Sasser, W. E. (1990).** "Zero Defections: Quality Comes to Services." *Harvard Business Review*, 68(5), 105–111. — Origin of the "5% retention → 25–95% profit" finding.

5. **Bain & Company / Reichheld, F.** *Prescription for Cutting Costs* (Bain & Company brief). — Retention vs. acquisition cost economics and profit impact of loyalty.

6. **Gallo, A. (2014).** "The Value of Keeping the Right Customers." *Harvard Business Review*, October 29, 2014. — Synthesis of the 5–25x cost-to-acquire-vs-retain range. https://hbr.org/2014/10/the-value-of-keeping-the-right-customers

7. **Neslin, S. A., Gupta, S., Kamakura, W., Lu, J., & Mason, C. H. (2006).** "Defection Detection: Measuring and Understanding the Predictive Accuracy of Customer Churn Models." *Journal of Marketing Research*, 43(2), 204–211. — Demonstrates profit gains from model-driven, lift-targeted churn management in telecom.

---

## Methods, tools & modeling

8. **Chen, T., & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 785–794. https://doi.org/10.1145/2939672.2939785

9. **Ke, G., et al. (2017).** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems (NeurIPS) 30*, 3146–3154. — Fourth benchmarked model.

10. **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830. — Pipelines, preprocessing, Logistic Regression, Random Forest, 5-fold cross-validation, and evaluation metrics.

11. **McKinney, W. (2010).** "Data Structures for Statistical Computing in Python." *Proceedings of the 9th Python in Science Conference* (pandas). — Data wrangling and the table merge.

12. **Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).** "New insights into churn prediction in the telecommunication sector: A profit driven data mining approach." *European Journal of Operational Research*, 218(1), 211–229. — Basis for our **value-based (Expected-Value-at-Risk) targeting**: ranking by expected revenue lost, not churn probability alone.

13. **Niculescu-Mizil, A., & Caruana, R. (2005).** "Predicting Good Probabilities with Supervised Learning." *Proceedings of the 22nd International Conference on Machine Learning (ICML '05)*, 625–632. — Calibration assessment (reliability, Brier score) and isotonic/Platt scaling, used to validate that our probabilities are trustworthy for dollar decisions.

14. **Gutierrez, P., & Gérardy, J.-Y. (2017).** "Causal Inference and Uplift Modelling: A Review of the Literature." *Proceedings of Machine Learning Research*, 67, 1–13. — Basis for the **uplift-modelling roadmap**: targeting persuadable customers using experimental (A/B) data.

---

## Provided course materials & dataset

15. **GCI World (2026).** *Company A Dataset Overview* and *Final Assignment README* (provided course materials).

16. **Dataset.** Cell2Cell-style telecom customer dataset provided by the course: `Client.csv` (customer profile, ~100,000 rows) joined to `Record.csv` (usage history + churn label) on `Customer_ID`. The modeling sample is balanced ~50/50 on churn by design (not the population base rate); the business case therefore uses an explicitly stated operating churn assumption (22%/year ≈ 2.0%/month).

---

## AI-use disclosure

Generative AI (ChatGPT / v0) was used to assist with drafting narrative text and assembling diagrams and
slides. The Omnicampus submission's references field contains the shared chat URL as required. All data
analysis, code, model training, metrics, lift/gains, feature importances, and the financial business case
were produced and verified by the authors and are reproducible end-to-end in the accompanying notebook.

## Reproducibility note

Every dataset statistic and model score in the deck is regenerated by running
`Company_A_Churn_Analysis.ipynb` top to bottom (Python 3, `pandas`, `numpy`, `scikit-learn`, `xgboost`,
`lightgbm`, `shap`, `matplotlib`; fixed `SEED=42`, 75/25 stratified split). The notebook shares identical
logic with `../analysis/pipeline.py`, the single source of truth for `results.json`, the deck, and these
docs. Headline results:

- **Data hygiene:** protected attributes (ethnicity, marital status, income, credit class) and 9
  high-missing (>20% NA) columns are dropped before modelling.
- **Fairness A/B:** removing protected attributes does **not** hurt performance — AUC moves
  **0.6903 → 0.6909 (+0.0006)** — so excluding them is ethically sound at zero predictive cost.
- **Models:** ARPU $58.72/mo · **XGBoost ROC-AUC 0.692, 5-fold CV 0.695 ± 0.003** (LightGBM 0.690,
  RF 0.676, LogReg 0.615). Soft-voting & stacking ensembles reached AUC 0.6921 (**+0.0005**, below the
  0.003 promotion bar) → keep XGBoost for simpler, governable deployment.
- **Calibration:** the model is already well-calibrated (**ECE 0.7%, Brier 0.221**; an isotonic calibrator
  confirms no material gain), then probabilities are prior-shifted from the oversampled 50/50 sample to a
  realistic **22%/yr** population churn rate, so EVaR figures are on a believable dollar scale.
- **Value targeting:** ranking by EVaR captures **43.7% of revenue-at-risk in the top 20% vs 28.3%** for
  churn-probability ranking (**1.55×**).
- **Business case (per 1M subscribers/yr):** base ≈ **$10.1M net benefit at 1.68× ROI**; bull ≈ $17.6M
  (3.52×); **bear is −$0.4M (−0.05×)** — the downside is disclosed, which is why scaling is gated on a
  paid A/B pilot.
