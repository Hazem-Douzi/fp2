# Company A — Telecom Churn: Profit-Driven, Calibrated, Fair

Reproducible solution to the churn assignment. Predicts churn, explains drivers,
and turns predictions into a profit-optimized retention plan — without using
protected demographic attributes.

## What's here

```
churn_project/
├── data/telecom/            # Client.csv, Record.csv (raw inputs, relative paths)
├── src/
│   ├── config.py            # paths, protected-attribute list, true base rate, economics
│   ├── data.py              # merge, clean, drop protected attrs, feature engineering
│   ├── model.py             # train/compare, tune, calibrate, base-rate reweight, SHAP
│   ├── business.py          # EVaR ranking, profit curve, uplift (T-learner), scenarios
│   ├── plots.py             # all figures
│   └── run_pipeline.py      # runs everything -> outputs/results/results.json + figures
├── churn_analysis.ipynb     # reproducible top-to-bottom notebook (calls src/)
├── build_deck.py            # builds the .pptx from results.json + figures
├── build_pdf.py             # builds the .pdf (mirrors the deck)
├── REFERENCES.md            # cited assumptions + methods
└── outputs/
    ├── results/results.json # every number used in the deck
    ├── figures/*.png        # all charts
    ├── Company_A_Churn_Deck.pptx
    └── Company_A_Churn_Deck.pdf
```

## Reproduce

```bash
pip install -r requirements.txt

# 1. Run the full analysis (writes results.json + figures)
python -m src.run_pipeline

# 2. (optional) Rebuild the notebook and execute it end to end
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace churn_analysis.ipynb

# 3. Rebuild the deck + PDF
python build_deck.py
python build_pdf.py
```

All paths are relative to `churn_project/`, so the project runs anywhere after
the two CSVs are placed in `data/telecom/`.

## Key design decisions (what makes this honest)

1. **No protected attributes.** Ethnicity, income, marital status, dwelling, etc.
   (20 fields) are dropped before modeling, with an automated guard that fails the
   build if one leaks through one-hot encoding.
2. **Believable accuracy.** Tuned + isotonic-calibrated XGBoost, ROC-AUC ≈ 0.70 —
   normal for behavioral telecom churn. A 0.95+ AUC would signal target leakage.
3. **Base-rate correction.** The sample is oversampled to ~50% churn; predictions are
   re-weighted to the true ~2% monthly rate so every dollar figure is credible.
4. **Profit-based decisions.** Customers are ranked by expected value at risk
   (P(churn) × margin × CLV) and the contact depth is chosen on a profit curve, not an
   arbitrary 0.5 threshold.
5. **Explainability & action.** SHAP drivers (all actionable, none demographic),
   K-means behavioral personas, and a T-learner uplift view of "persuadables."
6. **Honest economics.** Bear / base / bull scenarios with assumptions cited in
   `REFERENCES.md`; the bear case is allowed to be net-negative.
7. **Honest uplift caveat.** The data is observational (no treatment/control), so the
   uplift figures are a clearly-labeled simulation, paired with an A/B-test recommendation.
