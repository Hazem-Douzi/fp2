# References & Assumptions

This document backs every non-obvious modeling and business assumption with a
citation. Where a precise figure for "Company A" is unknown, we use published US
wireless-industry benchmarks and state the number as an explicit, sourced
assumption rather than an unsupported guess.

---

## 1. Churn base rate

The modeling sample provided is oversampled to ~50% churn (a common practice for
training class balance). The **true** monthly post-paid churn rate for US wireless
carriers is far lower; we re-weight model probabilities to a documented base rate
of **~2% monthly (~24% annualized)**.

- [1] Statista — *Churn rate of wireless carriers in the United States* (monthly
  post-paid churn for major carriers historically ~0.8%–2.0%).
  https://www.statista.com/statistics/283511/average-churn-rate-wireless-carriers-usa/
- [2] FCC — *Annual Report and Analysis of Competitive Market Conditions with
  Respect to Mobile Wireless Services* (industry churn context).
  https://www.fcc.gov/
- [3] The dataset documentation describes a sampled/balanced extract; oversampling
  for churn modeling is standard — see Burez & Van den Poel (2009), *Handling
  class imbalance in customer churn prediction*, Expert Systems with Applications
  36(3): 4626–4636. https://doi.org/10.1016/j.eswa.2008.05.027

## 2. Save rate (effectiveness of a retention offer)

Probability that a contacted, at-risk customer is actually retained by a proactive
offer. Industry win-back/save programs typically report **20%–45%**; we model
**bear 20% / base 30% / bull 45%**.

- [4] Reichheld, F. & Schefter, P. (2000), *E-Loyalty: Your Secret Weapon on the
  Web*, Harvard Business Review — economics of retention vs. acquisition.
- [5] Verbeke et al. (2012), *New insights into churn prediction in the telco
  sector: A profit-driven data mining approach*, EJOR 218(1): 211–229.
  https://doi.org/10.1016/j.ejor.2011.09.031 — profit-based retention framing and
  acceptance/save rates.

## 3. Retention offer cost

Per-contact incentive cost (discount, credit, device subsidy, agent time). Modeled
**bear $40 / base $30 / bull $25** per contacted customer.

- [6] Neslin et al. (2006), *Defection Detection: Measuring and Understanding the
  Predictive Accuracy of Customer Churn Models*, Journal of Marketing Research
  43(2): 204–211. https://doi.org/10.1509/jmkr.43.2.204 — incentive cost in
  churn-management ROI.

## 4. Gross margin & customer lifetime value (CLV)

Recovered value of a saved customer = monthly revenue × gross margin × expected
remaining tenure (CLV horizon). We use a gross margin of **~55%** and a CLV
horizon of **18 months (base)**, varied across scenarios.

- [7] Gupta, Lehmann & Stuart (2004), *Valuing Customers*, Journal of Marketing
  Research 41(1): 7–18. https://doi.org/10.1509/jmkr.41.1.7.25084 — CLV methodology.
- [8] Telecom operator gross margins commonly reported in the 50%–65% range in
  carrier 10-K filings (e.g., major US carriers' service-revenue margins).

## 5. Uplift / persuadables modeling

The provided data is **observational** (no randomized treatment/control), so true
causal uplift cannot be measured from it. We demonstrate the persuadables framework
with a clearly-labeled simulated response and recommend a live A/B holdout.

- [9] Radcliffe, N. & Surry, P. (2011), *Real-World Uplift Modelling with
  Significance-Based Uplift Trees*, Stochastic Solutions white paper.
- [10] Gutierrez, P. & Gérardy, J. (2017), *Causal Inference and Uplift Modelling:
  A Review of the Literature*, PMLR 67: 1–13. — T-learner / two-model approach.
- [11] Künzel et al. (2019), *Metalearners for estimating heterogeneous treatment
  effects using machine learning*, PNAS 116(10): 4156–4165.
  https://doi.org/10.1073/pnas.1804597116 — S/T/X-learners.

## 6. Methodology — calibration, imbalance, profit-based thresholding

- [12] Niculescu-Mizil, A. & Caruana, R. (2005), *Predicting Good Probabilities
  with Supervised Learning*, ICML — isotonic/Platt calibration.
- [13] King, G. & Zeng, L. (2001), *Logistic Regression in Rare Events Data*,
  Political Analysis 9(2): 137–163 — prior-correction / base-rate reweighting.
- [14] Elkan, C. (2001), *The Foundations of Cost-Sensitive Learning*, IJCAI —
  threshold selection from costs (profit curve).
- [15] Lundberg, S. & Lee, S. (2017), *A Unified Approach to Interpreting Model
  Predictions* (SHAP), NeurIPS. https://github.com/shap/shap
- [16] Chen, T. & Guestrin, C. (2016), *XGBoost: A Scalable Tree Boosting System*,
  KDD. https://doi.org/10.1145/2939672.2939785

## 7. Fairness / excluded attributes

Ethnicity, marital status, income, dwelling, household composition, vehicle
ownership, and credit-class proxies are **excluded from the model** by design.
Using protected or proxy attributes to target retention raises legal and ethical
risk and is not actionable.

- [17] Barocas, S., Hardt, M. & Narayanan, A. (2019), *Fairness and Machine
  Learning: Limitations and Opportunities*. https://fairmlbook.org/
- [18] FTC (2021), *Aiming for truth, fairness, and equity in your company's use of
  AI*. https://www.ftc.gov/business-guidance/blog/2021/04/aiming-truth-fairness-equity-your-companys-use-ai

---

*All dollar figures in the deck are explicit, sourced assumptions presented as
bear / base / bull scenarios. They are decision-support estimates, not guarantees,
and should be validated with a live A/B retention test (see "Next Steps").*
