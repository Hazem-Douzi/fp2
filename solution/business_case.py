"""Value-based targeting (EVaR) + retention ROI business case."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RNG = 42
DATA = "/home/ubuntu/fp2/solution/telecom"
client = pd.read_csv(f"{DATA}/Client.csv")
record = pd.read_csv(f"{DATA}/Record.csv")
df = record.merge(client, on="Customer_ID", how="inner")

y = df["churn"].astype(int)
rev = df["rev_Mean"].fillna(df["rev_Mean"].median())
X = df.drop(columns=["churn", "Customer_ID"])
for c in [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "str"]:
    X[c] = pd.factorize(X[c].astype(str))[0]
X = X.apply(pd.to_numeric, errors="coerce")

idx = np.arange(len(X))
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, idx, test_size=0.30, random_state=RNG, stratify=y)
pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                 ("clf", XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5,
                                       subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                       eval_metric="auc", random_state=RNG, n_jobs=-1))])
pipe.fit(Xtr, ytr)
proba = pipe.predict_proba(Xte)[:, 1]
rev_te = rev.values[ite]
yte_v = yte.values

out = {}

def capture_by_score(score, frac):
    k = int(np.ceil(len(score) * frac))
    order = np.argsort(-score)[:k]
    churn_capture = yte_v[order].sum() / yte_v.sum()
    base = yte_v.mean()
    lift = yte_v[order].mean() / base
    # revenue-at-risk captured = churner-revenue in the targeted set / total churner-revenue
    total_churn_rev = (rev_te * yte_v).sum()
    capt_churn_rev = (rev_te[order] * yte_v[order]).sum() / total_churn_rev
    return round(float(churn_capture), 4), round(float(lift), 4), round(float(capt_churn_rev), 4)

evar = proba * rev_te  # expected value at risk
print("frac | churn-score targeting           | EVaR (P*rev) targeting")
print("     | capture  lift   rev@risk capt    | capture  lift   rev@risk capt")
for f in [0.10, 0.20, 0.30]:
    cc, lf, rr = capture_by_score(proba, f)
    cc2, lf2, rr2 = capture_by_score(evar, f)
    print(f"{f:.2f} |  {cc:.3f}  {lf:.3f}   {rr:.3f}          |  {cc2:.3f}  {lf2:.3f}   {rr2:.3f}")
    out[f"score_top{int(f*100)}"] = {"capture": cc, "lift": lf, "rev_at_risk_capture": rr}
    out[f"evar_top{int(f*100)}"] = {"capture": cc2, "lift": lf2, "rev_at_risk_capture": rr2}

# ---------------- Revenue-centric ROI scenario (per 100,000 customers, annual) ----------
N = 100_000                 # active customer book (matches dataset; scales linearly)
ARPU_m = 58.7               # $/month, dataset rev_Mean mean
ARPU_y = ARPU_m * 12        # annual revenue per customer
annual_churn = 0.22         # 2%/mo voluntary churn compounding ~22%/yr (cited)
CAC = 350                   # cost per gross add to REPLACE a lost customer (cited $350-400)
margin = 0.33               # service-revenue contribution (Verizon adj EBITDA margin ~33%)
target_frac = 0.20
offer_cost = 50             # avg retention offer / loyalty credit per targeted customer
save_rate = 0.30            # share of targeted at-risk customers retained by the offer

rev_capture = out["evar_top20"]["rev_at_risk_capture"]   # 0.448  share of revenue-at-risk caught
cust_capture = out["evar_top20"]["capture"]              # 0.235  share of churners caught (by count)

churners = N * annual_churn
targeted = int(N * target_frac)
total_rev_at_risk = churners * ARPU_y                    # annual revenue exposed to churn

# Revenue retained = revenue-at-risk reached by EVaR targeting x save rate
rev_reached = total_rev_at_risk * rev_capture
rev_protected = rev_reached * save_rate
margin_protected = rev_protected * margin
# Customers actually saved (count) -> avoided reacquisition cost
churners_reached = churners * cust_capture
saved = churners_reached * save_rate
avoided_cac = saved * CAC
campaign_cost = targeted * offer_cost
gross_benefit = rev_protected + avoided_cac
net_benefit = gross_benefit - campaign_cost
roi = net_benefit / campaign_cost

# Baselines: random targeting vs naive churn-probability targeting (same 20% budget)
rev_protected_random = total_rev_at_risk * target_frac * save_rate
rev_protected_churnrank = total_rev_at_risk * out["score_top20"]["rev_at_risk_capture"] * save_rate

roi_out = {
    "N": N, "ARPU_month": ARPU_m, "ARPU_year": round(ARPU_y, 2),
    "annual_churn": annual_churn, "CAC": CAC, "margin": margin,
    "target_frac": target_frac, "offer_cost": offer_cost, "save_rate": save_rate,
    "evar_rev_capture_top20": rev_capture, "evar_cust_capture_top20": cust_capture,
    "total_churners": round(churners),
    "targeted_customers": targeted,
    "total_rev_at_risk": round(total_rev_at_risk),
    "rev_reached_by_targeting": round(rev_reached),
    "saved_customers": round(saved),
    "rev_protected_1yr": round(rev_protected),
    "margin_protected_1yr": round(margin_protected),
    "avoided_cac": round(avoided_cac),
    "campaign_cost": round(campaign_cost),
    "net_benefit": round(net_benefit),
    "roi": round(roi, 3),
    "rev_protected_random": round(rev_protected_random),
    "rev_protected_churnrank": round(rev_protected_churnrank),
    "evar_vs_random_x": round(rev_protected / rev_protected_random, 2),
    "evar_vs_churnrank_x": round(rev_protected / rev_protected_churnrank, 2),
}
print("\n=== Revenue-centric ROI scenario (per 100,000 customers, annual) ===")
for k, v in roi_out.items():
    print(f"  {k:28s} {v}")

# Scaled to a mid-size carrier (stated assumption)
for base in [2_000_000, 10_000_000]:
    factor = base / N
    print(f"  scaled to {base:,} subs: rev_protected=${rev_protected*factor:,.0f}  net=${net_benefit*factor:,.0f}")
roi_out["rev_protected_2M"] = round(rev_protected * (2_000_000 / N))
roi_out["net_benefit_2M"] = round(net_benefit * (2_000_000 / N))

# sensitivity table: net benefit vs save_rate x offer_cost
print("\n=== Sensitivity: net benefit ($) by save_rate (rows) x offer_cost (cols) ===")
sens = {}
for sr in [0.20, 0.30, 0.40]:
    row = {}
    for oc in [30, 50, 75]:
        rp = total_rev_at_risk * rev_capture * sr
        sv = churners * cust_capture * sr
        nb = rp + sv * CAC - targeted * oc
        row[oc] = round(nb)
    sens[sr] = row
    print(f"  save={sr:.0%}: " + "  ".join(f"offer${oc}->${v:,}" for oc, v in row.items()))

out["roi"] = roi_out
out["sensitivity"] = {str(k): v for k, v in sens.items()}
with open("/home/ubuntu/fp2/solution/business_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved business_results.json")
