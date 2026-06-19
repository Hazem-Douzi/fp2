"""Segment the at-risk pool into actionable intervention personas."""
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
X = df.drop(columns=["churn", "Customer_ID"]).copy()
for c in [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "str"]:
    X[c] = pd.factorize(X[c].astype(str))[0]
X = X.apply(pd.to_numeric, errors="coerce")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=RNG, stratify=y)
pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                 ("clf", XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5,
                                       subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                       eval_metric="auc", random_state=RNG, n_jobs=-1))])
pipe.fit(Xtr, ytr)
df["churn_prob"] = pipe.predict_proba(X)[:, 1]

# High-risk pool = top 30% by predicted churn probability
thr = df["churn_prob"].quantile(0.70)
pool = df[df["churn_prob"] >= thr].copy()
print(f"At-risk pool (top 30% by churn prob): {len(pool):,} customers, "
      f"actual churn rate {pool['churn'].mean():.1%} (book {df['churn'].mean():.1%})")

med_mou = df["mou_Mean"].median()

def persona(r):
    # priority order
    if (r.get("eqpdays", 0) >= 365) or (str(r.get("refurb_new")) == "R") or (pd.isna(r.get("hnd_webcap"))):
        return "Aging-Handset"
    if (r.get("ovrrev_Mean", 0) or 0) >= 10:
        return "Bill-Shock / Overage"
    if (pd.notna(r.get("mou_Mean")) and r["mou_Mean"] < med_mou) and ((r.get("change_mou", 0) or 0) < 0):
        return "Disengaging"
    return "Stable-Loyalty"

# use original (un-encoded) columns for rules
orig = record.merge(client, on="Customer_ID", how="inner")
orig["churn_prob"] = df["churn_prob"].values
orig["churn"] = y.values
pool_o = orig[orig["churn_prob"] >= thr].copy()
pool_o["persona"] = pool_o.apply(persona, axis=1)

g = pool_o.groupby("persona").agg(
    customers=("churn", "size"),
    churn_rate=("churn", "mean"),
    avg_rev=("rev_Mean", "mean"),
    avg_eqpdays=("eqpdays", "mean"),
    avg_mou=("mou_Mean", "mean"),
    avg_overage=("ovrrev_Mean", "mean"),
).sort_values("customers", ascending=False)
g["share"] = g["customers"] / g["customers"].sum()
print("\n=== Intervention personas within the at-risk pool ===")
print(g.round(2))

actions = {
    "Aging-Handset": "Proactive device-upgrade offer (financed/subsidized new handset)",
    "Bill-Shock / Overage": "Right-plan move to unlimited / overage alerts & auto-optimize",
    "Disengaging": "Engagement bundle, data/perk incentives, usage win-back",
    "Stable-Loyalty": "Loyalty credit / contract-renewal incentive / proactive check-in",
}
res = {"pool_size": int(len(pool_o)), "pool_churn_rate": round(float(pool_o["churn"].mean()), 4),
       "book_churn_rate": round(float(df["churn"].mean()), 4), "personas": {}}
for p, row in g.iterrows():
    res["personas"][p] = {"customers": int(row["customers"]), "share": round(float(row["share"]), 4),
                          "churn_rate": round(float(row["churn_rate"]), 4),
                          "avg_rev": round(float(row["avg_rev"]), 2),
                          "avg_eqpdays": round(float(row["avg_eqpdays"]), 1),
                          "avg_mou": round(float(row["avg_mou"]), 1),
                          "avg_overage": round(float(row["avg_overage"]), 2),
                          "action": actions[p]}
with open("/home/ubuntu/fp2/solution/persona_results.json", "w") as f:
    json.dump(res, f, indent=2)
print("\nSaved persona_results.json")
