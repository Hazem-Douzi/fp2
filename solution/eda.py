"""Comprehensive EDA for Company A telecom churn dataset."""
import json
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

DATA = "/home/ubuntu/fp2/solution/telecom"
client = pd.read_csv(f"{DATA}/Client.csv")
record = pd.read_csv(f"{DATA}/Record.csv")
df = record.merge(client, on="Customer_ID", how="inner")
print("client", client.shape, "record", record.shape, "merged", df.shape)

stats = {}
stats["n_rows"] = int(df.shape[0])
stats["n_cols"] = int(df.shape[1])

# churn balance
vc = df["churn"].value_counts(dropna=False)
print("\nchurn value_counts:\n", vc)
print("\nchurn proportion:\n", df["churn"].value_counts(normalize=True))
stats["churn_rate"] = float(df["churn"].mean())
stats["n_churn"] = int((df["churn"] == 1).sum())
stats["n_stay"] = int((df["churn"] == 0).sum())

# dtypes
print("\ndtypes counts:\n", df.dtypes.value_counts())
obj_cols = df.select_dtypes(include="object").columns.tolist()
print("\nobject cols:", obj_cols)

# missingness
miss = df.isna().sum().sort_values(ascending=False)
miss = miss[miss > 0]
print("\nColumns with missing values (count, pct):")
for c, v in miss.items():
    print(f"  {c:20s} {v:7d}  {100*v/len(df):5.1f}%")
stats["missing"] = {c: int(v) for c, v in miss.items()}

# key revenue / arpu
print("\n--- Revenue / ARPU ---")
for c in ["rev_Mean", "totmrc_Mean", "avgrev", "totrev", "avg6rev", "mou_Mean", "months", "eqpdays"]:
    print(f"  {c:14s} mean={df[c].mean():10.2f} median={df[c].median():10.2f} "
          f"p10={df[c].quantile(.1):9.2f} p90={df[c].quantile(.9):10.2f} na={df[c].isna().sum()}")
stats["arpu_mean"] = float(df["rev_Mean"].mean())
stats["arpu_median"] = float(df["rev_Mean"].median())
stats["mou_mean"] = float(df["mou_Mean"].mean())
stats["months_mean"] = float(df["months"].mean())
stats["months_median"] = float(df["months"].median())

# numeric churn-driver comparison: mean by churn class + simple correlation
num = df.select_dtypes(include=[np.number]).drop(columns=["Customer_ID"])
corr = num.corrwith(df["churn"]).drop("churn").sort_values()
print("\n--- Top NEGATIVE corr with churn (lower => churn) ---")
print(corr.head(15))
print("\n--- Top POSITIVE corr with churn (higher => churn) ---")
print(corr.tail(15))
stats["corr_top_pos"] = {k: round(float(v), 4) for k, v in corr.tail(15).items()}
stats["corr_top_neg"] = {k: round(float(v), 4) for k, v in corr.head(15).items()}

# eqpdays comparison
print("\n--- eqpdays by churn ---")
print(df.groupby("churn")["eqpdays"].mean())

# helper: churn rate by binned numeric feature
def churn_by_bins(col, q=5):
    s = df[[col, "churn"]].dropna()
    try:
        s["bin"] = pd.qcut(s[col], q, duplicates="drop")
    except Exception:
        return None
    g = s.groupby("bin", observed=True)["churn"].agg(["mean", "count"])
    return g

for col in ["eqpdays", "months", "change_mou", "mou_Mean", "rev_Mean",
            "custcare_Mean", "ovrrev_Mean", "drop_blk_Mean", "hnd_price",
            "totmrc_Mean", "recv_vce_Mean", "avg6mou", "actvsubs", "uniqsubs"]:
    g = churn_by_bins(col)
    if g is not None:
        print(f"\n--- churn rate by {col} quintile ---")
        print(g)

# categorical churn rates
print("\n=== CATEGORICAL churn rates ===")
cat_summary = {}
for col in ["crclscod", "asl_flag", "new_cell", "refurb_new", "hnd_webcap",
            "area", "prizm_social_one", "dualband", "marital", "creditcd",
            "ethnic", "ownrent", "dwlltype"]:
    if col in df.columns:
        g = df.groupby(col, dropna=False)["churn"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        g = g[g["count"] >= 200]
        print(f"\n--- {col} (>=200 obs) ---")
        print(g.head(12))

# asl_flag, new_cell, refurb churn
for col in ["asl_flag", "new_cell", "refurb_new"]:
    g = df.groupby(col)["churn"].mean()
    stats[f"churn_by_{col}"] = {str(k): round(float(v), 4) for k, v in g.items()}

with open("/home/ubuntu/fp2/solution/eda_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("\nSaved eda_stats.json")
