import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

BASE = "/vercel/share/v0-project/assignment/05. Final Assignment (due Jun 19th 11AM UTC)/Final Assignment/telecom"
client = pd.read_csv(f"{BASE}/Client.csv")
record = pd.read_csv(f"{BASE}/Record.csv")
print("client", client.shape, "record", record.shape)

df = record.merge(client, on="Customer_ID", how="inner", suffixes=("", "_c"))
print("merged", df.shape)
print("churn dtype/values:", df["churn"].value_counts(dropna=False).to_dict())

# overall churn rate
churn_rate = df["churn"].mean()
print("OVERALL CHURN RATE:", round(churn_rate*100, 2), "%")

# revenue figures
print("rev_Mean describe:\n", df["rev_Mean"].describe())
print("totrev describe:\n", df["totrev"].describe())
print("months describe:\n", df["months"].describe())
print("eqpdays describe:\n", df["eqpdays"].describe())

# missingness top
miss = df.isna().mean().sort_values(ascending=False)
print("\nTOP MISSING:\n", (miss[miss>0].head(25)*100).round(1))

# numeric correlation w/ churn
num = df.select_dtypes(include=[np.number]).copy()
corr = num.corr()["churn"].drop("churn").sort_values()
print("\nMOST NEGATIVE corr w churn:\n", corr.head(15).round(4))
print("\nMOST POSITIVE corr w churn:\n", corr.tail(15).round(4))

out = {
  "n_rows": int(df.shape[0]),
  "n_cols": int(df.shape[1]),
  "churn_rate": float(churn_rate),
  "rev_mean_avg": float(df["rev_Mean"].mean()),
  "rev_mean_median": float(df["rev_Mean"].median()),
  "months_mean": float(df["months"].mean()),
  "eqpdays_mean": float(df["eqpdays"].mean()),
  "totrev_mean": float(df["totrev"].mean()),
}
print("\nSUMMARY_JSON", json.dumps(out))
df.to_parquet("/vercel/share/v0-project/analysis/merged.parquet")
print("saved merged.parquet")
