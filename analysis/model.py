import pandas as pd, numpy as np, json, warnings, os
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

OUT = "/vercel/share/v0-project/public/charts"
os.makedirs(OUT, exist_ok=True)
DATA = "/vercel/share/v0-project/analysis"
BASE = "/vercel/share/v0-project/assignment/05. Final Assignment (due Jun 19th 11AM UTC)/Final Assignment/telecom"

client = pd.read_csv(f"{BASE}/Client.csv")
record = pd.read_csv(f"{BASE}/Record.csv")
df = record.merge(client, on="Customer_ID", how="inner")
y = df["churn"].astype(int)
df = df.drop(columns=["churn", "Customer_ID"])

# split numeric / categorical
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print("num", len(num_cols), "cat", len(cat_cols))

# limit high-cardinality categoricals
keep_cat = [c for c in cat_cols if df[c].nunique() <= 60]
X_num = df[num_cols].copy()
X_cat = pd.get_dummies(df[keep_cat].astype("object"), dummy_na=True)
X = pd.concat([X_num, X_cat], axis=1)
print("X shape", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_test_i = imp.transform(X_test)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_i)
X_test_s = scaler.transform(X_test_i)

results = {}

# 1) Logistic Regression baseline
lr = LogisticRegression(max_iter=1000, C=0.5)
lr.fit(X_train_s, y_train)
p_lr = lr.predict_proba(X_test_s)[:,1]
results["LogisticRegression"] = {
  "auc": roc_auc_score(y_test, p_lr),
  "accuracy": accuracy_score(y_test, (p_lr>0.5).astype(int)),
  "precision": precision_score(y_test, (p_lr>0.5).astype(int)),
  "recall": recall_score(y_test, (p_lr>0.5).astype(int)),
  "f1": f1_score(y_test, (p_lr>0.5).astype(int)),
}

# 2) Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=42)
rf.fit(X_train_i, y_train)
p_rf = rf.predict_proba(X_test_i)[:,1]
results["RandomForest"] = {
  "auc": roc_auc_score(y_test, p_rf),
  "accuracy": accuracy_score(y_test, (p_rf>0.5).astype(int)),
  "precision": precision_score(y_test, (p_rf>0.5).astype(int)),
  "recall": recall_score(y_test, (p_rf>0.5).astype(int)),
  "f1": f1_score(y_test, (p_rf>0.5).astype(int)),
}

# 3) XGBoost
xgbc = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="auc", n_jobs=-1, random_state=42)
xgbc.fit(X_train_i, y_train)
p_xgb = xgbc.predict_proba(X_test_i)[:,1]
results["XGBoost"] = {
  "auc": roc_auc_score(y_test, p_xgb),
  "accuracy": accuracy_score(y_test, (p_xgb>0.5).astype(int)),
  "precision": precision_score(y_test, (p_xgb>0.5).astype(int)),
  "recall": recall_score(y_test, (p_xgb>0.5).astype(int)),
  "f1": f1_score(y_test, (p_xgb>0.5).astype(int)),
}

for k,v in results.items():
    print(k, {kk: round(vv,4) for kk,vv in v.items()})

# Best model = XGBoost. Feature importance.
fi = pd.Series(xgbc.feature_importances_, index=X.columns).sort_values(ascending=False)
top_fi = fi.head(15)
print("\nTOP FEATURES:\n", top_fi)

# ---- Lift / decile analysis on XGBoost ----
order = np.argsort(-p_xgb)
y_sorted = y_test.values[order]
n = len(y_sorted)
deciles = np.array_split(np.arange(n), 10)
base = y_test.mean()
lift_rows = []
cum_churn = 0
for i, idx in enumerate(deciles):
    seg_rate = y_sorted[idx].mean()
    lift_rows.append({"decile": i+1, "churn_rate": float(seg_rate), "lift": float(seg_rate/base)})
top_decile_lift = lift_rows[0]["lift"]
# cumulative capture in top 3 deciles
top3_capture = y_sorted[:deciles[0].size+deciles[1].size+deciles[2].size].sum()/y_sorted.sum()
print("\nTop decile lift:", round(top_decile_lift,2), "Top-30% capture:", round(top3_capture,3))

# ===== CHARTS =====
plt.rcParams.update({"figure.dpi":120, "font.size":11})
TEAL="#0d9488"; SLATE="#334155"; AMBER="#d97706"; RED="#dc2626"

# Chart 1: churn balance
fig, ax = plt.subplots(figsize=(5,3.2))
vc = y.value_counts().sort_index()
ax.bar(["Retained","Churned"], vc.values, color=[SLATE, RED])
ax.set_title("Churn distribution (modeling sample)")
for i,v in enumerate(vc.values): ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(f"{OUT}/churn_balance.png"); plt.close()

# Chart 2: ROC curves
fig, ax = plt.subplots(figsize=(5,4))
for name,p,c in [("Logistic",p_lr,SLATE),("Random Forest",p_rf,AMBER),("XGBoost",p_xgb,TEAL)]:
    fpr,tpr,_ = roc_curve(y_test,p)
    ax.plot(fpr,tpr,label=f"{name} (AUC={roc_auc_score(y_test,p):.3f})",color=c,lw=2)
ax.plot([0,1],[0,1],"--",color="#94a3b8")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves — model comparison"); ax.legend(loc="lower right",fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/roc.png"); plt.close()

# Chart 3: feature importance
fig, ax = plt.subplots(figsize=(6,4.5))
top_fi[::-1].plot(kind="barh", ax=ax, color=TEAL)
ax.set_title("Top 15 churn drivers (XGBoost importance)")
plt.tight_layout(); plt.savefig(f"{OUT}/feature_importance.png"); plt.close()

# Chart 4: lift by decile
fig, ax = plt.subplots(figsize=(6,3.6))
ds=[r["decile"] for r in lift_rows]; ls=[r["lift"] for r in lift_rows]
ax.bar(ds, ls, color=TEAL); ax.axhline(1,color=RED,ls="--",label="Random (lift=1)")
ax.set_xlabel("Risk decile (1=highest risk)"); ax.set_ylabel("Lift vs average")
ax.set_title("Model lift by risk decile"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/lift.png"); plt.close()

# Chart 5: eqpdays vs churn (binned)
dfb = pd.DataFrame({"eqpdays": df["eqpdays"], "churn": y})
dfb["bin"] = pd.cut(dfb["eqpdays"], bins=[0,150,300,450,600,2000],
                    labels=["0-150","150-300","300-450","450-600","600+"])
g = dfb.groupby("bin")["churn"].mean()
fig, ax = plt.subplots(figsize=(5.5,3.4))
ax.plot(g.index.astype(str), g.values*100, marker="o", color=RED, lw=2)
ax.set_xlabel("Equipment age (days)"); ax.set_ylabel("Churn rate (%)")
ax.set_title("Churn rises with handset age")
plt.tight_layout(); plt.savefig(f"{OUT}/eqpdays_churn.png"); plt.close()

# Chart 6: churn by months in service (tenure)
dft = pd.DataFrame({"months": df["months"], "churn": y})
dft["bin"] = pd.cut(dft["months"], bins=[0,6,12,18,24,36,70],
                    labels=["6-12","12","12-18","18-24","24-36","36+"])
g2 = dft.groupby("bin")["churn"].mean()
fig, ax = plt.subplots(figsize=(5.5,3.4))
ax.plot(g2.index.astype(str), g2.values*100, marker="o", color=AMBER, lw=2)
ax.set_xlabel("Tenure (months in service)"); ax.set_ylabel("Churn rate (%)")
ax.set_title("Churn by customer tenure")
plt.tight_layout(); plt.savefig(f"{OUT}/tenure_churn.png"); plt.close()

summary = {
  "results": {k:{kk:round(vv,4) for kk,vv in v.items()} for k,v in results.items()},
  "best_model": "XGBoost",
  "top_features": [{"feature":f,"importance":round(float(v),4)} for f,v in top_fi.items()],
  "lift": lift_rows,
  "top_decile_lift": round(top_decile_lift,3),
  "top30_capture": round(float(top3_capture),3),
  "base_churn": round(float(base),4),
  "eqpdays_churn": {str(k):round(float(v),4) for k,v in g.items()},
  "tenure_churn": {str(k):round(float(v),4) for k,v in g2.items()},
}
with open(f"{DATA}/model_results.json","w") as f:
    json.dump(summary, f, indent=2)
print("\nSAVED model_results.json and 6 charts")
