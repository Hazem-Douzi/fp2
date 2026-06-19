"""Model training & comparison for Company A churn PoC."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
RNG = 42
DATA = "/home/ubuntu/fp2/solution/telecom"

client = pd.read_csv(f"{DATA}/Client.csv")
record = pd.read_csv(f"{DATA}/Record.csv")
df = record.merge(client, on="Customer_ID", how="inner")

y = df["churn"].astype(int)
X = df.drop(columns=["churn", "Customer_ID"])

# Label-encode object/str columns (keep NaN as its own category)
obj_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "str"]
for c in obj_cols:
    X[c] = pd.factorize(X[c].astype(str))[0]
X = X.apply(pd.to_numeric, errors="coerce")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=RNG, stratify=y)

results = {}

def lift_at(y_true, y_score, frac):
    """Cumulative lift & recall in the top `frac` of customers ranked by score."""
    n = len(y_true)
    k = int(np.ceil(n * frac))
    order = np.argsort(-y_score)
    top = np.asarray(y_true)[order][:k]
    base = np.mean(y_true)
    capture = top.sum() / np.asarray(y_true).sum()  # recall of churners
    precision = top.mean()
    lift = precision / base
    return round(float(lift), 3), round(float(capture), 3), round(float(precision), 3)

def evaluate(name, model, scale=False):
    if scale:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", model)])
    else:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", model)])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba)
    acc = accuracy_score(yte, pred)
    l10, c10, p10 = lift_at(yte, proba, 0.10)
    l20, c20, p20 = lift_at(yte, proba, 0.20)
    results[name] = {"auc": round(float(auc), 4), "accuracy": round(float(acc), 4),
                     "lift_top10": l10, "capture_top10": c10, "prec_top10": p10,
                     "lift_top20": l20, "capture_top20": c20, "prec_top20": p20}
    print(f"{name:22s} AUC={auc:.4f}  acc={acc:.4f}  lift@10%={l10}  capture@10%={c10}  lift@20%={l20}  capture@20%={c20}")
    return pipe, proba

print("=== Model comparison (held-out 30% test) ===")
lr_pipe, lr_p = evaluate("LogisticRegression", LogisticRegression(max_iter=2000, C=1.0), scale=True)
rf_pipe, rf_p = evaluate("RandomForest", RandomForestClassifier(
    n_estimators=400, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=RNG))
xgb_pipe, xgb_p = evaluate("XGBoost", XGBClassifier(
    n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=2.0, eval_metric="auc", random_state=RNG, n_jobs=-1))
lgbm_pipe, lgbm_p = evaluate("LightGBM", LGBMClassifier(
    n_estimators=800, learning_rate=0.03, num_leaves=48, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=2.0, random_state=RNG, n_jobs=-1, verbose=-1))

# 5-fold CV AUC for the best (LightGBM/XGB) for robustness
cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
xgb_cv = cross_val_score(XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="auc",
    random_state=RNG, n_jobs=-1), X.fillna(X.median()), y, cv=cv, scoring="roc_auc")
print(f"\nXGBoost 5-fold CV AUC: {xgb_cv.mean():.4f} +/- {xgb_cv.std():.4f}")
results["xgb_cv_auc_mean"] = round(float(xgb_cv.mean()), 4)
results["xgb_cv_auc_std"] = round(float(xgb_cv.std()), 4)

# Feature importance from XGBoost (gain)
booster = xgb_pipe.named_steps["clf"]
imp = pd.Series(booster.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Top 20 features (XGBoost gain importance) ===")
print(imp.head(20))
results["top_features"] = {k: round(float(v), 4) for k, v in imp.head(20).items()}

with open("/home/ubuntu/fp2/solution/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved model_results.json")
