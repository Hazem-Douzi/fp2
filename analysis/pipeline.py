"""
Company A churn — single reproducible pipeline.
Fixes applied vs the earlier draft:
  * PROTECTED attributes (ethnicity, marital status, income, credit class) are DROPPED,
    and we measure the AUC cost of doing so (fairness A/B).
  * High-missing junk columns (>20% NA) are removed.
  * Real engineered behavioural features (usage/revenue trends, overage share,
    dropped-call rate, care intensity, handset months).
  * Probabilities are calibrated to the assumed population base rate via prior-shift
    correction, so P(churn) and EVaR live on a real scale.
  * Real K-means segmentation (not hand-coded thresholds).
  * Profit-curve threshold optimisation (not a fixed 0.5 / fixed 20%).
  * SHAP explanations for the winning model.
  * Bear / base / bull business scenarios with explicit, sourced assumptions.

Run:  python analysis/pipeline.py        (set TELECOM_DATA_DIR to override data location)
Outputs: analysis/results.json  (single source of truth for deck + notebook)
"""
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------- locate data
def find_data():
    cand = []
    env = os.environ.get("TELECOM_DATA_DIR")
    if env: cand.append(env)
    cand += [".", "data", "../data",
             "assignment/05. Final Assignment (due Jun 19th 11AM UTC)/Final Assignment/telecom"]
    for d in cand:
        if d and os.path.exists(os.path.join(d, "Client.csv")) and os.path.exists(os.path.join(d, "Record.csv")):
            return d
    raise FileNotFoundError("Client.csv / Record.csv not found. Set TELECOM_DATA_DIR.")

DATA = find_data()
client = pd.read_csv(os.path.join(DATA, "Client.csv"))
record = pd.read_csv(os.path.join(DATA, "Record.csv"))
df = client.merge(record, on="Customer_ID")
R = {"seed": SEED}
R["n_rows"], R["n_cols_raw"] = int(df.shape[0]), int(df.shape[1])
R["sample_churn_rate"] = round(df["churn"].mean(), 4)

# ----------------------------------------------------------------- EDA facts
R["ARPU_month"] = round(df["rev_Mean"].mean(), 2)
R["median_tenure"] = float(df["months"].median())
R["mean_tenure"] = round(df["months"].mean(), 1)
R["mean_eqpdays"] = round(df["eqpdays"].mean(), 0)

# Real handset-age bins IN MONTHS (eqpdays/30) -- honest labels
df["handset_months"] = df["eqpdays"] / 30.0
edges = [0, 6, 12, 18, 24, np.inf]
labels = ["0-6mo", "6-12mo", "12-18mo", "18-24mo", "24mo+"]
df["hs_bin"] = pd.cut(df["handset_months"], bins=edges, labels=labels, right=False)
hs = df.groupby("hs_bin", observed=True)["churn"].agg(["mean", "size"])
R["handset_bins"] = {str(k): {"churn": round(v["mean"], 4), "n": int(v["size"])} for k, v in hs.iterrows()}

# correlations (numeric only) for context
num = df.select_dtypes("number").drop(columns=["churn"], errors="ignore")
corr = num.corrwith(df["churn"]).dropna().sort_values()
R["corr_neg"] = {k: round(v, 3) for k, v in corr.head(6).items()}
R["corr_pos"] = {k: round(v, 3) for k, v in corr.tail(6).items()[::-1]} if False else {k: round(v, 3) for k, v in corr.tail(6)[::-1].items()}

# ----------------------------------------------------------------- feature engineering
def eps(s): return s.replace(0, np.nan)
df["usage_trend"]   = df["change_mou"]  / (df["mou_Mean"].abs() + 1)
df["revenue_trend"] = df["change_rev"]  / (df["rev_Mean"].abs() + 1)
df["overage_share"] = df["ovrrev_Mean"] / (df["rev_Mean"].abs() + 1)
df["dropblk_rate"]  = df["drop_blk_Mean"] / (df["attempt_Mean"].abs() + 1)
df["care_per_mou"]  = df["custcare_Mean"] / (df["mou_Mean"].abs() + 1)
df["rev_per_sub"]   = df["rev_Mean"] / (df["uniqsubs"].abs() + 1)
ENGINEERED = ["handset_months","usage_trend","revenue_trend","overage_share",
              "dropblk_rate","care_per_mou","rev_per_sub"]

# ----------------------------------------------------------------- column hygiene
PROTECTED = ["ethnic", "marital", "income", "crclscod"]   # dropped for fairness
miss = df.isna().mean()
HIGH_MISSING = [c for c in df.columns if miss[c] > 0.20 and c not in ["churn"]]
DROP_ID = ["Customer_ID"]
R["protected_dropped"] = PROTECTED
R["high_missing_dropped"] = sorted(HIGH_MISSING)
R["engineered_features"] = ENGINEERED

def build_matrix(frame, include_protected):
    drop = set(DROP_ID + HIGH_MISSING)
    if not include_protected:
        drop |= set(PROTECTED)
    X = frame.drop(columns=[c for c in drop if c in frame.columns] + ["churn"], errors="ignore")
    # one-hot encode remaining low-cardinality categoricals
    cat = [c for c in X.columns if X[c].dtype == "object"]
    cat = [c for c in cat if X[c].nunique() <= 20]
    X = X.drop(columns=[c for c in X.columns if X[c].dtype == "object" and c not in cat], errors="ignore")
    X = pd.get_dummies(X, columns=cat, dummy_na=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True)).fillna(0)
    return X

y = df["churn"].astype(int)

# ----------------------------------------------------------------- fairness A/B
def auc_for(include_protected):
    X = build_matrix(df, include_protected)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                      random_state=SEED, n_jobs=-1)
    m.fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

auc_with = auc_for(True)
auc_without = auc_for(False)
R["fairness"] = {"auc_with_protected": round(auc_with, 4),
                 "auc_without_protected": round(auc_without, 4),
                 "auc_delta": round(auc_with - auc_without, 4)}

# ----------------------------------------------------------------- main models (protected REMOVED)
X = build_matrix(df, include_protected=False)
R["n_features_model"] = int(X.shape[1])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=14,
                                             min_samples_leaf=20, random_state=SEED, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.04,
                             subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                             random_state=SEED, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=600, max_depth=-1, num_leaves=48,
                               learning_rate=0.03, subsample=0.9, colsample_bytree=0.9,
                               random_state=SEED, n_jobs=-1, verbose=-1),
}
# Logistic needs scaling
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

res = {}
proba = {}
for name, m in models.items():
    if name == "Logistic Regression":
        m.fit(Xtr_s, ytr); p = m.predict_proba(Xte_s)[:, 1]
    else:
        m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    res[name] = {"auc": round(roc_auc_score(yte, p), 4),
                 "acc": round(accuracy_score(yte, pred), 4),
                 "prec": round(precision_score(yte, pred), 4),
                 "rec": round(recall_score(yte, pred), 4),
                 "f1": round(f1_score(yte, pred), 4)}
    proba[name] = p
R["models"] = res
best_name = max(res, key=lambda k: res[k]["auc"])
R["best_model"] = best_name

# 5-fold CV for the winner
best = models[best_name]
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
Xcv = Xtr_s if best_name == "Logistic Regression" else Xtr
cv = cross_val_score(best, Xcv, ytr, cv=skf, scoring="roc_auc", n_jobs=-1)
R["cv_auc_mean"], R["cv_auc_std"] = round(cv.mean(), 4), round(cv.std(), 4)

p_best = proba[best_name]

# ----------------------------------------------------------------- calibration (reliability on sample scale)
frac_pos, mean_pred = calibration_curve(yte, p_best, n_bins=10, strategy="quantile")
R["calibration"] = {"mean_pred": [round(x, 4) for x in mean_pred],
                    "frac_pos": [round(x, 4) for x in frac_pos]}

# prior-shift correction: sample prior 0.5 -> assumed population prior PI
PI = 0.22
def prior_shift(p, pi, p_sample=0.5):
    odds = (p / (1 - p)) * (pi / (1 - pi)) / (p_sample / (1 - p_sample))
    return odds / (1 + odds)
p_pop = prior_shift(np.clip(p_best, 1e-6, 1 - 1e-6), PI)
R["assumed_population_churn_annual"] = PI
R["pop_prob_mean"] = round(float(p_pop.mean()), 4)

# ----------------------------------------------------------------- targeting (rank by EVaR vs score vs random)
rev_med = df["rev_Mean"].median()
annual_rev = (df.loc[yte.index, "rev_Mean"].fillna(rev_med).clip(lower=0).values * 12)
evar = p_pop * annual_rev
order = {"by_evar": np.argsort(-evar),
         "by_score": np.argsort(-p_best),
         "by_random": np.random.RandomState(SEED).permutation(len(p_best))}
yte_arr = yte.values
tot_churn = yte_arr.sum()
tot_rev_risk = annual_rev[yte_arr == 1].sum()
def capture(idx, frac):
    k = int(len(idx) * frac); sel = idx[:k]
    return {"cust_capture": round(yte_arr[sel].sum() / tot_churn, 4),
            "rev_capture": round(annual_rev[sel][yte_arr[sel] == 1].sum() / tot_rev_risk, 4)}
R["targeting"] = {}
for key, idx in order.items():
    R["targeting"][key] = {str(f): capture(idx, f) for f in (0.1, 0.2, 0.3)}

# ----------------------------------------------------------------- profit-curve threshold optimisation
# Population-scaled: expected net value of contacting customer i =
#   save_rate * (margin*annual_rev_i + CAC) * P_pop(churn_i)  -  offer
# Rank by that expected value; the optimum is where marginal EV crosses zero.
N = 1_000_000
margin = 0.50; save_rate = 0.30; offer = 30.0; CAC = 350.0
ev_contact = save_rate * (margin * annual_rev + CAC) * p_pop - offer
ev_sorted = np.sort(ev_contact)[::-1]
n_test = len(ev_sorted)
scale = N / n_test                      # express cumulative net per 1,000,000 subscribers
cum = np.cumsum(ev_sorted) * scale
profit_curve = []
for f in np.arange(0.05, 1.001, 0.05):
    k = max(1, int(n_test * f)) - 1
    profit_curve.append({"frac": round(float(f), 2), "net": round(float(cum[k]))})
R["profit_curve"] = profit_curve
R["optimal_contact_frac"] = round(float((ev_contact > 0).mean()), 2)   # share worth contacting
R["optimal_net_1M"] = round(float(cum[max(0, (ev_sorted > 0).sum() - 1)]))

# ----------------------------------------------------------------- real K-means personas
seg_feats = ["handset_months", "rev_Mean", "mou_Mean", "overage_share",
             "usage_trend", "custcare_Mean", "months"]
S = df[seg_feats].apply(pd.to_numeric, errors="coerce").fillna(df[seg_feats].median(numeric_only=True))
Ss = StandardScaler().fit_transform(S)
km = KMeans(n_clusters=4, random_state=SEED, n_init=10)
df["cluster"] = km.fit_predict(Ss)
personas = []
for c in range(4):
    sub = df[df["cluster"] == c]
    personas.append({"cluster": int(c), "n": int(len(sub)),
                     "share": round(len(sub) / len(df), 3),
                     "churn": round(sub["churn"].mean(), 3),
                     "rev": round(sub["rev_Mean"].mean(), 1),
                     "handset_months": round(sub["handset_months"].mean(), 1),
                     "overage_share": round(sub["overage_share"].mean(), 3),
                     "mou": round(sub["mou_Mean"].mean(), 0),
                     "months": round(sub["months"].mean(), 0)})
personas.sort(key=lambda x: -x["churn"])
R["personas"] = personas

# ----------------------------------------------------------------- SHAP (winner)
try:
    import shap
    Xshap = Xte.sample(min(2000, len(Xte)), random_state=SEED)
    expl = shap.TreeExplainer(models[best_name])
    sv = expl.shap_values(Xshap)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = np.abs(sv).mean(0)
    top = pd.Series(imp, index=Xshap.columns).sort_values(ascending=False).head(12)
    R["shap_top"] = {k: round(float(v), 4) for k, v in top.items()}
    np.save("analysis/_shap_values.npy", sv)
    Xshap.reset_index(drop=True).to_csv("analysis/_shap_X.csv", index=False)
except Exception as e:
    R["shap_top"] = {}; R["shap_error"] = str(e)

# ----------------------------------------------------------------- business case (bear/base/bull)
A = R["ARPU_month"]
evar_rev20 = R["targeting"]["by_evar"]["0.2"]["rev_capture"]
evar_cust20 = R["targeting"]["by_evar"]["0.2"]["cust_capture"]
score_rev20 = R["targeting"]["by_score"]["0.2"]["rev_capture"]
churners = N * PI
rev_at_risk = churners * A * 12
def scenario(save, offer, cac, target=0.2):
    rev_prot = rev_at_risk * evar_rev20 * save
    margin_prot = rev_prot * margin
    avoided = churners * evar_cust20 * save * cac
    cost = N * target * offer
    net = margin_prot + avoided - cost
    return {"save_rate": save, "offer": offer, "cac": cac,
            "rev_protected": round(rev_prot), "net_benefit": round(net),
            "roi": round(net / cost, 2), "saved_customers": round(churners * evar_cust20 * save)}
R["business"] = {
    "N": N, "ARPU_month": A, "annual_churn": PI, "margin": margin,
    "target_frac": 0.2, "rev_at_risk": round(rev_at_risk),
    "evar_rev_capture20": evar_rev20, "evar_cust_capture20": evar_cust20,
    "evar_vs_score_x": round(evar_rev20 / score_rev20, 2),
    "bear": scenario(0.15, 40, 300),
    "base": scenario(0.30, 30, 350),
    "bull": scenario(0.40, 25, 400),
}
R["clv"] = round(A * margin / (PI / 12), 0)

# quotation framed as % of value created (base case)
net_base = R["business"]["base"]["net_benefit"]
setup, monthly = 300_000, 45_000
annual_fee = setup + monthly * 12
R["quote"] = {"setup_fee": setup, "monthly_fee": monthly, "annual_fee": annual_fee,
              "net_value_1M": net_base, "pct_of_net": round(annual_fee / net_base * 100, 1),
              "client_keeps": round(net_base - annual_fee)}

with open("analysis/results.json", "w") as f:
    json.dump(R, f, indent=2)

print("=== SUMMARY ===")
print("rows", R["n_rows"], "| model features", R["n_features_model"])
print("FAIRNESS auc with/without protected:", R["fairness"])
for k, v in res.items(): print(" ", k, v["auc"])
print("best", best_name, "CV", R["cv_auc_mean"], "+/-", R["cv_auc_std"])
print("EVaR rev capture@20%", evar_rev20, "vs score", score_rev20, "=", R["business"]["evar_vs_score_x"], "x")
print("optimal contact frac", R["optimal_contact_frac"], "net@opt", R["optimal_net_1M"])
print("BASE net", R["business"]["base"]["net_benefit"], "ROI", R["business"]["base"]["roi"])
print("personas churn:", [p["churn"] for p in personas])
print("shap top:", list(R["shap_top"].items())[:5])
print("handset bins:", R["handset_bins"])
print("DONE -> analysis/results.json")
