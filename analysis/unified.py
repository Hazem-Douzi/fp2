"""
Company A telecom churn - unified analysis.
Produces ONE consistent JSON (analysis/results.json) + charts (deliverables/figs)
used across the notebook, the deck, and the references.

Run: python analysis/unified.py
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
DATA = os.environ.get("TELECOM_DATA_DIR",
    "assignment/05. Final Assignment (due Jun 19th 11AM UTC)/Final Assignment/telecom")
FIGS = "deliverables/figs"
os.makedirs(FIGS, exist_ok=True)

NAVY="#0f2742"; ORANGE="#e8833a"; TEAL="#2a9d8f"; GRAY="#94a3b8"; LIGHT="#e2e8f0"
plt.rcParams.update({"font.size":12,"axes.edgecolor":"#cbd5e1","axes.linewidth":0.8,
                     "figure.facecolor":"white","axes.facecolor":"white"})

R = {}  # results dict

# ----------------------------------------------------------------- load + merge
client = pd.read_csv(f"{DATA}/Client.csv")
record = pd.read_csv(f"{DATA}/Record.csv")
key = "Customer_ID" if "Customer_ID" in client.columns else client.columns[0]
df = record.merge(client, on=key, how="inner", suffixes=("","_c"))
df = df.loc[:, ~df.columns.duplicated()]
target = "churn"
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])
df[target] = df[target].astype(int)

R["n_customers"] = int(len(df))
R["n_features"] = int(df.shape[1]-1)
R["churn_rate_sample"] = round(float(df[target].mean()),4)

def col(*names):
    for n in names:
        if n in df.columns: return n
    return None

c_rev = col("rev_Mean","avgrev","totrev")
c_mou = col("mou_Mean","avgmou")
c_eqp = col("eqpdays")
c_hnd = col("hnd_price")
c_mon = col("months","eqpdays")
c_mrc = col("totmrc_Mean")
c_ovr = col("ovrrev_Mean","ovrmou_Mean")

R["ARPU_month"] = round(float(df[c_rev].dropna().mean()),2) if c_rev else None
R["tenure_median"] = round(float(df[c_mon].dropna().median()),1) if c_mon else None

# missingness
miss = (df.isna().mean()*100).round(1).sort_values(ascending=False)
R["missing_top"] = {k:float(v) for k,v in miss.head(10).items()}

# ----------------------------------------------------------------- correlations
num = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
corr = num.corrwith(df[target]).dropna().sort_values()
R["corr_neg"] = {k:round(float(v),3) for k,v in corr.head(6).items()}
R["corr_pos"] = {k:round(float(v),3) for k,v in corr.tail(6).items()}

# eqpdays churn by bins
if c_eqp:
    q = pd.qcut(df[c_eqp].rank(method="first"), 5, labels=["<6mo","6-12mo","12-18mo","18-24mo","24mo+"])
    eb = df.groupby(q)[target].mean()
    R["eqp_bins"] = {str(k):round(float(v),3) for k,v in eb.items()}

# categorical levers
def lever(c):
    if c and c in df.columns:
        g = df.groupby(df[c].astype(str))[target].mean()
        return {str(k):round(float(v),3) for k,v in g.items() if g.index.size<=8}
    return None
R["lever_asl"] = lever(col("asl_flag"))
R["lever_refurb"] = lever(col("refurb_new"))
R["lever_web"] = lever(col("creditcd","webcap","WEBCAP"))

# ----------------------------------------------------------------- features
drop = [target, key] + [c for c in df.columns if "id" in c.lower() and df[c].nunique()>len(df)*0.5]
X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
# engineered ratios
if c_ovr and c_rev: X["overage_share"] = (df[c_ovr]/(df[c_rev].abs()+1)).clip(-5,5)
if c_mou and c_eqp: X["mou_per_eqpday"] = (df[c_mou]/(df[c_eqp]+1)).clip(0,50)
y = df[target].values

numcols = X.select_dtypes(include=[np.number]).columns.tolist()
catcols = [c for c in X.columns if c not in numcols and X[c].nunique()<=20]
X = X[numcols+catcols]

pre = ColumnTransformer([
    ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), numcols),
    ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                      ("oh",OneHotEncoder(handle_unknown="ignore",max_categories=12))]), catcols),
])

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=SEED,stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=400,max_depth=14,
                        min_samples_leaf=20,n_jobs=-1,random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=600,max_depth=5,learning_rate=0.03,
                  subsample=0.85,colsample_bytree=0.85,eval_metric="logloss",
                  n_jobs=-1,random_state=SEED),
    "LightGBM": LGBMClassifier(n_estimators=600,max_depth=6,learning_rate=0.03,
                  subsample=0.85,colsample_bytree=0.85,n_jobs=-1,random_state=SEED,verbose=-1),
}

def lift_capture(y_true,p,frac):
    n=int(len(p)*frac); idx=np.argsort(p)[::-1][:n]
    base=y_true.mean()
    cap=y_true[idx].sum()/y_true.sum()
    lift=(y_true[idx].mean()/base) if base>0 else 0
    return round(float(lift),3), round(float(cap),3)

res={}; roc_data={}; best=None; best_auc=-1; best_pipe=None; best_p=None
for name,clf in models.items():
    pipe=Pipeline([("pre",pre),("clf",clf)]); pipe.fit(Xtr,ytr)
    p=pipe.predict_proba(Xte)[:,1]; pred=(p>=0.5).astype(int)
    auc=roc_auc_score(yte,p)
    l20,c20=lift_capture(yte,p,0.20); l30,c30=lift_capture(yte,p,0.30)
    res[name]={"auc":round(float(auc),4),"accuracy":round(float(accuracy_score(yte,pred)),3),
               "precision":round(float(precision_score(yte,pred)),3),
               "recall":round(float(recall_score(yte,pred)),3),
               "f1":round(float(f1_score(yte,pred)),3),
               "lift20":l20,"capture20":c20,"lift30":l30,"capture30":c30}
    fpr,tpr,_=roc_curve(yte,p); roc_data[name]=(fpr,tpr,auc)
    if auc>best_auc: best_auc=auc; best=name; best_pipe=pipe; best_p=p
R["models"]=res; R["best_model"]=best

# CV on best
cv=cross_val_score(best_pipe,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),
                   scoring="roc_auc",n_jobs=-1)
R["cv_auc_mean"]=round(float(cv.mean()),4); R["cv_auc_std"]=round(float(cv.std()),4)

# feature importance (best tree model -> use XGBoost)
xgb_pipe=Pipeline([("pre",pre),("clf",models["XGBoost"])]); xgb_pipe.fit(Xtr,ytr)
fn=xgb_pipe.named_steps["pre"].get_feature_names_out()
imp=xgb_pipe.named_steps["clf"].feature_importances_
fi=sorted(zip(fn,imp),key=lambda t:t[1],reverse=True)[:12]
R["top_features"]={f.split("__")[-1]:round(float(v),4) for f,v in fi}

# ----------------------------------------------------------------- EVaR
rev_te = df.loc[Xte.index, c_rev].fillna(df[c_rev].median()).values if c_rev else np.ones(len(yte))
rev_at_risk_total = (yte*rev_te).sum()
def capture_table(score):
    out={}
    order=np.argsort(score)[::-1]
    for frac in (0.10,0.20,0.30):
        n=int(len(score)*frac); idx=order[:n]
        cust_cap=yte[idx].sum()/yte.sum()
        rev_cap=(yte[idx]*rev_te[idx]).sum()/rev_at_risk_total
        out[str(frac)]={"cust_capture":round(float(cust_cap),4),"rev_capture":round(float(rev_cap),4)}
    return out
evar_score = best_p*rev_te
R["targeting"]={"by_score":capture_table(best_p),"by_evar":capture_table(evar_score)}
rng=np.random.default_rng(SEED)
rand=rng.random(len(yte))
R["targeting"]["by_random"]=capture_table(rand)

# ----------------------------------------------------------------- personas
pool_n=int(len(yte)*0.30)
pool_idx=np.argsort(evar_score)[::-1][:pool_n]
pool=df.loc[Xte.index[pool_idx]].copy()
def safe(c): return pool[c] if c in pool.columns else pd.Series(np.nan,index=pool.index)
eqp=safe(c_eqp); ovr=safe(col("ovrrev_Mean","ovrmou_Mean")); rev=safe(c_rev)
persona=np.where(eqp.fillna(0)>=400,"Aging-Handset",
         np.where(ovr.fillna(0)>=30,"Bill-Shock / Overage",
         np.where(rev.fillna(0)>=df[c_rev].median(),"Stable-Loyalty","Disengaging")))
pool["persona"]=persona
P={}
for name,g in pool.groupby("persona"):
    P[name]={"customers":int(len(g)),"share":round(len(g)/len(pool),3),
             "churn_rate":round(float(g[target].mean()),3),
             "avg_rev":round(float(g[c_rev].mean()),2) if c_rev else None,
             "avg_eqpdays":round(float(g[c_eqp].mean()),0) if c_eqp else None}
R["personas"]=P
R["pool_size_frac"]=0.30

# ----------------------------------------------------------------- business case
# Conservative, defensible model. Target top 20% by EVaR.
#   benefit = (contribution margin of revenue protected) + (avoided reacquisition cost)
#   - revenue protected uses REVENUE capture (value-weighted)
#   - saved customers (avoided CAC) use CUSTOMER capture (head-count)
A=R["ARPU_month"]; annual_churn=0.22; margin=0.50; save_rate=0.30
offer_cost=30; target_frac=0.20; CAC=350           # $30 blended offer (loyalty nudge..financed upgrade)
N=1_000_000
evar_rev_cap =R["targeting"]["by_evar"]["0.2"]["rev_capture"]
evar_cust_cap=R["targeting"]["by_evar"]["0.2"]["cust_capture"]
churners=N*annual_churn
rev_at_risk=churners*A*12
rev_protected=rev_at_risk*evar_rev_cap*save_rate            # value-weighted
margin_protected=rev_protected*margin
saved=churners*evar_cust_cap*save_rate                      # head-count
avoided_cac=saved*CAC
campaign_cost=N*target_frac*offer_cost
gross_benefit=margin_protected+avoided_cac
net=gross_benefit-campaign_cost
roi=net/campaign_cost
# comparison vs naive (same 20% contacted)
score_rev_cap=R["targeting"]["by_score"]["0.2"]["rev_capture"]
rand_rev_cap=R["targeting"]["by_random"]["0.2"]["rev_capture"]
R["business"]={"N":N,"ARPU_month":A,"annual_churn":annual_churn,"margin":margin,
    "save_rate":save_rate,"offer_cost":offer_cost,"target_frac":target_frac,"CAC":CAC,
    "rev_at_risk":round(rev_at_risk),"evar_rev_capture20":evar_rev_cap,
    "evar_cust_capture20":evar_cust_cap,
    "rev_protected":round(rev_protected),"margin_protected":round(margin_protected),
    "avoided_cac":round(avoided_cac),"campaign_cost":round(campaign_cost),
    "gross_benefit":round(gross_benefit),
    "net_benefit":round(net),"roi":round(roi,2),
    "saved_customers":round(saved),
    "evar_vs_score_x":round(evar_rev_cap/score_rev_cap,2),
    "evar_vs_random_x":round(evar_rev_cap/rand_rev_cap,2),
    "net_5M":round(net*5)}
# CLV (contribution)
R["clv"]=round(A*margin/(annual_churn/12),0)

# sensitivity grid: save_rate x offer_cost (NET per 1M, value-weighted)
sens={}
for sr in (0.20,0.30,0.40):
    sens[f"{sr:.2f}"]={}
    for oc in (20,30,50):
        mp=rev_at_risk*evar_rev_cap*sr*margin
        ac=churners*evar_cust_cap*sr*CAC
        cc=N*target_frac*oc
        sens[f"{sr:.2f}"][str(oc)]=round(mp+ac-cc)
R["sensitivity"]=sens

# ----------------------------------------------------------------- quotation
# Fee framed (like the winning sample decks) as a small % of value created.
val=net  # client's net benefit per 1M subs / yr
setup_fee=300000; monthly_fee=45000; annual_fee=setup_fee+monthly_fee*12
R["quote"]={"net_value_1M":round(val),"gross_value_1M":round(gross_benefit),
            "setup_fee":setup_fee,"monthly_fee":monthly_fee,"annual_fee":annual_fee,
            "pct_of_net":round(annual_fee/val*100,1),
            "client_keeps":round(val-annual_fee)}

with open("analysis/results.json","w") as f: json.dump(R,f,indent=2)
print(json.dumps(R,indent=2)[:1500])
print("\n=== KEY ===")
print(f"n={R['n_customers']} churn(sample)={R['churn_rate_sample']} ARPU={R['ARPU_month']} tenure_med={R['tenure_median']}")
print(f"best={best} AUC={best_auc:.4f} CV={R['cv_auc_mean']}±{R['cv_auc_std']}")
print(f"EVaR top20 rev_cap={evar_rev_cap}  score={score_rev_cap}  random={rand_rev_cap}")
print(f"net/1M=${net:,.0f} ROI={roi:.2f}x  CLV=${R['clv']:,.0f}")
print("saved unified results.json")
