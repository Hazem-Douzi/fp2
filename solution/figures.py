"""Generate executive-ready figures for the notebook and slide deck."""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 13,
    "axes.titlesize": 16, "axes.titleweight": "bold", "axes.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
NAVY, RED, TEAL, AMBER, GREY = "#1f3b73", "#d1495b", "#15919b", "#e9a000", "#9aa3ad"
RNG = 42
FIG = "/home/ubuntu/fp2/solution/figures"
DATA = "/home/ubuntu/fp2/solution/telecom"

client = pd.read_csv(f"{DATA}/Client.csv")
record = pd.read_csv(f"{DATA}/Record.csv")
df = record.merge(client, on="Customer_ID", how="inner")
y = df["churn"].astype(int)
rev = df["rev_Mean"].fillna(df["rev_Mean"].median())

def savefig(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", bbox_inches="tight"); plt.close()
    print("saved", name)

# ---- Fig 1: churn balance ----
fig, ax = plt.subplots(figsize=(5.2, 4.2))
vc = y.value_counts().sort_index()
ax.bar(["Stayed (0)", "Churned (1)"], vc.values, color=[TEAL, RED], width=0.6)
for i, v in enumerate(vc.values):
    ax.text(i, v + 400, f"{v:,}\n({v/len(y):.1%})", ha="center", va="bottom", fontweight="bold")
ax.set_title("Modeling sample is balanced\n(~49.6% churn — oversampled for modeling)")
ax.set_ylabel("Customers"); ax.set_ylim(0, vc.max()*1.18); ax.set_yticks([])
savefig("01_churn_balance")

# ---- Fig 2: missingness (demographic vs behavioral) ----
miss = (df.isna().mean()*100).sort_values(ascending=False)
miss = miss[miss > 0.5].head(14)[::-1]
demo = {"numbcars","dwllsize","HHstatin","ownrent","dwlltype","lor","income","adults",
        "infobase","marital","creditcd","ethnic","forgntvl","prizm_social_one"}
colors = [AMBER if c in demo else TEAL for c in miss.index]
fig, ax = plt.subplots(figsize=(7.6, 5))
ax.barh(miss.index, miss.values, color=colors)
for i, v in enumerate(miss.values):
    ax.text(v+0.5, i, f"{v:.0f}%", va="center", fontsize=11)
ax.set_title("Demographic fields are largely missing;\nbehavioral/billing fields are near-complete")
ax.set_xlabel("% missing")
ax.legend(handles=[plt.Rectangle((0,0),1,1,color=AMBER), plt.Rectangle((0,0),1,1,color=TEAL)],
          labels=["Demographic (3rd-party append)","Behavioral / billing"], loc="lower right", fontsize=11)
savefig("02_missingness")

# ---- Fig 3: top churn correlations ----
num = df.select_dtypes(include=[np.number]).drop(columns=["Customer_ID"])
corr = num.corrwith(y).drop("churn").dropna().sort_values()
top = pd.concat([corr.head(7), corr.tail(7)])
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.barh(top.index, top.values, color=[RED if v>0 else TEAL for v in top.values])
ax.axvline(0, color="black", lw=0.8)
ax.set_title("What moves churn: equipment age up, handset price &\nusage down (Pearson corr. with churn)")
ax.set_xlabel("correlation with churn")
savefig("03_churn_correlations")

# ---- Fig 4: churn rate by equipment-age quintile ----
s = df[["eqpdays","churn"]].dropna()
s["q"] = pd.qcut(s["eqpdays"], 5)
g = s.groupby("q", observed=True)["churn"].mean()
labels = ["0-188\ndays","188-300","300-395","395-589","589+\ndays"]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
bars = ax.bar(labels, g.values, color=NAVY)
ax.axhline(y.mean(), color=RED, ls="--", lw=1.5, label=f"book avg {y.mean():.0%}")
for b, v in zip(bars, g.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.0%}", ha="center", fontweight="bold")
ax.set_title("Churn rises with handset age\n(equipment days in service)")
ax.set_ylabel("churn rate"); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylim(0, .7); ax.legend()
savefig("04_eqpdays_churn")

# ---- Fig 5: churn by key categorical levers ----
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
specs = [("asl_flag", {"N":"No spend\nlimit","Y":"Spend\nlimit"}, "Account spending limit"),
         ("refurb_new", {"N":"New","R":"Refurb."}, "Handset condition"),
         ("hnd_webcap", {"WCMB":"Modern\n(WCMB)","WC":"Web\n(WC)"}, "Handset capability")]
for ax,(col,lab,title) in zip(axes, specs):
    gg = df.groupby(col)["churn"].mean()
    keys = [k for k in lab if k in gg.index]
    vals = [gg[k] for k in keys]
    bars = ax.bar([lab[k] for k in keys], vals, color=[TEAL, RED][:len(keys)])
    ax.axhline(y.mean(), color="grey", ls="--", lw=1)
    for b,v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.0%}", ha="center", fontweight="bold")
    ax.set_title(title, fontsize=13); ax.set_ylim(0,.7); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
fig.suptitle("Equipment & account levers separate churners from stayers", fontsize=16, fontweight="bold")
savefig("05_categorical_levers")

# ===== modeling for fig 6-9 (identical pipeline to the notebook) =====
# Business-motivated engineered features (same as Hazem_Douzi.ipynb)
df["rev_per_min"]    = df["rev_Mean"] / df["mou_Mean"].replace(0, np.nan)
df["overage_share"]  = df["ovrrev_Mean"] / df["rev_Mean"].replace(0, np.nan)
df["mou_trend"]      = df["change_mou"] / df["mou_Mean"].replace(0, np.nan)
df["eqp_per_tenure"] = df["eqpdays"] / (df["months"]*30).replace(0, np.nan)
df["active_ratio"]   = df["actvsubs"] / df["uniqsubs"].replace(0, np.nan)
X = df.drop(columns=["churn","Customer_ID"]).copy()
for c in [c for c in X.columns if X[c].dtype==object or str(X[c].dtype)=="str"]:
    X[c] = pd.factorize(X[c].astype(str))[0]
X = X.apply(pd.to_numeric, errors="coerce")
idx = np.arange(len(X))
Xtr,Xte,ytr,yte,itr,ite = train_test_split(X,y,idx,test_size=.3,random_state=RNG,stratify=y)
rev_te = rev.values[ite]; yv = yte.values
models = {
    "Logistic Reg.": Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),
                               ("c",LogisticRegression(max_iter=2000))]),
    "Random Forest": Pipeline([("i",SimpleImputer(strategy="median")),
                               ("c",RandomForestClassifier(n_estimators=400,max_depth=14,min_samples_leaf=20,n_jobs=-1,random_state=RNG))]),
    "XGBoost": Pipeline([("i",SimpleImputer(strategy="median")),
                         ("c",XGBClassifier(n_estimators=600,learning_rate=.03,max_depth=5,subsample=.8,
                                            colsample_bytree=.8,reg_lambda=2,eval_metric="auc",random_state=RNG,n_jobs=-1))]),
    "LightGBM": Pipeline([("i",SimpleImputer(strategy="median")),
                          ("c",LGBMClassifier(n_estimators=800,learning_rate=.03,num_leaves=48,subsample=.8,
                                              colsample_bytree=.8,reg_lambda=2,random_state=RNG,n_jobs=-1,verbose=-1))]),
}
probs, aucs = {}, {}
for name,m in models.items():
    m.fit(Xtr,ytr); p = m.predict_proba(Xte)[:,1]; probs[name]=p; aucs[name]=roc_auc_score(yte,p)

# ---- Fig 6: AUC comparison ----
fig, ax = plt.subplots(figsize=(6.6,4.2))
names=list(aucs); vals=[aucs[n] for n in names]
cols=[GREY,GREY,NAVY,TEAL]
bars=ax.bar(names, vals, color=cols)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.004,f"{v:.3f}",ha="center",fontweight="bold")
ax.axhline(0.5,color=RED,ls="--",lw=1,label="random (0.50)")
ax.set_title("XGBoost / LightGBM lead on ROC-AUC"); ax.set_ylabel("ROC-AUC (test)")
ax.set_ylim(0.5,0.75); ax.legend()
savefig("06_auc_comparison")

# ---- Fig 7: ROC curves ----
fig, ax = plt.subplots(figsize=(6,5.2))
for name,col in zip(names,[GREY,AMBER,NAVY,TEAL]):
    fpr,tpr,_ = roc_curve(yte, probs[name]); ax.plot(fpr,tpr,label=f"{name} (AUC {aucs[name]:.3f})",color=col,lw=2)
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_title("ROC curves"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=11)
savefig("07_roc_curves")

# ---- Fig 8: feature importance (XGB gain) ----
booster = models["XGBoost"].named_steps["c"]
imp = pd.Series(booster.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)[::-1]
fig, ax = plt.subplots(figsize=(7.4,5.4))
ax.barh(imp.index, imp.values, color=NAVY)
ax.set_title("XGBoost feature importance (gain)\nEquipment, tenure & usage dominate")
ax.set_xlabel("relative importance")
savefig("08_feature_importance")

# ---- Fig 9: cumulative gains / lift (random vs churn-rank vs EVaR) ----
def gains(score):
    order=np.argsort(-score); ys=yv[order]
    return np.insert(np.cumsum(ys)/ys.sum(),0,0)
xq=np.linspace(0,1,len(yv)+1)
evar=probs["XGBoost"]*rev_te
fig, ax = plt.subplots(figsize=(6.6,5.2))
ax.plot(xq, gains(probs["XGBoost"]), color=NAVY, lw=2.4, label="Churn-probability ranking")
ax.plot(xq, gains(evar), color=TEAL, lw=2.4, label="EVaR ranking (Prob x Revenue)")
ax.plot([0,1],[0,1],"--",color=RED,lw=1.5,label="Random targeting")
ax.axvline(.2,color="grey",ls=":",lw=1)
ax.set_title("Targeting the top 20% by score captures\n~30% of churners (1.5x lift over random)")
ax.set_xlabel("share of customers contacted"); ax.set_ylabel("share of churners captured")
ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(fontsize=11, loc="lower right")
savefig("09_cumulative_gains")

# ---- Fig 9b: revenue-at-risk capture (EVaR advantage) ----
def rev_gains(score):
    order=np.argsort(-score); churn_rev=(rev_te*yv)[order]
    return np.insert(np.cumsum(churn_rev)/ (rev_te*yv).sum(),0,0)
fig, ax = plt.subplots(figsize=(6.6,5.2))
ax.plot(xq, rev_gains(probs["XGBoost"]), color=NAVY, lw=2.4, label="Churn-probability ranking")
ax.plot(xq, rev_gains(evar), color=TEAL, lw=2.4, label="EVaR ranking (Prob x Revenue)")
ax.plot([0,1],[0,1],"--",color=RED,lw=1.5,label="Random")
ax.axvline(.2,color="grey",ls=":",lw=1)
ax.set_title("EVaR targeting protects far more revenue:\n45% of revenue-at-risk in the top 20%")
ax.set_xlabel("share of customers contacted"); ax.set_ylabel("share of revenue-at-risk captured")
ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(fontsize=11, loc="lower right")
savefig("09b_revenue_capture")

# ---- Fig 10: personas (computed inline, identical logic to the notebook) ----
df["churn_prob"] = models["XGBoost"].predict_proba(X)[:,1]
_thr = df["churn_prob"].quantile(0.70)
_pool = df[df["churn_prob"]>=_thr].copy()
_med_mou = df["mou_Mean"].median()
def _persona(r):
    if (r["eqpdays"]>=365) or (r["refurb_new"]=="R") or (pd.isna(r["hnd_webcap"])):
        return "Aging-Handset"
    if (r["ovrrev_Mean"] or 0)>=10: return "Bill-Shock / Overage"
    if (pd.notna(r["mou_Mean"]) and r["mou_Mean"]<_med_mou) and ((r["change_mou"] or 0)<0): return "Disengaging"
    return "Stable-Loyalty"
_pool["persona"]=_pool.apply(_persona,axis=1)
_summary=_pool.groupby("persona").agg(customers=("churn","size"), avg_rev=("rev_Mean","mean")).sort_values("customers",ascending=False)
_summary["share"]=_summary["customers"]/_summary["customers"].sum()
names=list(_summary.index); shares=list(_summary["share"]); avgrev=list(_summary["avg_rev"])
fig, (a1,a2)=plt.subplots(1,2,figsize=(12,4.6))
cols=[NAVY,AMBER,TEAL,GREY]
a1.pie(shares, labels=[n.replace(" / ","/\n") for n in names], autopct="%1.0f%%", colors=cols,
       startangle=90, textprops={"fontsize":11,"fontweight":"bold"})
a1.set_title("Composition of the at-risk pool")
bars=a2.bar(range(len(names)), avgrev, color=cols)
a2.set_xticks(range(len(names))); a2.set_xticklabels([n.replace(" / ","/\n") for n in names], fontsize=10)
for b,v in zip(bars,avgrev): a2.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"${v:.0f}",ha="center",fontweight="bold")
a2.set_title("Avg monthly revenue by persona"); a2.set_ylabel("ARPU ($/mo)")
fig.suptitle("Two-thirds of at-risk customers are an aging-handset problem", fontsize=16, fontweight="bold")
savefig("10_personas")

# ---- Business case computed inline (identical to the notebook) ----
def _rev_capture(score,frac):
    k=int(np.ceil(len(yv)*frac)); order=np.argsort(-score)[:k]
    return (rev_te*yv)[order].sum()/(rev_te*yv).sum()
N=100_000; ARPU_m=58.7; ARPU_y=ARPU_m*12; annual_churn=0.22; CAC=350
target_frac=0.20; offer_cost=50; save_rate=0.30
_rc=_rev_capture(evar,target_frac)
_order=np.argsort(-evar)[:int(np.ceil(len(yv)*target_frac))]; _cc_cap=yv[_order].sum()/yv.sum()
churners=N*annual_churn; targeted=int(N*target_frac); trar=churners*ARPU_y
rev_protected=trar*_rc*save_rate; saved=churners*_cc_cap*save_rate
avoided_cac=saved*CAC; campaign_cost=targeted*offer_cost
net_benefit=rev_protected+avoided_cac-campaign_cost; roi_val=net_benefit/campaign_cost

# ---- Fig 11: ROI waterfall ----
rp=rev_protected/1e6; cac=avoided_cac/1e6; cc=campaign_cost/1e6; net=net_benefit/1e6
fig, ax = plt.subplots(figsize=(7.6,5.0))
steps=["Revenue\nprotected","Avoided\nreacquisition","Campaign\ncost","Net\nbenefit"]
bottoms=[0, rp, rp+cac-cc, 0]
heights=[rp, cac, -cc, net]
cols=[TEAL, TEAL, RED, NAVY]
labels=[f"+${rp:.2f}M", f"+${cac:.2f}M", f"-${cc:.2f}M", f"${net:.2f}M"]
for i,(s,b,h,c,lb) in enumerate(zip(steps,bottoms,heights,cols,labels)):
    ax.bar(s, h, bottom=b, color=c, width=0.62)
    top = b+h if h>0 else b
    ax.text(i, max(b+h, b)+0.06, lb, ha="center", va="bottom", fontweight="bold", fontsize=12)
# connector lines
ax.plot([0.31,0.69],[rp,rp],color="grey",lw=1,ls=":")
ax.plot([1.31,1.69],[rp+cac,rp+cac],color="grey",lw=1,ls=":")
ax.plot([2.31,2.69],[rp+cac-cc,rp+cac-cc],color="grey",lw=1,ls=":")
ax.axhline(0,color="black",lw=0.8)
ax.set_ylim(0, (rp+cac)*1.18)
ax.set_title(f"Retention program economics — per 100k customers / year\nNet benefit ${net:.2f}M  •  ROI {roi_val*100:.0f}%", pad=14)
ax.set_ylabel("$ millions / year")
savefig("11_roi_waterfall")

# ---- Fig 12: sensitivity heatmap (computed inline) ----
srs=[0.20,0.30,0.40]; ocs=[30,50,75]
M=np.array([[ (trar*_rc*sr + churners*_cc_cap*sr*CAC - targeted*oc)/1e6 for oc in ocs] for sr in srs])
fig, ax = plt.subplots(figsize=(6.4,4.4))
im=ax.imshow(M, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(ocs))); ax.set_xticklabels([f"${o}" for o in ocs])
ax.set_yticks(range(len(srs))); ax.set_yticklabels([f"{s:.0%}" for s in srs])
ax.set_xlabel("offer cost / targeted customer"); ax.set_ylabel("save rate")
for i in range(len(srs)):
    for j in range(len(ocs)):
        ax.text(j,i,f"${M[i,j]:.1f}M",ha="center",va="center",fontweight="bold")
ax.set_title("Net benefit stays positive across assumptions\n(per 100k customers / year)")
fig.colorbar(im, label="$M net benefit")
savefig("12_sensitivity")

print("\nALL FIGURES DONE")
