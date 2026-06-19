"""
Generates polished, consistently-styled charts for the slide deck.
Reuses the merged dataset + saved model_results.json.
Output PNGs -> deliverables/figs/
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

DATA = "assignment/05. Final Assignment (due Jun 19th 11AM UTC)/Final Assignment/telecom"
OUT = "/vercel/share/v0-project/deliverables/figs"
os.makedirs(OUT, exist_ok=True)

# ---- palette ----
NAVY="#1f3a5f"; BLUE="#2563eb"; ORANGE="#f97316"; TEAL="#0d9488"
SLATE="#475569"; LIGHT="#e2e8f0"; RED="#dc2626"
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.edgecolor": "#cbd5e1", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": "#eef2f6", "grid.linewidth": 1.0,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": NAVY, "axes.labelcolor": NAVY, "xtick.color": SLATE, "ytick.color": SLATE,
})

def save(fig, name):
    fig.tight_layout()
    p=os.path.join(OUT,name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", p)

print("loading data...")
client=pd.read_csv(os.path.join(DATA,"Client.csv"))
record=pd.read_csv(os.path.join(DATA,"Record.csv"))
df=client.merge(record,on="Customer_ID",how="inner")
res=json.load(open("/vercel/share/v0-project/analysis/model_results.json"))

# 1) target balance
fig,ax=plt.subplots(figsize=(6,4.2))
vc=df["churn"].value_counts().sort_index()
ax.bar(["Retained","Churned"],vc.values,color=[BLUE,ORANGE],width=0.6)
for i,v in enumerate(vc.values): ax.text(i,v+800,f"{v:,}",ha="center",fontweight="bold",color=NAVY)
ax.set_title("Modeling sample is balanced ~50/50"); ax.set_ylabel("Customers"); ax.set_ylim(0,vc.max()*1.15)
save(fig,"01_balance.png")

# 2) ARPU + tenure
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
axes[0].hist(df["rev_Mean"].clip(upper=200).dropna(),bins=40,color=BLUE,alpha=.9)
axes[0].axvline(df["rev_Mean"].mean(),color=ORANGE,lw=2.5,ls="--")
axes[0].set_title(f"ARPU mean = ${df['rev_Mean'].mean():.2f}/mo"); axes[0].set_xlabel("Monthly revenue ($)"); axes[0].set_ylabel("Customers")
axes[1].hist(df["months"].dropna(),bins=40,color=TEAL,alpha=.9)
axes[1].axvline(df["months"].mean(),color=ORANGE,lw=2.5,ls="--")
axes[1].set_title(f"Tenure mean = {df['months'].mean():.0f} months"); axes[1].set_xlabel("Months in service")
save(fig,"02_arpu_tenure.png")

# 3) correlation drivers
num=df.select_dtypes(include=[np.number]).drop(columns=["churn"],errors="ignore")
corr=num.corrwith(df["churn"]).dropna().sort_values()
top=pd.concat([corr.tail(7),corr.head(7)]).sort_values()
labels={"eqpdays":"Days on current handset","hnd_price":"Handset price","totmrc_Mean":"Monthly recurring charge",
        "mou_Mean":"Minutes of use","uniqsubs":"# unique subscriptions","months":"Tenure (months)",
        "ovrrev_Mean":"Overage revenue","vceovr_Mean":"Voice overage","actvsubs":"# active subs",
        "mou_cvce_Mean":"Completed voice mins","complete_Mean":"Completed calls","change_mou":"Change in usage",
        "avg3mou":"Avg 3-mo minutes","models":"# handset models","ovrmou_Mean":"Overage minutes"}
fig,ax=plt.subplots(figsize=(8.5,5.2))
cols=[RED if v>0 else BLUE for v in top.values]
ax.barh([labels.get(i,i) for i in top.index],top.values,color=cols)
ax.axvline(0,color=NAVY,lw=1)
ax.set_title("What correlates with churn  (red = increases churn)")
ax.set_xlabel("Correlation with churn")
save(fig,"03_drivers.png")

# 4) equipment age -> churn (the key insight)
tmp=df.dropna(subset=["eqpdays"]).copy()
tmp["b"]=pd.cut(tmp["eqpdays"],[-10,180,360,540,720,2000],labels=["0-6mo","6-12mo","12-18mo","18-24mo","24mo+"])
g=tmp.groupby("b")["churn"].mean()
fig,ax=plt.subplots(figsize=(8,4.4))
ax.plot(g.index.astype(str),g.values*100,marker="o",ms=10,lw=3,color=ORANGE)
for x,y in zip(range(len(g)),g.values*100): ax.text(x,y+0.8,f"{y:.0f}%",ha="center",fontweight="bold",color=NAVY)
ax.set_title("Churn climbs steadily as handsets age"); ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("Time on current handset")
ax.set_ylim(g.values.min()*100-4,g.values.max()*100+5)
save(fig,"04_eqpdays.png")

# 5) ROC / model comparison
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score
y=df["churn"].astype(int); X=df.drop(columns=["churn","Customer_ID"],errors="ignore")
ncol=X.select_dtypes(include=[np.number]).columns.tolist()
ccol=[c for c in X.select_dtypes(include=["object"]).columns if X[c].nunique()<=40]
X=X[ncol+ccol]
pre=ColumnTransformer([("n",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),ncol),
                       ("c",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),ccol)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
mods={"Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
      "Random Forest":RandomForestClassifier(n_estimators=300,max_depth=14,n_jobs=-1,random_state=42),
      "XGBoost":XGBClassifier(n_estimators=400,max_depth=5,learning_rate=.05,subsample=.9,colsample_bytree=.9,eval_metric="logloss",n_jobs=-1,random_state=42)}
cmap={"Logistic Regression":SLATE,"Random Forest":TEAL,"XGBoost":ORANGE}
fig,ax=plt.subplots(figsize=(6.2,5.2))
best_proba=None
for name,clf in mods.items():
    pipe=Pipeline([("p",pre),("c",clf)]); pipe.fit(Xtr,ytr)
    pr=pipe.predict_proba(Xte)[:,1]; fpr,tpr,_=roc_curve(yte,pr); auc=roc_auc_score(yte,pr)
    lw=3.5 if name=="XGBoost" else 2
    ax.plot(fpr,tpr,lw=lw,color=cmap[name],label=f"{name}  (AUC={auc:.3f})")
    if name=="XGBoost": best_proba=pr
ax.plot([0,1],[0,1],"--",color="#94a3b8")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC — XGBoost wins"); ax.legend(loc="lower right",fontsize=11)
save(fig,"05_roc.png")

# 6) feature importance
imp=pd.DataFrame(res["top_features"]).head(10)
nice={"eqpdays":"Days on handset","refurb_new_N":"New (not refurb) handset","months":"Tenure",
      "hnd_webcap_WCMB":"Web-capable handset","ethnic_Z":"Demographic seg. Z","refurb_new_R":"Refurbished handset",
      "hnd_price":"Handset price","asl_flag_N":"No account-spending limit","mou_Mean":"Minutes of use",
      "crclscod_EA":"Credit class EA"}
imp["lab"]=imp["feature"].map(lambda f:nice.get(f,f))
fig,ax=plt.subplots(figsize=(8.5,5))
ax.barh(imp["lab"][::-1],imp["importance"][::-1],color=BLUE)
ax.set_title("Top churn predictors (XGBoost)"); ax.set_xlabel("Importance")
save(fig,"06_importance.png")

# 7) lift / gains
lift=pd.DataFrame(res["lift"])
fig,ax=plt.subplots(figsize=(8,4.4))
bars=ax.bar(lift["decile"],lift["lift"],color=[ORANGE if d<=3 else LIGHT for d in lift["decile"]])
ax.axhline(1,color=NAVY,ls="--",lw=1.5)
ax.text(8,1.03,"average customer",color=NAVY,fontsize=11)
ax.set_title("Top-3 risk deciles capture 42% of churners"); ax.set_xlabel("Risk decile (1 = highest risk)"); ax.set_ylabel("Lift vs. average")
ax.set_xticks(lift["decile"])
save(fig,"07_lift.png")

# 8) business case waterfall (per 1M subs)
gross=36.999; cost=14.4; net=22.6
fig,ax=plt.subplots(figsize=(7.5,4.6))
ax.bar(["Gross value\nof retained\ncustomers"],[gross],color=TEAL,width=0.55)
ax.bar(["Program\ncost"],[cost],color=RED,width=0.55)
ax.bar(["NET annual\nbenefit"],[net],color=ORANGE,width=0.55)
for i,v in enumerate([gross,cost,net]): ax.text(i,v+0.6,f"${v:.1f}M",ha="center",fontweight="bold",color=NAVY,fontsize=13)
ax.set_title("Annual retention economics  (per 1M subscribers)"); ax.set_ylabel("$ millions / year"); ax.set_ylim(0,gross*1.18)
save(fig,"08_waterfall.png")

# 9) scaling
subs=[1,3,5,10]; netv=[22.6*s for s in subs]
fig,ax=plt.subplots(figsize=(7.5,4.4))
ax.bar([f"{s}M" for s in subs],netv,color=NAVY,width=0.6)
for i,v in enumerate(netv): ax.text(i,v+3,f"${v:.0f}M",ha="center",fontweight="bold",color=NAVY)
ax.set_title("Net annual benefit scales with subscriber base"); ax.set_xlabel("Subscriber base"); ax.set_ylabel("Net benefit ($M/yr)")
ax.set_ylim(0,max(netv)*1.15)
save(fig,"09_scaling.png")

print("ALL DECK CHARTS DONE")
