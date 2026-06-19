"""Generate all deck/notebook charts from analysis/results.json + raw data.
Run after unified.py.  Saves PNGs to deliverables/figs/."""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

R=json.load(open("analysis/results.json"))
DATA=os.environ.get("TELECOM_DATA_DIR",
  "assignment/05. Final Assignment (due Jun 19th 11AM UTC)/Final Assignment/telecom")
FIGS="deliverables/figs"; os.makedirs(FIGS,exist_ok=True)

NAVY="#0f2742"; ORANGE="#e8833a"; TEAL="#2a9d8f"; GRAY="#94a3b8"; LIGHT="#cbd5e1"; RED="#c1432e"
plt.rcParams.update({"font.size":13,"axes.edgecolor":"#cbd5e1","axes.linewidth":0.9,
  "axes.grid":True,"grid.color":"#eef2f7","figure.facecolor":"white","axes.facecolor":"white",
  "axes.spines.top":False,"axes.spines.right":False})

def save(fig,name):
    fig.tight_layout(); fig.savefig(f"{FIGS}/{name}",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("saved",name)

# ---- 1. churn balance + ARPU/tenure value box (text-light bar) ----
fig,ax=plt.subplots(figsize=(7,4.2))
cr=R["churn_rate_sample"]
ax.bar(["Stayed","Churned"],[1-cr,cr],color=[TEAL,ORANGE],width=.6)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
for i,v in enumerate([1-cr,cr]): ax.text(i,v+.01,f"{v*100:.1f}%",ha="center",fontweight="bold")
ax.set_title("Modeling sample is balanced 50/50 (oversampled)",fontweight="bold",color=NAVY)
ax.set_ylim(0,.6); save(fig,"01_balance.png")

# ---- 2. correlation drivers ----
neg=R["corr_neg"]; pos=R["corr_pos"]
items=sorted({**neg,**pos}.items(),key=lambda t:t[1])
labels=[k for k,_ in items]; vals=[v for _,v in items]
cols=[TEAL if v<0 else ORANGE for v in vals]
fig,ax=plt.subplots(figsize=(8,5))
ax.barh(labels,vals,color=cols)
ax.axvline(0,color=NAVY,lw=1)
ax.set_title("What moves churn: point-biserial correlation with churn",fontweight="bold",color=NAVY)
ax.set_xlabel("← retains        correlation with churn        increases →")
save(fig,"02_correlations.png")

# ---- 3. churn by handset age bins ----
eb=R.get("eqp_bins",{})
fig,ax=plt.subplots(figsize=(7.5,4.4))
xs=list(eb.keys()); ys=[eb[k] for k in xs]
ax.plot(xs,ys,marker="o",color=ORANGE,lw=3,ms=9)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
for x,y in zip(xs,ys): ax.text(x,y+.012,f"{y*100:.0f}%",ha="center",fontweight="bold",color=NAVY)
ax.set_ylim(min(ys)-0.04, max(ys)+0.06)
ax.set_title("Churn jumps once a handset passes ~12 months",fontweight="bold",color=NAVY,pad=12)
ax.set_xlabel("Days on current handset (quintiles)"); ax.set_ylabel("Churn rate")
save(fig,"03_eqpdays.png")

# ---- 4. AUC comparison ----
m=R["models"]; names=list(m.keys()); aucs=[m[n]["auc"] for n in names]
order=np.argsort(aucs); names=[names[i] for i in order]; aucs=[aucs[i] for i in order]
cols=[ORANGE if n==R["best_model"] else GRAY for n in names]
fig,ax=plt.subplots(figsize=(8,4.4))
ax.barh(names,aucs,color=cols)
for i,v in enumerate(aucs): ax.text(v+.003,i,f"{v:.3f}",va="center",fontweight="bold",color=NAVY)
ax.set_xlim(.5,.75); ax.axvline(.5,color=GRAY,ls="--",lw=1)
ax.set_title(f"Model comparison — {R['best_model']} wins (ROC-AUC)",fontweight="bold",color=NAVY)
ax.set_xlabel("ROC-AUC (0.5 = random)")
save(fig,"04_auc.png")

# ---- 5. feature importance ----
fi=R["top_features"]; ks=list(fi.keys())[::-1][-10:]; vs=[fi[k] for k in ks]
fig,ax=plt.subplots(figsize=(7.5,5))
ax.barh(ks,vs,color=NAVY)
ax.set_title("XGBoost feature importance (top drivers)",fontweight="bold",color=NAVY)
save(fig,"05_importance.png")

# ---- 6. EVaR vs score vs random: revenue-at-risk captured at top 20% ----
ev=R["targeting"]["by_evar"]["0.2"]["rev_capture"]
sc=R["targeting"]["by_score"]["0.2"]["rev_capture"]
rd=R["targeting"]["by_random"]["0.2"]["rev_capture"]
fig,ax=plt.subplots(figsize=(7.5,4.6))
bars=ax.bar(["Random\noutreach","Churn-score\nranking","EVaR\n(value-weighted)"],[rd,sc,ev],
       color=[GRAY,TEAL,ORANGE],width=.6)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
for b,v in zip(bars,[rd,sc,ev]): ax.text(b.get_x()+b.get_width()/2,v+.01,f"{v*100:.0f}%",ha="center",fontweight="bold",color=NAVY)
ax.set_title("Revenue-at-risk captured in the SAME top-20% contacted",fontweight="bold",color=NAVY)
ax.set_ylim(0,.55)
save(fig,"06_evar.png")

# ---- 7. cumulative gains (capture vs population) using best model on raw ----
# reconstruct gains curve from capture points
fr=[0,.10,.20,.30,1.0]
cap=[0,R["targeting"]["by_evar"]["0.1"]["cust_capture"],
       R["targeting"]["by_evar"]["0.2"]["cust_capture"],
       R["targeting"]["by_evar"]["0.3"]["cust_capture"],1.0]
fig,ax=plt.subplots(figsize=(7.5,4.6))
ax.plot([0,1],[0,1],ls="--",color=GRAY,label="Random")
ax.plot(fr,cap,marker="o",color=ORANGE,lw=3,label="Model (EVaR rank)")
ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("Customers contacted (by risk)"); ax.set_ylabel("Churners captured")
ax.set_title("Cumulative gains — target less, catch more",fontweight="bold",color=NAVY)
ax.legend(); save(fig,"07_gains.png")

# ---- 8. ROI waterfall ----
b=R["business"]
steps=["Margin\nprotected","Avoided\nreacq. cost","Campaign\ncost","Net\nbenefit"]
vals=[b["margin_protected"],b["avoided_cac"],-b["campaign_cost"],0]
fig,ax=plt.subplots(figsize=(8,4.6))
cum=0; 
for i,(s,v) in enumerate(zip(steps,vals)):
    if s.startswith("Net"):
        ax.bar(i,b["net_benefit"],color=NAVY); ax.text(i,b["net_benefit"]+3e5,f"${b['net_benefit']/1e6:.1f}M",ha="center",fontweight="bold",color=NAVY)
    else:
        c=TEAL if v>=0 else RED
        ax.bar(i,v,bottom=cum if v>=0 else cum+v,color=c)
        ax.text(i,(cum+ (v if v>0 else 0))+ (3e5 if v>0 else -9e5),f"${v/1e6:+.1f}M",ha="center",fontweight="bold",color=NAVY)
        cum+=v
ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps)
ax.yaxis.set_major_formatter(lambda x,_:f"${x/1e6:.0f}M")
ax.set_title(f"Net benefit per 1M subscribers / yr — ROI {b['roi']}x",fontweight="bold",color=NAVY)
save(fig,"08_waterfall.png")

# ---- 9. sensitivity heatmap ----
sens=R["sensitivity"]; srs=sorted(sens.keys()); ocs=sorted(sens[srs[0]].keys(),key=int)
mat=np.array([[sens[sr][oc]/1e6 for oc in ocs] for sr in srs])
fig,ax=plt.subplots(figsize=(7.5,4.6))
im=ax.imshow(mat,cmap="YlGn",aspect="auto")
ax.set_xticks(range(len(ocs))); ax.set_xticklabels([f"${o}" for o in ocs])
ax.set_yticks(range(len(srs))); ax.set_yticklabels([f"{float(s)*100:.0f}%" for s in srs])
ax.set_xlabel("Offer cost / contacted customer"); ax.set_ylabel("Save rate")
for i in range(len(srs)):
    for j in range(len(ocs)):
        ax.text(j,i,f"${mat[i,j]:.1f}M",ha="center",va="center",fontweight="bold",color=NAVY)
ax.set_title("Net benefit / 1M subs — positive in every scenario",fontweight="bold",color=NAVY)
save(fig,"09_sensitivity.png")

# ---- 10. personas ----
P=R["personas"]; names=list(P.keys()); shares=[P[n]["share"] for n in names]
fig,ax=plt.subplots(figsize=(8,4.6))
cols=[ORANGE,TEAL,NAVY,GRAY][:len(names)]
b2=ax.bar(names,shares,color=cols,width=.6)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
for bb,n in zip(b2,names):
    ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+.008,
            f"{P[n]['share']*100:.0f}%\n${P[n]['avg_rev']:.0f}/mo",ha="center",fontweight="bold",color=NAVY,fontsize=10)
ax.set_title("At-risk pool splits into 4 actionable personas",fontweight="bold",color=NAVY)
ax.set_ylim(0,max(shares)+.12); plt.xticks(rotation=12,ha="right")
save(fig,"10_personas.png")

print("ALL CHARTS DONE")
