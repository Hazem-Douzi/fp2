"""Generate all deck/notebook charts from analysis/results.json + SHAP artifacts.
Run AFTER analysis/pipeline.py.  Saves PNGs to deliverables/figs/."""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

R = json.load(open("analysis/results.json"))
FIGS = "deliverables/figs"; os.makedirs(FIGS, exist_ok=True)

NAVY="#0f2742"; ORANGE="#e8833a"; TEAL="#2a9d8f"; GRAY="#94a3b8"; RED="#c1432e"
plt.rcParams.update({"font.size":13,"axes.edgecolor":"#cbd5e1","axes.linewidth":0.9,
  "axes.grid":True,"grid.color":"#eef2f7","figure.facecolor":"white","axes.facecolor":"white",
  "axes.spines.top":False,"axes.spines.right":False})

def save(fig,name):
    fig.tight_layout(); fig.savefig(f"{FIGS}/{name}",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("saved",name)

def money(x): return f"${x/1e6:.1f}M"

# ---- 1. churn balance ----
fig,ax=plt.subplots(figsize=(7,4.2))
cr=R["sample_churn_rate"]
ax.bar(["Stayed","Churned"],[1-cr,cr],color=[TEAL,ORANGE],width=.6)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
for i,v in enumerate([1-cr,cr]): ax.text(i,v+.008,f"{v*100:.1f}%",ha="center",fontweight="bold",color=NAVY)
ax.set_title("Modeling sample is balanced ~50/50 (oversampled)",fontweight="bold",color=NAVY)
ax.set_ylim(0,.6); save(fig,"01_balance.png")

# ---- 2. correlation drivers ----
items=sorted({**R["corr_neg"],**R["corr_pos"]}.items(),key=lambda t:t[1])
labels=[k for k,_ in items]; vals=[v for _,v in items]
cols=[TEAL if v<0 else ORANGE for v in vals]
fig,ax=plt.subplots(figsize=(8,5))
ax.barh(labels,vals,color=cols); ax.axvline(0,color=NAVY,lw=1)
ax.set_title("What moves churn: correlation with the churn flag",fontweight="bold",color=NAVY)
ax.set_xlabel("retains  <-        correlation with churn        ->  increases")
save(fig,"02_correlations.png")

# ---- 3. churn by handset age (REAL months) ----
eb=R["handset_bins"]; xs=list(eb.keys()); ys=[eb[k]["churn"] for k in xs]
fig,ax=plt.subplots(figsize=(7.5,4.4))
ax.plot(xs,ys,marker="o",color=ORANGE,lw=3,ms=9)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
for x,y in zip(xs,ys): ax.text(x,y+.012,f"{y*100:.0f}%",ha="center",fontweight="bold",color=NAVY)
ax.set_ylim(min(ys)-0.04,max(ys)+0.06)
ax.set_title("Churn climbs once a handset passes ~12 months",fontweight="bold",color=NAVY,pad=12)
ax.set_xlabel("Months on current handset"); ax.set_ylabel("Churn rate")
save(fig,"03_handset.png")

# ---- 4. model comparison (AUC) ----
m=R["models"]; names=list(m.keys()); aucs=[m[n]["auc"] for n in names]
order=np.argsort(aucs); names=[names[i] for i in order]; aucs=[aucs[i] for i in order]
cols=[ORANGE if n==R["best_model"] else GRAY for n in names]
fig,ax=plt.subplots(figsize=(8,4.4))
ax.barh(names,aucs,color=cols)
for i,v in enumerate(aucs): ax.text(v+.003,i,f"{v:.3f}",va="center",fontweight="bold",color=NAVY)
ax.set_xlim(.5,.75); ax.axvline(.5,color=GRAY,ls="--",lw=1)
ax.set_title(f"4 models benchmarked - {R['best_model']} wins (ROC-AUC)",fontweight="bold",color=NAVY)
ax.set_xlabel("ROC-AUC (0.5 = random)"); save(fig,"04_models.png")

# ---- 5. SHAP importance (mean |SHAP|) ----
sh=R["shap_top"]; ks=list(sh.keys())[::-1]; vs=[sh[k] for k in ks]
fig,ax=plt.subplots(figsize=(7.6,5))
ax.barh(ks,vs,color=NAVY)
ax.set_title("Why customers leave: SHAP feature impact (XGBoost)",fontweight="bold",color=NAVY)
ax.set_xlabel("Mean |SHAP value| (impact on churn odds)")
save(fig,"05_shap.png")

# ---- 6. calibration curve (raw vs isotonic) + Brier/ECE ----
cal=R["calibration"]; cm=R["calibration_metrics"]
fig,ax=plt.subplots(figsize=(6.8,5))
ax.plot([0,1],[0,1],ls="--",color=GRAY,label="Perfect calibration")
ax.plot(cal["mean_pred"],cal["frac_pos"],marker="o",color=ORANGE,lw=2.5,label="Raw XGBoost")
ax.plot(cal["mean_pred_cal"],cal["frac_pos_cal"],marker="s",color=TEAL,lw=2.2,label="Isotonic-calibrated")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed churn rate")
ax.set_title("Reliability: probabilities track reality closely",fontweight="bold",color=NAVY)
ax.text(0.04,0.92,f"ECE {cm['ece_raw']*100:.1f}%  |  Brier {cm['brier_raw']:.3f}",
        transform=ax.transAxes,fontsize=10.5,color=NAVY,fontweight="bold")
ax.legend(loc="lower right"); save(fig,"06_calibration.png")

# ---- 7. EVaR vs FOUR baselines (rev-at-risk captured @20%) ----
bl=R["baseline_rev20"]
spec=[("Random\noutreach","by_random",GRAY),
      ("Rule: oldest\nhandset","by_rule",GRAY),
      ("Revenue-only\nranking","by_revenue",GRAY),
      ("Churn-score\nranking","by_score",TEAL),
      ("EVaR\n(value-weighted)","by_evar",ORANGE)]
names=[s[0] for s in spec]; vals=[bl[s[1]] for s in spec]; cols=[s[2] for s in spec]
fig,ax=plt.subplots(figsize=(9,4.7))
bars=ax.bar(names,vals,color=cols,width=.64)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.008,f"{v*100:.0f}%",ha="center",fontweight="bold",color=NAVY)
ax.set_title("Revenue-at-risk captured in the SAME top-20% contacted",fontweight="bold",color=NAVY)
ax.set_ylim(0,max(vals)+.08); save(fig,"07_evar.png")

# ---- 8. profit curve (population-scaled, per 1M) ----
pc=R["profit_curve"]; fr=[p["frac"] for p in pc]; net=[p["net"]/1e6 for p in pc]
opt=R["optimal_contact_frac"]
fig,ax=plt.subplots(figsize=(7.8,4.6))
ax.plot(fr,net,marker="o",color=ORANGE,lw=3,ms=6)
ax.axvline(opt,color=NAVY,ls="--",lw=1.5)
ax.text(opt,min(net),f"  optimum ~{opt*100:.0f}%",color=NAVY,fontweight="bold",va="bottom")
ax.yaxis.set_major_formatter(lambda x,_:f"${x:.0f}M"); ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("Share of base contacted (ranked by expected value)"); ax.set_ylabel("Net benefit / 1M subs")
ax.set_title("Profit curve: contact only where expected value is positive",fontweight="bold",color=NAVY)
save(fig,"08_profit_curve.png")

# ---- 9. bear/base/bull scenarios ----
b=R["business"]; sc_names=["Bear","Base","Bull"]; keys=["bear","base","bull"]
nets=[b[k]["net_benefit"]/1e6 for k in keys]; rois=[b[k]["roi"] for k in keys]
cols=[RED if n<0 else (TEAL if k!="base" else ORANGE) for n,k in zip(nets,keys)]
fig,ax=plt.subplots(figsize=(8,4.6))
bars=ax.bar(sc_names,nets,color=cols,width=.6)
ax.axhline(0,color=NAVY,lw=1)
for bb,n,r in zip(bars,nets,rois):
    ax.text(bb.get_x()+bb.get_width()/2, n+(0.3 if n>=0 else -0.6),
            f"${n:.1f}M\n{r:.2f}x ROI",ha="center",fontweight="bold",color=NAVY,fontsize=11)
ax.yaxis.set_major_formatter(lambda x,_:f"${x:.0f}M")
ax.set_title("Net benefit / 1M subs - bear / base / bull",fontweight="bold",color=NAVY)
ax.set_ylim(min(nets)-2,max(nets)+3); save(fig,"09_scenarios.png")

# ---- 10. personas ----
P=R["personas"]
labels=[f"Persona {i+1}" for i in range(len(P))]
shares=[p["share"] for p in P]
fig,ax=plt.subplots(figsize=(8,4.6))
cols=[ORANGE,TEAL,NAVY,GRAY][:len(P)]
bars=ax.bar(labels,shares,color=cols,width=.62)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
for bb,p in zip(bars,P):
    ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+.008,
            f"{p['share']*100:.0f}%\n${p['rev']:.0f}/mo\n{p['churn']*100:.0f}% churn",
            ha="center",fontweight="bold",color=NAVY,fontsize=9.5)
ax.set_title("At-risk base splits into 4 actionable personas (K-means)",fontweight="bold",color=NAVY)
ax.set_ylim(0,max(shares)+.16); save(fig,"10_personas.png")

# ---- 11. fairness A/B ----
f=R["fairness"]
fig,ax=plt.subplots(figsize=(6.8,4.4))
bars=ax.bar(["With protected\nattributes","Protected\nattributes removed"],
            [f["auc_with_protected"],f["auc_without_protected"]],color=[GRAY,TEAL],width=.55)
for bb,v in zip(bars,[f["auc_with_protected"],f["auc_without_protected"]]):
    ax.text(bb.get_x()+bb.get_width()/2,v+.002,f"{v:.4f}",ha="center",fontweight="bold",color=NAVY)
ax.set_ylim(.6,.71)
improve = f["auc_without_protected"] - f["auc_with_protected"]
ax.set_title(f"Removing protected attributes: {improve:+.4f} AUC (no cost)",fontweight="bold",color=NAVY)
ax.set_ylabel("ROC-AUC"); save(fig,"11_fairness.png")

# ---- 12. ensemble experiment: AUC vs revenue capture@20% ----
en=R["ensemble"]; enames=list(en.keys())
aucs=[en[n]["auc"] for n in enames]; revs=[en[n]["rev20"] for n in enames]
x=np.arange(len(enames)); w=0.38
fig,ax=plt.subplots(figsize=(8.6,4.8))
ax2=ax.twinx()
b1=ax.bar(x-w/2,aucs,w,color=NAVY,label="ROC-AUC")
b2=ax2.bar(x+w/2,revs,w,color=ORANGE,label="Rev-at-risk capture @20%")
ax.set_xticks(x); ax.set_xticklabels(enames,rotation=12,ha="right",fontsize=10.5)
ax.set_ylim(.66,.70); ax2.set_ylim(.40,.46)
ax.set_ylabel("ROC-AUC",color=NAVY); ax2.set_ylabel("Revenue capture @20%",color=ORANGE)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0)); ax2.grid(False)
for xi,a in zip(x,aucs): ax.text(xi-w/2,a+.0008,f"{a:.4f}",ha="center",fontsize=8.6,fontweight="bold",color=NAVY)
ax.set_title(f"Ensembles tested: +{R['ensemble_auc_gain']:.4f} AUC -> keep XGBoost (simpler to govern)",
             fontweight="bold",color=NAVY,fontsize=12)
save(fig,"12_ensemble.png")

# ---- 13. Precision-Recall curve (honest view on imbalance) ----
pr=R["pr_curve"]; base=R["test_positive_rate"]; prauc=R["best_pr_auc"]
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(pr["recall"],pr["precision"],color=ORANGE,lw=2.6,label=f"{R['best_model']} (PR-AUC {prauc:.3f})")
ax.axhline(base,color=GRAY,ls="--",lw=1.4,label=f"No-skill baseline ({base*100:.0f}%)")
ax.set_xlabel("Recall (share of churners caught)"); ax.set_ylabel("Precision")
ax.set_ylim(0,1); ax.set_xlim(0,1)
ax.set_title("Precision-Recall: honest view beyond ROC-AUC",fontweight="bold",color=NAVY)
ax.legend(loc="upper right"); save(fig,"13_pr_curve.png")

# ---- 14. feature-group ablation (does more signal help?) ----
ab=R["ablation"]; steps=[a["step"] for a in ab]; aucs=[a["auc"] for a in ab]; nf=[a["n_features"] for a in ab]
fig,ax=plt.subplots(figsize=(8.4,4.7))
ax.plot(steps,aucs,marker="o",color=NAVY,lw=2.6,ms=9)
for x,(a,n) in enumerate(zip(aucs,nf)):
    ax.text(x,a+0.0015,f"{a:.3f}",ha="center",fontweight="bold",color=NAVY,fontsize=10.5)
    ax.text(x,a-0.0025,f"{n} feats",ha="center",color=GRAY,fontsize=8.5)
lo=min(aucs); hi=max(aucs); pad=max(0.004,(hi-lo)*0.4)
ax.set_ylim(lo-pad,hi+pad)
ax.set_title("Ablation: behavioural signal helps, then plateaus near ~0.69",fontweight="bold",color=NAVY)
ax.set_ylabel("Test ROC-AUC"); ax.set_xlabel("Cumulative feature groups added")
save(fig,"14_ablation.png")

print("ALL CHARTS DONE")
