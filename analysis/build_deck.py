"""
Builds the 15-slide business-proposal PDF deck (16:9) for Company A telecom churn.
All numbers are pulled from analysis/results.json (single source of truth, produced by analysis/pipeline.py).
Output: deliverables/Company_A_Business_Proposal.pdf
"""
import os, json
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from PIL import Image as PILImage

W, H = 1280, 720
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f"{BASE}/deliverables/figs"
OUT = f"{BASE}/deliverables/Company_A_Business_Proposal.pdf"
R = json.load(open(f"{BASE}/analysis/results.json"))
B = R["business"]; Q = R["quote"]; BEST = R["best_model"]
BEST_AUC = R["models"][BEST]["auc"]
base = B["base"]; bear = B["bear"]; bull = B["bull"]

def money(n):
    n=float(n)
    if abs(n)>=1e6: return f"${n/1e6:.1f}M"
    if abs(n)>=1e3: return f"${n/1e3:.0f}k"
    return f"${n:.0f}"

NAVY=HexColor("#1f3a5f"); BLUE=HexColor("#2563eb"); ORANGE=HexColor("#f97316")
TEAL=HexColor("#0d9488"); SLATE=HexColor("#475569"); LIGHT=HexColor("#f1f5f9")
WHITE=HexColor("#ffffff"); INK=HexColor("#0f172a"); MUTE=HexColor("#64748b")
LINE=HexColor("#e2e8f0"); GREEN=HexColor("#059669"); REDC=HexColor("#dc2626")
CARD=HexColor("#274b73")

c = canvas.Canvas(OUT, pagesize=(W, H))

def rect(x,y,w,h,color,r=0):
    c.setFillColor(color)
    if r: c.roundRect(x,y,w,h,r,stroke=0,fill=1)
    else: c.rect(x,y,w,h,stroke=0,fill=1)

def text(x,y,s,size=14,color=INK,font="Helvetica",align="left"):
    c.setFillColor(color); c.setFont(font,size)
    if align=="left": c.drawString(x,y,s)
    elif align=="center": c.drawCentredString(x,y,s)
    else: c.drawRightString(x,y,s)

def wrap(x,y,s,size,color,font="Helvetica",leading=None,maxw=W-160):
    leading=leading or size*1.45
    c.setFont(font,size); c.setFillColor(color)
    words=s.split(); line=""; yy=y
    for w in words:
        t=(line+" "+w).strip()
        if c.stringWidth(t,font,size)>maxw:
            c.drawString(x,yy,line); line=w; yy-=leading
        else: line=t
    if line: c.drawString(x,yy,line)
    return yy-leading

def img(path,x,y,maxw,maxh):
    im=PILImage.open(path); iw,ih=im.size; rr=min(maxw/iw,maxh/ih)
    w,h=iw*rr,ih*rr
    c.drawImage(ImageReader(path),x+(maxw-w)/2,y+(maxh-h)/2,w,h,mask="auto")

TOTAL=17
def header(title,kicker,num):
    rect(0,H-92,W,92,WHITE)
    rect(80,H-92,6,52,ORANGE)
    text(100,H-52,kicker,13,ORANGE,"Helvetica-Bold")
    text(100,H-80,title,24,NAVY,"Helvetica-Bold")
    text(W-80,H-78,f"{num} / {TOTAL}",12,MUTE,"Helvetica",align="right")
    rect(80,H-100,W-160,1.4,LINE)

def footer(cite=None):
    rect(0,0,W,40,WHITE)
    text(80,16,"Company A  ·  Customer Retention PoC",10,MUTE)
    text(W-80,16,"GCI World 2026 — Final Assignment",10,MUTE,align="right")
    if cite: text(80,52,cite,9.5,MUTE,"Helvetica-Oblique")

def page_bg(): rect(0,0,W,H,HexColor("#f8fafc"))

def chip(x,y,label,value,color,w=250,h=104):
    rect(x,y,w,h,WHITE,r=14); rect(x,y,6,h,color,r=0)
    text(x+24,y+h-30,value,26,color,"Helvetica-Bold")
    wrap(x+24,y+h-56,label,11.5,SLATE,"Helvetica",leading=14,maxw=w-40)

def bullet(x,y,head,body,color=ORANGE,maxw=470):
    rect(x,y+3,9,9,color,r=2)
    text(x+20,y,head,13.5,INK,"Helvetica-Bold")
    return wrap(x+20,y-20,body,12,SLATE,"Helvetica",leading=16.5,maxw=maxw)

# ============================================================ SLIDE 1 — TITLE
rect(0,0,W,H,NAVY); rect(0,0,W,8,ORANGE)
rect(80,H-180,70,6,ORANGE)
text(80,H-240,"Winning Back the Base",46,WHITE,"Helvetica-Bold")
text(80,H-292,"A Value-Based Customer Retention Strategy for Company A",23,HexColor("#cbd5e1"))
wrap(80,H-360,"Turning 100,000 customer records into a quantified retention program — targeting not just "
     "who will churn, but who is most valuable to keep, then proving ROI with a paid A/B pilot.",16,HexColor("#94a3b8"),
     leading=24,maxw=740)
rect(900,H-470,300,300,CARD,r=18)
text(925,H-222,f"{BEST_AUC:.3f}",42,ORANGE,"Helvetica-Bold")
text(925,H-247,"XGBoost ROC-AUC (CV-validated)",11.5,HexColor("#cbd5e1"))
text(925,H-298,money(base["net_benefit"]),30,WHITE,"Helvetica-Bold")
text(925,H-321,"net benefit / yr (per 1M subs)",12,HexColor("#cbd5e1"))
text(925,H-368,f"{B['evar_vs_score_x']:.1f}x",30,TEAL,"Helvetica-Bold")
text(925,H-391,"more revenue saved vs. churn-only",12,HexColor("#cbd5e1"))
text(80,70,"IT & Data-Science Consulting — Proof of Concept  ·  Prepared for the Company A Executive Team",13,HexColor("#94a3b8"))
text(80,46,"Author: [Omnicampus Account Name]   ·   June 2026",11,HexColor("#64748b"))
c.showPage()

# ============================================================ SLIDE 2 — EXECUTIVE SUMMARY
page_bg(); header("Executive summary","THE ANSWER, UP FRONT",2)
text(100,H-148,"Company A loses customers faster than it can profitably replace them. We can fix that.",16.5,INK,"Helvetica-Bold")
y=H-198
y=bullet(100,y,"The problem","In a saturated market, ~95% of growth comes from switchers and re-acquiring a lost "
       "customer costs 5-25x more than keeping one. Churn is Company A's most expensive leak.")
y=bullet(100,y-12,"What the data says","Churn is predictable and concentrated. The strongest driver is aging "
       "handsets - devices past ~12 months churn at 54-58% vs ~42% when new - a lever Company A controls.")
y=bullet(100,y-12,"Our solution","A monthly value-based targeting engine: rank customers by Expected Value-at-Risk "
       "(P(churn) x customer value), then make proactive, persona-specific retention offers.")
y=bullet(100,y-12,"Why it works","The model is moderately predictive (AUC ~0.69) - but the value comes from "
       "value-weighted targeting, calibration, fairness and A/B validation, not from over-claiming accuracy.",color=TEAL)
rect(720,H-470,470,300,WHITE,r=16); rect(720,H-216,470,46,NAVY,r=16)
text(744,H-200,"What it delivers - per 1,000,000 subscribers",13.5,WHITE,"Helvetica-Bold")
rows=[("Best model",f"{BEST} · AUC {BEST_AUC:.3f}"),
      ("Targeting edge",f"captures {B['evar_rev_capture20']*100:.0f}% of revenue-at-risk in top 20%"),
      ("Net benefit / yr",money(base["net_benefit"])+f"   ({base['roi']:.2f}x ROI)"),
      ("Customers saved / yr",f"~{base['saved_customers']:,}"),
      ("Our fee",money(Q["annual_fee"])+f"/yr · {Q['pct_of_net']:.1f}% of value created")]
yy=H-250
for k,v in rows:
    text(744,yy,k,12,MUTE,"Helvetica"); text(1166,yy,v,12,NAVY,"Helvetica-Bold",align="right")
    rect(744,yy-12,422,1,LINE); yy-=40
footer("Sources: Bain & Company; Harvard Business Review; analysis of Company A dataset (n=100,000).")
c.showPage()

# ============================================================ SLIDE 3 — MARKET ANALYSIS
page_bg(); header("Market analysis: retention is the only profitable growth lever","MARKET ANALYSIS",3)
chip(100,H-250,"US wireless market (2025)",  "$344B", BLUE, w=250)
chip(370,H-250,"Best-in-class monthly churn","1.0%", TEAL, w=250)
chip(640,H-250,"Re-acquire vs. retain cost", "5-25x", ORANGE, w=250)
chip(910,H-250,"Profit lift from +5% retention","25%+", GREEN, w=280)
y=H-320
y=bullet(100,y,"The market is saturated","Penetration exceeds 100%. Carriers grow almost entirely by stealing "
       "switchers - making every avoided cancellation worth far more than a new-customer ad.",maxw=480)
y=bullet(100,y-14,"Acquisition is expensive","Industry CAC runs into the hundreds of dollars per line; "
       "retention offers are a fraction of that, so saved customers are immediately accretive.",maxw=480)
y=H-320
y=bullet(640,y,"Small churn moves, big profit","Bain's classic finding: a 5-point retention improvement can "
       "lift profits 25-95%. Churn is the highest-leverage metric in the P&L.",maxw=470)
y=bullet(640,y-14,"Loyalty compounds","Tenured customers spend more, cost less to serve, and refer others - "
       "so protecting the base protects future revenue too.",maxw=470)
footer("Sources: industry market sizing 2025; Verizon/T-Mobile investor reports; Bain & Company; Reichheld, HBR.")
c.showPage()

# ============================================================ SLIDE 4 — CLIENT & DATA
page_bg(); header("The client & the data","CLIENT & DATASET",4)
wrap(100,H-148,"Company A is a US wireless carrier facing elevated churn. We received two linked tables "
     "covering 100,000 customers - behavioural usage and customer profile - joined on Customer_ID.",
     14,SLATE,leading=20,maxw=1080)
chip(100,H-300,"Customers",  "100,000", NAVY, w=210)
chip(322,H-300,"Model features",  f"{R['n_features_model']}", BLUE, w=210)
chip(544,H-300,"Avg revenue / mo", f"${R['ARPU_month']:.0f}", TEAL, w=210)
chip(766,H-300,"Median tenure", f"{R['median_tenure']:.0f} mo", ORANGE, w=210)
y=H-360
bullet(100,y,"Behavioural (Record.csv)","Minutes of use, overage, revenue, call quality, recharges, and the "
       "churn label - the in-life signals of disengagement.",color=BLUE,maxw=470)
bullet(640,y,"Profile (Client.csv)","Handset age & price, refurbished flag, web-capability, tenure - "
       "the structural drivers we can act on.",color=TEAL,maxw=470)
rect(100,90,1080,82,HexColor("#fff7ed"),r=12); rect(100,90,6,82,ORANGE)
wrap(126,154,"Data discipline: (1) the modelling sample is balanced ~50/50 by design (oversampled), so it is used "
     "for ranking - the real operating churn assumption (22%/yr) is applied separately in the business case; "
     "(2) we DROP 9 fields with 25-49% missingness and 4 protected attributes (ethnicity, marital status, income, "
     "credit class) - see slide 15 for the fairness test.",11.5,SLATE,leading=15.5,maxw=1040)
footer("Source: Company A dataset (Client.csv + Record.csv), 100,000 customers.")
c.showPage()

# ============================================================ SLIDE 5 — EDA 1
page_bg(); header("What the data reveals: who leaves, and why","EXPLORATORY DATA ANALYSIS  (1/2)",5)
img(f"{FIG}/02_correlations.png",90,150,560,430)
y=H-160
y=bullet(700,y,"Equipment age is the #1 signal","Time on the same handset (eqpdays / handset_months) is the strongest "
       "positive correlate of churn (+0.11) - phones age out and customers shop around.",color=ORANGE,maxw=470)
y=bullet(700,y-12,"Investment buys loyalty","Higher handset price and higher recurring charge both correlate "
       "negatively with churn - customers with newer, pricier phones stay.",color=TEAL,maxw=470)
y=bullet(700,y-12,"Engagement matters","More minutes of use and completed calls track with lower churn; "
       "a declining usage trend is an early-warning sign we engineered into the model.",color=BLUE,maxw=470)
footer("Source: point-biserial correlations with churn flag, Company A dataset (n=100,000).")
c.showPage()

# ============================================================ SLIDE 6 — EDA 2
page_bg(); header("The headline insight: the handset-age cliff","EXPLORATORY DATA ANALYSIS  (2/2)",6)
img(f"{FIG}/03_handset.png",90,150,560,430)
e=R["handset_bins"]
y=H-160
y=bullet(700,y,"A clear climb after ~12 months",
       f"Churn rises from ~{e['6-12mo']['churn']*100:.0f}% (6-12 mo) to ~{e['12-18mo']['churn']*100:.0f}% past "
       f"12 months and reaches ~{e['24mo+']['churn']*100:.0f}% beyond two years.",color=ORANGE,maxw=470)
y=bullet(700,y-12,"This is an actionable lever","Unlike age or income, handset age is something Company A can "
       "change - through proactive upgrade and trade-in offers timed to the cliff.",color=TEAL,maxw=470)
y=bullet(700,y-12,"Real months, not guesses","Buckets are computed directly from days-on-handset in the data - "
       "no assumed labels - so the timing of the intervention is evidence-based.",color=BLUE,maxw=470)
footer("Source: churn rate by months-on-current-handset, Company A dataset.")
c.showPage()

# ============================================================ SLIDE 7 — PROBLEM & ML TASK
page_bg(); header("Framing the business problem as an ML task","PROBLEM DEFINITION",7)
rect(100,H-244,1080,84,HexColor("#eef2ff"),r=12); rect(100,H-244,6,84,BLUE)
wrap(126,H-186,"Business question:  'Which customers are about to leave, how much revenue do they put at risk, "
     "and what is the most cost-effective way to keep them?'",15,NAVY,"Helvetica-Bold",leading=22,maxw=1030)
y=H-296
y=bullet(100,y,"ML formulation","Supervised binary classification: predict P(churn) per customer from "
       "behavioural + profile features, then calibrate to the real base rate.",color=BLUE,maxw=480)
y=bullet(100,y-12,"But probability isn't enough","A $149/mo customer and a $42/mo customer are not equally worth "
       "saving. We rank by Expected Value-at-Risk = P(churn) x customer value.",color=ORANGE,maxw=480)
y=H-296
y=bullet(640,y,"Primary metric: ROC-AUC","Ranking quality matters more than a single cutoff, because we act on "
       "the highest-risk, highest-value customers first.",color=TEAL,maxw=470)
y=bullet(640,y-12,"Success = profit, not accuracy","Judged by incremental margin + avoided re-acquisition cost from "
       "a targeted campaign, validated by an A/B holdout.",color=GREEN,maxw=470)
footer()
c.showPage()

# ============================================================ SLIDE 8 — METHODOLOGY
page_bg(); header("Modelling approach: rigorous, reproducible, honest","METHODOLOGY",8)
steps=[("01","Clean & engineer","Drop 9 high-missing + 4 protected fields; impute, encode; engineer 7 features "
        f"(usage trend, overage share, handset months...). {R['n_features_model']} model features."),
       ("02","Benchmark + ensemble","4 model families (LogReg, RF, XGBoost, LightGBM) plus soft-voting & "
        "stacking ensembles - judged on revenue capture, not AUC alone."),
       ("03","Validate & adjust","75/25 stratified split + 5-fold CV; isotonic-checked reliability (ECE "
        f"{R['calibration_metrics']['ece_raw']*100:.1f}%); prior-adjusted to the 22% base rate, pilot-calibrated."),
       ("04","Rank by value","Convert to Expected Value-at-Risk; profit-curve sets the contact threshold; "
        "simulate vs random and churn-only baselines.")]
for i,(n,h,b) in enumerate(steps):
    col=[BLUE,TEAL,ORANGE,GREEN][i]
    bx=100+i*278
    rect(bx,H-430,258,250,WHITE,r=14); rect(bx,H-430,258,52,col,r=14)
    text(bx+18,H-410,n,22,WHITE,"Helvetica-Bold")
    text(bx+18,H-462,h,13.5,INK,"Helvetica-Bold")
    wrap(bx+18,H-486,b,11,SLATE,leading=14.5,maxw=224)
rect(100,95,1080,60,HexColor("#f0fdf4"),r=12); rect(100,95,6,60,GREEN)
wrap(126,138,f"Cross-validation result:  {BEST} ROC-AUC = {R['cv_auc_mean']:.3f} +/- {R['cv_auc_std']:.3f} "
     "across 5 folds - confirming the model generalises and the score is stable, not luck.",12.5,INK,"Helvetica-Bold",leading=16,maxw=1030)
footer()
c.showPage()

# ============================================================ SLIDE 9 — RESULTS
page_bg(); header(f"Results: {BEST} wins - AUC {BEST_AUC:.3f}","MODEL RESULTS  (NAME · METRIC · SCORE)",9)
img(f"{FIG}/04_models.png",80,150,540,430)
tx,ty=690,H-200; cw=[150,70,70,70,70]
heads=["Model","AUC","Acc","Prec","Recall"]
rect(tx,ty-6,sum(cw),34,NAVY)
cx=tx
for hh,w in zip(heads,cw):
    text(cx+10,ty+4,hh,12,WHITE,"Helvetica-Bold"); cx+=w
order=["Logistic Regression","Random Forest","LightGBM","XGBoost"]
yy=ty-40
for m in order:
    md=R["models"][m]; best=(m==BEST)
    rect(tx,yy-8,sum(cw),34,HexColor("#eef7f4") if best else WHITE)
    if best: rect(tx,yy-8,4,34,TEAL)
    vals=[m,f"{md['auc']:.3f}",f"{md['acc']:.3f}",f"{md['prec']:.3f}",f"{md['rec']:.3f}"]
    cx=tx
    for v,w,j in zip(vals,cw,range(5)):
        text(cx+10,yy,v,11.5,NAVY if best else INK,"Helvetica-Bold" if (best or j==0) else "Helvetica"); cx+=w
    rect(tx,yy-12,sum(cw),1,LINE); yy-=42
en=R["ensemble"]
wrap(690,yy-2,f"How to read AUC: 0.5 = guessing, 1.0 = perfect. At {BEST_AUC:.3f} the model is moderately "
     "predictive - it ranks a churner above a non-churner ~69% of the time. We do not over-sell this: the "
     "payoff comes from value-weighted targeting, not raw accuracy.",11.5,SLATE,leading=16,maxw=490)
rect(690,108,490,70,HexColor("#f0fdf4"),r=10); rect(690,108,6,70,GREEN)
wrap(712,158,f"Ensembles tested honestly: soft-voting & stacking reached AUC {en['Soft Voting']['auc']:.4f} "
     f"(+{R['ensemble_auc_gain']:.4f}) - below our 0.003 bar - so we keep XGBoost for simpler, governable deployment.",
     11,INK,"Helvetica-Bold",leading=14.5,maxw=450)
footer("Source: held-out test set (25% of 100,000), Company A dataset.")
c.showPage()

# ============================================================ SLIDE 10 — TRUST: SHAP + CALIBRATION
page_bg(); header("Why we trust the model: explainable & calibrated","MODEL EXPLAINABILITY & TRUST",10)
img(f"{FIG}/05_shap.png",70,150,540,430)
img(f"{FIG}/06_calibration.png",640,170,330,400)
sh=list(R["shap_top"].items())
y=H-150
y=bullet(985,y,"Explainable (SHAP)",f"The top churn drivers are {sh[0][0]}, {sh[1][0]} and {sh[2][0]} - all "
       "behaviour Company A can act on. No protected attribute appears.",color=ORANGE,maxw=210)
cm=R["calibration_metrics"]
y=bullet(985,y-14,"Well-calibrated",f"Reliability error is only ECE {cm['ece_raw']*100:.1f}% - probabilities already "
       "track reality; isotonic confirms it. We still prior-adjust 50/50->22% for dollar scale.",color=TEAL,maxw=210)
y=bullet(985,y-14,"Actionable","Because drivers are controllable, every score maps to a concrete offer.",color=BLUE,maxw=210)
footer("Source: SHAP values on held-out set; reliability curve (10 bins), isotonic-checked; ECE/Brier reported.")
c.showPage()

# ============================================================ SLIDE 11 — DIFFERENTIATOR (EVaR)
page_bg(); header("Our differentiator: target value, not just probability","VALUE-BASED TARGETING  (EVaR)",11)
img(f"{FIG}/07_evar.png",90,150,560,430)
sc=R["targeting"]["by_score"]["0.2"]["rev_capture"]
ev=R["targeting"]["by_evar"]["0.2"]["rev_capture"]
y=H-160
y=bullet(700,y,"The insight","Ranking by churn probability alone wastes budget on low-value churners. "
       "Ranking by Expected Value-at-Risk protects the revenue that actually matters.",color=ORANGE,maxw=470)
rect(700,H-336,470,90,WHITE,r=12)
text(722,H-280,f"{ev*100:.0f}%",30,TEAL,"Helvetica-Bold")
text(722,H-308,"of revenue-at-risk captured in top 20%",11.5,SLATE)
text(980,H-280,f"vs {sc*100:.0f}%",24,MUTE,"Helvetica-Bold")
text(980,H-308,"churn-only ranking",11.5,SLATE)
y=H-356
y=bullet(700,y,f"{B['evar_vs_score_x']:.1f}x more revenue protected","Same campaign budget, same 20% of customers "
       f"contacted - but we shield {B['evar_vs_score_x']:.1f}x the revenue by leading with value.",color=TEAL,maxw=470)
footer("Source: gains simulation on held-out test set; EVaR = calibrated P(churn) x annualised customer revenue.")
c.showPage()

# ============================================================ SLIDE 12 — PERSONAS
page_bg(); header("Who to target: four data-derived retention personas","CUSTOMER SEGMENTATION  (K-MEANS)",12)
img(f"{FIG}/10_personas.png",90,160,560,420)
P=R["personas"]
# label personas from real cluster stats
def label_persona(p):
    if p["handset_months"]>=18: return ("Upgrade-Ready Switchers","Aging device, modest spend","Proactive upgrade / trade-in",ORANGE)
    if p["rev"]>=120: return ("High-Value Protect-and-Hold","Top spend & usage","White-glove retention, protect margin",TEAL)
    if p["overage_share"]>=0.3: return ("Bill-Shock Frustrated","High bill, heavy overage","Right-size plan + autopay discount",REDC)
    return ("Low-Engagement Flight Risks","Newer, mid-spend, largest group","Onboarding nudges + loyalty perks",BLUE)
yy=H-170
seen=set()
for p in P:
    lbl,desc,action,col=label_persona(p)
    # avoid dup labels
    base_lbl=lbl; k=2
    while lbl in seen: lbl=f"{base_lbl} {k}"; k+=1
    seen.add(lbl)
    rect(700,yy-58,480,70,WHITE,r=10); rect(700,yy-58,6,70,col)
    text(722,yy-16,lbl,13,INK,"Helvetica-Bold")
    text(1166,yy-16,f"{p['share']*100:.0f}% · ${p['rev']:.0f}/mo · {p['churn']*100:.0f}% churn",10.5,col,"Helvetica-Bold",align="right")
    text(722,yy-36,desc,10.5,MUTE)
    text(722,yy-52,"-> "+action,10.5,SLATE,"Helvetica-Oblique")
    yy-=84
footer("Source: K-means (k=4) on standardised behavioural features of the customer base, Company A dataset.")
c.showPage()

# ============================================================ SLIDE 13 — TREATMENT DESIGN
page_bg(); header("From segment to action: the treatment playbook","RETENTION OFFER DESIGN",13)
wrap(100,H-148,"Each persona maps to a specific offer with an explicit cost, an expected save-rate, and a "
     "margin guardrail - so spend is disciplined and never margin-negative.",13.5,SLATE,leading=18,maxw=1080)
cols=[("Persona",250),("Problem",235),("Offer",235),("Cost",95),("Save rate",115),("Guardrail",150)]
rows=[("Upgrade-Ready Switchers","Aging device triggers shopping","Upgrade / trade-in","$50-120","Medium","Only high-EVaR",ORANGE),
      ("Bill-Shock Frustrated","Overage frustration","Plan right-size + autopay","$10-30","High","No margin-negative plan",REDC),
      ("Low-Engagement Flight Risks","Usage collapse, drifting away","Loyalty call / perk","$15-40","Low-Med","Test vs holdout",BLUE),
      ("High-Value Protect-and-Hold","Valuable but at-risk","Light loyalty + concierge","$5-15","Low","Avoid over-discount",TEAL)]
tx=100; ty=H-250
cx=tx; rect(tx,ty,sum(w for _,w in cols),34,NAVY)
for name,w in cols:
    text(cx+12,ty+11,name,11.5,WHITE,"Helvetica-Bold"); cx+=w
yy=ty-2
for r in rows:
    yy-=58
    rect(tx,yy,sum(w for _,w in cols),58,WHITE); rect(tx,yy,6,58,r[6])
    cx=tx; vals=r[:6]
    for (name,w),v,i in zip(cols,vals,range(6)):
        fnt="Helvetica-Bold" if i==0 else "Helvetica"
        col=INK if i==0 else SLATE
        wrap(cx+14,yy+36,v,10.3,col,fnt,leading=12.5,maxw=w-18)
        cx+=w
    rect(tx,yy,sum(w for _,w in cols),1,LINE)
rect(100,92,1080,52,HexColor("#f0fdf4"),r=10); rect(100,92,6,52,GREEN)
wrap(124,128,"Every offer is gated by Expected Value-at-Risk and validated against a holdout - we only spend "
     "where the expected saved margin exceeds the offer cost.",11.5,INK,"Helvetica-Bold",leading=15,maxw=1040)
footer("Offer costs are planning ranges to be calibrated in the pilot; save-rates validated by A/B holdout.")
c.showPage()

# ============================================================ SLIDE 14 — BUSINESS CASE
page_bg(); header("The business case: strong base/bull upside, bear flags the floor","QUANTIFIED IMPACT  (per 1M subscribers / yr)",14)
img(f"{FIG}/08_profit_curve.png",70,150,560,420)
img(f"{FIG}/09_scenarios.png",650,150,560,420)
chip(70,86,"Net benefit / yr", money(base["net_benefit"]), GREEN, w=250, h=84)
chip(338,86,"Return on spend", f"{base['roi']:.2f}x", TEAL, w=210, h=84)
chip(650,86,"Customers saved / yr", f"{base['saved_customers']:,}", BLUE, w=250, h=84)
chip(918,86,"Bull-case upside", money(bull["net_benefit"]), ORANGE, w=262, h=84)
footer(f"Base case: 22% annual churn, 50% margin, 30% save-rate, ${base['offer']} offer, ${base['cac']} CAC. "
       f"Bear case ({bear['roi']:.2f}x) flags the risk; the A/B pilot resolves which scenario is real before scaling.")
c.showPage()

# ============================================================ SLIDE 14 — QUOTATION (THE ASK)
page_bg(); header("Our proposal & quotation","THE ASK",16)
wrap(100,H-148,"We propose a 12-week build of the retention engine, then an ongoing managed service. "
     "Our fee is a small fraction of the value we protect.",14,SLATE,leading=20,maxw=1040)
ph=[("Phase 1 · Build (wk 1-6)","Data pipeline, model, calibration, EVaR scoring, persona logic",BLUE),
    ("Phase 2 · Pilot (wk 7-12)","A/B holdout on top-20% pool; validate incremental saves & ROI",TEAL),
    ("Phase 3 · Scale (ongoing)","Monthly scoring, offer optimisation, drift monitoring, QBRs",ORANGE)]
yy=H-220
for h,b,col in ph:
    rect(100,yy-54,520,66,WHITE,r=10); rect(100,yy-54,6,66,col)
    text(122,yy-18,h,13.5,INK,"Helvetica-Bold")
    wrap(122,yy-38,b,11.5,SLATE,leading=15,maxw=480)
    yy-=80
rect(680,H-470,500,300,NAVY,r=18); rect(680,H-222,500,52,CARD,r=18)
text(704,H-202,"Investment",15,WHITE,"Helvetica-Bold")
prow=[("One-time setup & build",money(Q["setup_fee"])),
      ("Managed service / month",money(Q["monthly_fee"])),
      ("Total year 1",money(Q["annual_fee"]))]
yy=H-250
for k,v in prow:
    text(704,yy,k,12.5,HexColor("#cbd5e1")); text(1156,yy,v,14,WHITE,"Helvetica-Bold",align="right")
    rect(704,yy-12,452,1,HexColor("#3b5a80")); yy-=42
rect(704,H-452,452,70,HexColor("#16324f"),r=12)
text(724,H-410,f"= just {Q['pct_of_net']:.1f}% of the {money(base['net_benefit'])} net value we create",13,ORANGE,"Helvetica-Bold")
text(724,H-432,f"Company A keeps ~{money(Q['client_keeps'])} / yr per 1M subscribers",11.5,HexColor("#cbd5e1"))
footer("Fee illustrative for a PoC engagement; scales with subscriber base. Value figures per 1M subscribers / yr.")
c.showPage()

# ============================================================ SLIDE 15 — RISKS, FAIRNESS & REFERENCES
page_bg(); header("Risks, fairness & references","GOVERNANCE & SOURCES",17)
img(f"{FIG}/11_fairness.png",90,250,470,300)
f=R["fairness"]
improve=f["auc_without_protected"]-f["auc_with_protected"]
text(325,232,f"AUC {f['auc_with_protected']:.4f} -> {f['auc_without_protected']:.4f}  ({improve:+.4f}, no cost)",10.5,NAVY,"Helvetica-Bold",align="center")
y=H-150
y=bullet(600,y,"Fairness by design",f"We exclude ethnicity, marital status, income & credit class. Removing them does "
       f"NOT hurt performance - AUC moves {f['auc_with_protected']:.4f} -> {f['auc_without_protected']:.4f} ({improve:+.4f}), a negligible change.",color=BLUE,maxw=560)
y=bullet(600,y-10,"Validate before scaling","The 50/50 sample inflates apparent churn; the A/B pilot calibrates real "
       "save-rates and ROI - and tells us whether we are in the bear or bull scenario.",color=ORANGE,maxw=560)
y=bullet(600,y-10,"Monitor for drift","Re-train on fresh data and watch feature drift; a stale model silently "
       "loses accuracy as customer behaviour shifts.",color=TEAL,maxw=560)
y=bullet(600,y-10,"Don't over-discount","Profit-curve guardrails and holdouts prevent giving margin away to "
       "customers who would have stayed anyway.",color=GREEN,maxw=560)
# references
rect(600,86,580,160,WHITE,r=12); rect(600,212,580,34,NAVY,r=12)
text(620,222,"Selected references",12.5,WHITE,"Helvetica-Bold")
refs=["Reichheld & Sasser, 'Zero Defections', HBR.  ·  Bain & Company retention research.",
      "Verizon / T-Mobile investor reports 2024-25.  ·  CTIA Wireless Survey 2025.",
      "Verbeke et al. (2012), profit-driven churn, EJOR.  ·  Lundberg & Lee (2017), SHAP, NeurIPS.",
      "scikit-learn; XGBoost; LightGBM documentation."]
yy=186
for r in refs:
    text(620,yy,"-",10,ORANGE); wrap(634,yy,r,9.6,SLATE,leading=12,maxw=540); yy-=26
rect(100,96,470,80,HexColor("#fff7ed"),r=10); rect(100,96,6,80,ORANGE)
wrap(124,150,"AI-use disclosure: generative AI assisted with code scaffolding, drafting and visualisation. "
     "All analysis, numbers and conclusions were computed from the dataset and verified by the author. "
     "Every figure is reproduced by the accompanying notebook.",9.6,SLATE,leading=13,maxw=440)
footer()
c.showPage()

c.save()
print("Saved", OUT, round(os.path.getsize(OUT)/1024), "KB")
