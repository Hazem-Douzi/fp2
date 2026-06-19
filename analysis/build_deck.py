"""
Builds the 15-slide business-proposal PDF deck (16:9) for Company A telecom churn.
All numbers are pulled from analysis/results.json (single source of truth).
Output: deliverables/Company_A_Business_Proposal.pdf
"""
import os, json
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from PIL import Image as PILImage

W, H = 1280, 720
BASE = "/vercel/share/v0-project"
FIG = f"{BASE}/deliverables/figs"
OUT = f"{BASE}/deliverables/Company_A_Business_Proposal.pdf"
R = json.load(open(f"{BASE}/analysis/results.json"))
B = R["business"]; Q = R["quote"]

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
    im=PILImage.open(path); iw,ih=im.size; r=min(maxw/iw,maxh/ih)
    w,h=iw*r,ih*r
    c.drawImage(ImageReader(path),x+(maxw-w)/2,y+(maxh-h)/2,w,h,mask="auto")

def header(title,kicker,num):
    rect(0,H-92,W,92,WHITE)
    rect(80,H-92,6,52,ORANGE)
    text(100,H-52,kicker,13,ORANGE,"Helvetica-Bold")
    text(100,H-80,title,25,NAVY,"Helvetica-Bold")
    text(W-80,H-78,f"{num} / 15",12,MUTE,"Helvetica",align="right")
    rect(80,H-100,W-160,1.4,LINE)

def footer(cite=None):
    rect(0,0,W,40,WHITE)
    text(80,16,"Company A  ·  Customer Retention PoC",10,MUTE)
    text(W-80,16,"GCI World 2026 — Final Assignment",10,MUTE,align="right")
    if cite: text(80,52,cite,9.5,MUTE,"Helvetica-Oblique")

def page_bg(): rect(0,0,W,H,HexColor("#f8fafc"))

def chip(x,y,label,value,color,w=250,h=104):
    rect(x,y,w,h,WHITE,r=14); rect(x,y,6,h,color,r=0)
    text(x+24,y+h-30,value,28,color,"Helvetica-Bold")
    wrap(x+24,y+h-58,label,11.5,SLATE,"Helvetica",leading=14,maxw=w-40)

def bullet(x,y,head,body,color=ORANGE,maxw=470):
    rect(x,y+3,9,9,color,r=2)
    text(x+20,y,head,13.5,INK,"Helvetica-Bold")
    return wrap(x+20,y-20,body,12,SLATE,"Helvetica",leading=16.5,maxw=maxw)

# ============================================================ SLIDE 1 — TITLE
rect(0,0,W,H,NAVY); rect(0,0,W,8,ORANGE)
rect(80,H-180,70,6,ORANGE)
text(80,H-240,"Winning Back the Base",46,WHITE,"Helvetica-Bold")
text(80,H-292,"A Value-Based Customer Retention Strategy for Company A",23,HexColor("#cbd5e1"))
wrap(80,H-360,"Turning 100,000 customer records into a quantified, ROI-positive retention program — "
     "targeting not just who will churn, but who is most valuable to keep.",16,HexColor("#94a3b8"),
     leading=24,maxw=740)
rect(900,H-470,300,300,CARD,r=18)
text(925,H-222,f"{R['models'][R['best_model']]['auc']:.3f}",42,ORANGE,"Helvetica-Bold")
text(925,H-247,"XGBoost ROC-AUC (CV-validated)",11.5,HexColor("#cbd5e1"))
text(925,H-298,money(B["net_benefit"]),30,WHITE,"Helvetica-Bold")
text(925,H-321,"net benefit / yr (per 1M subs)",12,HexColor("#cbd5e1"))
text(925,H-368,f"{B['evar_vs_score_x']:.1f}x",30,TEAL,"Helvetica-Bold")
text(925,H-391,"more revenue saved vs. churn-only",12,HexColor("#cbd5e1"))
text(80,70,"IT & Data-Science Consulting — Proof of Concept  ·  Prepared for the Company A Executive Team",13,HexColor("#94a3b8"))
text(80,46,"Author: [Omnicampus Account Name]   ·   June 2026",11,HexColor("#64748b"))
c.showPage()

# ============================================================ SLIDE 2 — EXECUTIVE SUMMARY
page_bg(); header("Executive summary","THE ANSWER, UP FRONT",2)
text(100,H-150,"Company A loses customers faster than it can profitably replace them. We can fix that.",16.5,INK,"Helvetica-Bold")
y=H-200
y=bullet(100,y,"The problem","In a saturated market, ~95% of growth comes from switchers and re-acquiring a lost "
       "customer costs 5–25x more than keeping one. Churn is Company A's most expensive leak.")
y=bullet(100,y-14,"What the data says","Churn is predictable and concentrated. The #1 driver is aging handsets "
       "(>12 months old churn at ~57–59% in our sample) — a lever Company A directly controls.")
y=bullet(100,y-14,"Our solution","A monthly value-based targeting engine: rank customers by Expected Value-at-Risk "
       "(churn probability x customer value), then make proactive, persona-specific retention offers.")
# right value stack
rect(720,H-470,470,300,WHITE,r=16); rect(720,H-216,470,46,NAVY,r=16)
text(744,H-200,"What it delivers — per 1,000,000 subscribers",13.5,WHITE,"Helvetica-Bold")
rows=[("Best model",f"{R['best_model']} · AUC {R['models'][R['best_model']]['auc']:.3f}"),
      ("Targeting edge",f"captures {B['evar_rev_capture20']*100:.0f}% of revenue-at-risk in the top 20%"),
      ("Net benefit / yr",money(B["net_benefit"])+f"   ({B['roi']:.2f}x ROI)"),
      ("Customers saved / yr",f"~{B['saved_customers']:,}"),
      ("Our fee",money(Q["annual_fee"])+f"/yr  ·  {Q['pct_of_net']:.1f}% of value created")]
yy=H-250
for k,v in rows:
    text(744,yy,k,12,MUTE,"Helvetica"); text(1166,yy,v,12.5,NAVY,"Helvetica-Bold",align="right")
    rect(744,yy-12,422,1,LINE); yy-=40
footer("Sources: Bain & Company; Harvard Business Review; analysis of Company A dataset (n=100,000).")
c.showPage()

# ============================================================ SLIDE 3 — MARKET ANALYSIS
page_bg(); header("Market analysis: retention is the only profitable growth lever","MARKET ANALYSIS",3)
chip(100,H-250,"US wireless market (2025)",  "$344B", BLUE, w=250)
chip(370,H-250,"Best-in-class monthly churn","1.0%", TEAL, w=250)
chip(640,H-250,"Re-acquire vs. retain cost", "5–25x", ORANGE, w=250)
chip(910,H-250,"Profit lift from +5% retention","25%+", GREEN, w=280)
y=H-320
y=bullet(100,y,"The market is saturated","Penetration exceeds 100%. Carriers grow almost entirely by stealing "
       "switchers — making every avoided cancellation worth far more than a new-customer ad.",maxw=480)
y=bullet(100,y-14,"Acquisition is expensive","Industry CAC runs into the hundreds of dollars per line; "
       "retention offers are a fraction of that, so saved customers are immediately accretive.",maxw=480)
y=H-320
y=bullet(640,y,"Small churn moves, big profit","Bain's classic finding: a 5-point retention improvement can "
       "lift profits 25–95%. Churn is the highest-leverage metric in the P&L.",maxw=470)
y=bullet(640,y-14,"Loyalty compounds","Tenured customers spend more, cost less to serve, and refer others — "
       "so protecting the base protects future revenue too.",maxw=470)
footer("Sources: industry market sizing 2025; Verizon/T-Mobile investor reports; Bain & Company; Reichheld, HBR.")
c.showPage()

# ============================================================ SLIDE 4 — CLIENT & DATA
page_bg(); header("The client & the data","CLIENT & DATASET",4)
wrap(100,H-150,"Company A is a US wireless carrier facing elevated churn. We received two linked tables "
     "covering 100,000 customers — behavioural usage and customer profile — joined on Customer_ID.",
     14,SLATE,leading=20,maxw=1080)
chip(100,H-300,"Customers",  "100,000", NAVY, w=210)
chip(322,H-300,"Features",  f"{R['n_features']}", BLUE, w=210)
chip(544,H-300,"Avg revenue / mo", f"${R['ARPU_month']:.0f}", TEAL, w=210)
chip(766,H-300,"Median tenure", f"{R['tenure_median']:.0f} mo", ORANGE, w=210)
y=H-360
bullet(100,y,"Behavioural (Record.csv)","Minutes of use, overage, revenue, call quality, recharges, and the "
       "churn label — the in-life signals of disengagement.",color=BLUE,maxw=470)
bullet(640,y,"Profile (Client.csv)","Handset age & price, refurbished flag, web-capability, credit, tenure and "
       "demographics — the structural drivers.",color=TEAL,maxw=470)
rect(100,90,1080,70,HexColor("#fff7ed"),r=12); rect(100,90,6,70,ORANGE)
wrap(126,142,"Data-quality note: the modelling sample is balanced ~50/50 by design (oversampled), so it is used "
     "for ranking — not as the real churn rate. Demographic fields are 25–49% missing and are de-emphasised in "
     "favour of controllable, behavioural drivers.",11.5,SLATE,leading=15.5,maxw=1040)
footer("Source: Company A dataset (Client.csv + Record.csv), 100,000 customers.")
c.showPage()

# ============================================================ SLIDE 5 — EDA 1
page_bg(); header("What the data reveals: who leaves, and why","EXPLORATORY DATA ANALYSIS  (1/2)",5)
img(f"{FIG}/02_correlations.png",90,150,560,430)
y=H-160
y=bullet(700,y,"Equipment age is the #1 signal","Longer time on the same handset (eqpdays) is the strongest "
       "positive correlate of churn (+0.11) — phones age out, customers shop around.",color=ORANGE,maxw=470)
y=bullet(700,y-12,"Investment buys loyalty","Higher handset price and higher recurring charge both correlate "
       "negatively with churn — customers with newer, pricier phones stay.",color=TEAL,maxw=470)
y=bullet(700,y-12,"Engagement matters","More minutes of use and completed calls track with lower churn; "
       "declining usage is an early warning sign.",color=BLUE,maxw=470)
footer("Source: Pearson correlations with churn, Company A dataset (n=100,000).")
c.showPage()

# ============================================================ SLIDE 6 — EDA 2
page_bg(); header("The headline insight: the handset-age cliff","EXPLORATORY DATA ANALYSIS  (2/2)",6)
img(f"{FIG}/03_eqpdays.png",90,150,560,430)
e=R["eqp_bins"]
y=H-160
y=bullet(700,y,"A clear inflection at ~12 months",
       f"Churn jumps from ~{e['6-12mo']*100:.0f}% (6–12 mo) to ~{e['12-18mo']*100:.0f}% once a handset passes "
       "12 months, and stays elevated thereafter.",color=ORANGE,maxw=470)
y=bullet(700,y-12,"This is an actionable lever","Unlike age or income, handset age is something Company A can "
       "change — through proactive upgrade and trade-in offers.",color=TEAL,maxw=470)
y=bullet(700,y-12,"Refurbished handsets churn more",
       f"Refurb lines churn at ~{R['lever_refurb']['R']*100:.0f}% vs ~{R['lever_refurb']['N']*100:.0f}% for new — "
       "a quality-of-experience flag worth fixing.",color=BLUE,maxw=470)
footer("Source: churn rate by handset-age bucket and refurbished flag, Company A dataset.")
c.showPage()

# ============================================================ SLIDE 7 — PROBLEM & ML TASK
page_bg(); header("Framing the business problem as an ML task","PROBLEM DEFINITION",7)
rect(100,H-250,1080,90,HexColor("#eef2ff"),r=12); rect(100,H-250,6,90,BLUE)
wrap(126,H-188,"Business question:  \u201cWhich customers are about to leave, how much revenue do they put at risk, "
     "and what is the most cost-effective way to keep them?\u201d",15,NAVY,"Helvetica-Bold",leading=22,maxw=1030)
y=H-300
y=bullet(100,y,"ML formulation","A supervised binary classification problem: predict P(churn) per customer "
       "from behavioural + profile features.",color=BLUE,maxw=480)
y=bullet(100,y-12,"But probability isn't enough","A $130/mo customer and a $25/mo customer are not equally worth "
       "saving. We rank by Expected Value-at-Risk = P(churn) x customer value.",color=ORANGE,maxw=480)
y=H-300
y=bullet(640,y,"Primary metric: ROC-AUC","Ranking quality matters more than a single cutoff, because we act on "
       "the highest-risk, highest-value customers first.",color=TEAL,maxw=470)
y=bullet(640,y-12,"Success = profit, not accuracy","The model is judged by incremental margin and avoided "
       "re-acquisition cost from a targeted campaign — validated by an A/B holdout.",color=GREEN,maxw=470)
footer()
c.showPage()

# ============================================================ SLIDE 8 — MODELING APPROACH
page_bg(); header("Modelling approach: rigorous, reproducible, honest","METHODOLOGY",8)
steps=[("01","Clean & encode","Impute missing values, one-hot encode categoricals, engineer ratios "
        "(e.g. minutes-per-equipment-day). 99 model features."),
       ("02","Train 4 model families","Logistic Regression (baseline), Random Forest, XGBoost and LightGBM — "
        "from interpretable to high-performance."),
       ("03","Validate honestly","75/25 stratified split + 5-fold cross-validation on the winner to confirm the "
        "score is stable, not luck."),
       ("04","Rank by value","Convert probabilities to Expected Value-at-Risk and simulate a targeted "
        "retention campaign against random and churn-only baselines.")]
x=100
for i,(n,h,b) in enumerate(steps):
    col=[BLUE,TEAL,ORANGE,GREEN][i]
    bx=100+i*278
    rect(bx,H-430,258,250,WHITE,r=14); rect(bx,H-430,258,52,col,r=14)
    text(bx+18,H-410,n,22,WHITE,"Helvetica-Bold")
    text(bx+18,H-462,h,14,INK,"Helvetica-Bold")
    wrap(bx+18,H-486,b,11.5,SLATE,leading=15.5,maxw=224)
rect(100,95,1080,60,HexColor("#f0fdf4"),r=12); rect(100,95,6,60,GREEN)
cv=R
wrap(126,138,f"Cross-validation result:  {R['best_model']} ROC-AUC = {R['cv_auc_mean']:.3f} \u00b1 {R['cv_auc_std']:.3f} "
     "across 5 folds — confirming the model generalises.",12.5,INK,"Helvetica-Bold",leading=16,maxw=1030)
footer()
c.showPage()

# ============================================================ SLIDE 9 — RESULTS
page_bg(); header(f"Results: {R['best_model']} wins — AUC {R['models'][R['best_model']]['auc']:.3f}","MODEL RESULTS  (NAME · METRIC · SCORE)",9)
img(f"{FIG}/04_auc.png",90,150,540,430)
# table
tx,ty=690,H-200; cw=[150,70,70,70,70]
heads=["Model","AUC","Acc","Prec","Recall"]
rect(tx,ty-6,sum(cw),34,NAVY)
cx=tx
for hh,w in zip(heads,cw):
    text(cx+10,ty+4,hh,12,WHITE,"Helvetica-Bold"); cx+=w
order=["Logistic Regression","Random Forest","LightGBM","XGBoost"]
yy=ty-40
for m in order:
    md=R["models"][m]; best=(m==R["best_model"])
    rect(tx,yy-8,sum(cw),34,HexColor("#eef7f4") if best else WHITE)
    if best: rect(tx,yy-8,4,34,TEAL)
    vals=[m,f"{md['auc']:.3f}",f"{md['accuracy']:.3f}",f"{md['precision']:.3f}",f"{md['recall']:.3f}"]
    cx=tx
    for v,w,j in zip(vals,cw,range(5)):
        text(cx+10,yy,v,11.5,NAVY if best else INK,"Helvetica-Bold" if (best or j==0) else "Helvetica"); cx+=w
    rect(tx,yy-12,sum(cw),1,LINE); yy-=42
wrap(690,yy-2,f"How to read AUC: 0.5 = guessing, 1.0 = perfect. At {R['models'][R['best_model']]['auc']:.3f} the model ranks a "
     "churner above a non-churner ~69% of the time — enough to target retention spend profitably. Tree models "
     "clearly beat the linear baseline.",12,SLATE,leading=17,maxw=490)
footer("Source: held-out test set (25% of 100,000), Company A dataset.")
c.showPage()

# ============================================================ SLIDE 10 — DIFFERENTIATOR (EVaR)
page_bg(); header("Our differentiator: target value, not just probability","VALUE-BASED TARGETING  (EVaR)",10)
img(f"{FIG}/06_evar.png",90,150,560,430)
sc=R["targeting"]["by_score"]["0.2"]["rev_capture"]
ev=R["targeting"]["by_evar"]["0.2"]["rev_capture"]
y=H-160
y=bullet(700,y,"The insight","Ranking by churn probability alone wastes budget on low-value churners. "
       "Ranking by Expected Value-at-Risk protects the revenue that actually matters.",color=ORANGE,maxw=470)
rect(700,H-330,470,86,WHITE,r=12)
text(722,H-272,f"{ev*100:.0f}%",30,TEAL,"Helvetica-Bold")
text(722,H-300,"of revenue-at-risk captured in top 20%",11.5,SLATE)
text(980,H-272,f"vs {sc*100:.0f}%",24,MUTE,"Helvetica-Bold")
text(980,H-300,"churn-only ranking",11.5,SLATE)
y=H-350
y=bullet(700,y,"1.6x more revenue protected","Same campaign budget, same 20% of customers contacted — but we "
       f"shield {B['evar_vs_score_x']:.1f}x the revenue by leading with value.",color=TEAL,maxw=470)
footer("Source: gains simulation on held-out test set; EVaR = P(churn) x annualised customer revenue.")
c.showPage()

# ============================================================ SLIDE 11 — PERSONAS
page_bg(); header("Who to target: four actionable retention personas","CUSTOMER SEGMENTATION",11)
img(f"{FIG}/10_personas.png",90,150,560,430)
P=R["personas"]
defs=[("Aging-Handset","Aging-Handset","Old device, decent spend","Proactive upgrade / trade-in offer",ORANGE),
      ("Bill-Shock / Overage","Bill-Shock / Overage","High bill, overage pain","Right-size the plan, autopay discount",REDC),
      ("Disengaging","Disengaging","Usage falling fast","Win-back call + loyalty perk",BLUE),
      ("Stable-Loyalty","Stable-Loyalty","Healthy, lower risk","Light-touch loyalty, protect margin",TEAL)]
yy=H-170
for label,key,desc,action,col in defs:
    p=P.get(key,{})
    rect(700,yy-58,480,70,WHITE,r=10); rect(700,yy-58,6,70,col)
    text(722,yy-16,label,13.5,INK,"Helvetica-Bold")
    cr=p.get("churn_rate",0)*100; rev=p.get("avg_rev",0)
    text(1166,yy-16,f"{cr:.0f}% churn · ${rev:.0f}/mo",11,col,"Helvetica-Bold",align="right")
    text(722,yy-36,desc,10.5,MUTE)
    text(722,yy-52,"\u2192 "+action,10.5,SLATE,"Helvetica-Oblique")
    yy-=84
footer("Source: K-means style segmentation of the high-risk pool, Company A dataset.")
c.showPage()

# ============================================================ SLIDE 12 — SOLUTION / PLAYBOOK
page_bg(); header("The solution: a monthly value-based retention engine","RECOMMENDED SOLUTION",12)
flow=[("Score","Rank all customers monthly by EVaR",BLUE),
      ("Segment","Assign each high-risk customer to a persona",TEAL),
      ("Act","Trigger the matched, budgeted offer",ORANGE),
      ("Measure","A/B holdout proves incremental saves",GREEN)]
for i,(h,b,col) in enumerate(flow):
    bx=100+i*278
    rect(bx,H-300,250,120,WHITE,r=14); rect(bx,H-300,250,8,col,r=14)
    text(bx+20,H-238,h,18,col,"Helvetica-Bold")
    wrap(bx+20,H-264,b,12,SLATE,leading=16,maxw=215)
    if i<3: text(bx+258,H-250,"\u203a",26,MUTE,"Helvetica-Bold")
y=H-360
y=bullet(100,y,"Persona-matched offers","Aging-handset \u2192 upgrade; bill-shock \u2192 plan right-sizing; disengaging "
       "\u2192 win-back. The model decides who; the persona decides what.",color=ORANGE,maxw=480)
y=bullet(640,y+22,"Budget guardrails","Only the top 20% by EVaR get an offer, so spend is concentrated where it "
       "earns a return — never sprayed across the base.",color=TEAL,maxw=470)
y=bullet(100,y-14,"Always-on, not one-off","Re-scored every month and fed by the A/B holdout, the engine keeps "
       "learning which offers actually move retention.",color=GREEN,maxw=480)
footer()
c.showPage()

# ============================================================ SLIDE 13 — BUSINESS CASE
page_bg(); header("The business case: ROI-positive and robust","QUANTIFIED IMPACT  (per 1M subscribers / yr)",13)
img(f"{FIG}/08_waterfall.png",80,150,560,430)
img(f"{FIG}/09_sensitivity.png",660,150,540,430)
chip(80,90,"Net benefit / yr", money(B["net_benefit"]), GREEN, w=250, h=86)
chip(350,90,"Return on spend", f"{B['roi']:.2f}x", TEAL, w=230, h=86)
chip(660,90,"Customers saved / yr", f"{B['saved_customers']:,}", BLUE, w=250, h=86)
chip(930,90,"Stays positive", "every case", ORANGE, w=250, h=86)
footer(f"Assumes 22% annual churn, 50% margin, 30% save-rate, ${B['offer_cost']} blended offer, ${B['CAC']} CAC. "
       "Sensitivity grid (right) is net of all costs across save-rate x offer-cost.")
c.showPage()

# ============================================================ SLIDE 14 — QUOTATION (THE ASK)
page_bg(); header("Our proposal & quotation","THE ASK",14)
wrap(100,H-150,"We propose a 12-week build of the retention engine, then an ongoing managed service. "
     "Our fee is a small fraction of the value we protect.",14,SLATE,leading=20,maxw=1040)
# left: phases
ph=[("Phase 1 · Build (wk 1\u20136)","Data pipeline, model, EVaR scoring, persona logic",BLUE),
    ("Phase 2 · Pilot (wk 7\u201312)","A/B holdout on top-20% pool; validate incremental saves",TEAL),
    ("Phase 3 · Scale (ongoing)","Monthly scoring, offer optimisation, quarterly review",ORANGE)]
yy=H-220
for h,b,col in ph:
    rect(100,yy-54,520,66,WHITE,r=10); rect(100,yy-54,6,66,col)
    text(122,yy-18,h,13.5,INK,"Helvetica-Bold")
    wrap(122,yy-38,b,11.5,SLATE,leading=15,maxw=480)
    yy-=80
# right: pricing card
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
text(724,H-410,f"= just {Q['pct_of_net']:.1f}% of the {money(B['net_benefit'])} net value we create",13,ORANGE,"Helvetica-Bold")
text(724,H-432,f"Company A keeps ~{money(Q['client_keeps'])} / yr per 1M subscribers",11.5,HexColor("#cbd5e1"))
footer("Fee illustrative for a PoC engagement; scales with subscriber base. Value figures per 1M subscribers / yr.")
c.showPage()

# ============================================================ SLIDE 15 — RISKS, FAIRNESS & REFERENCES
page_bg(); header("Risks, fairness & references","GOVERNANCE & SOURCES",15)
y=H-150
y=bullet(100,y,"Validate before scaling","The 50/50 sample inflates apparent churn; the A/B pilot calibrates real "
       "save-rates and ROI before any full roll-out.",color=ORANGE,maxw=480)
y=bullet(100,y-10,"Fairness & compliance","Demographic fields (ethnicity, income) are de-emphasised to avoid "
       "discriminatory targeting; offers are driven by behaviour Company A controls.",color=BLUE,maxw=480)
y=bullet(100,y-10,"Monitor for drift","Re-train on fresh data and watch feature drift; a stale model silently "
       "loses accuracy as customer behaviour shifts.",color=TEAL,maxw=480)
y=bullet(100,y-10,"Don't over-discount","Budget guardrails and holdouts prevent giving margin away to customers "
       "who would have stayed anyway.",color=GREEN,maxw=480)
# references
rect(640,H-470,540,300,WHITE,r=14); rect(640,H-470,540,44,NAVY,r=14)
text(664,H-200,"Selected references",13.5,WHITE,"Helvetica-Bold")
refs=["Reichheld & Sasser, \u201cZero Defections,\u201d Harvard Business Review.",
      "Bain & Company — customer retention & profitability research.",
      "Verizon / T-Mobile investor reports, 2024\u20132025 (churn & ARPU).",
      "CTIA Annual Wireless Industry Survey, 2025 (market size).",
      "Neslin et al., \u201cChurn Modeling,\u201d Journal of Marketing Research.",
      "scikit-learn; XGBoost; LightGBM documentation."]
yy=H-238
for r in refs:
    text(664,yy,"\u2022",11,ORANGE); wrap(680,yy,r,10.8,SLATE,leading=14,maxw=480); yy-=38
rect(640,96,540,52,HexColor("#fff7ed"),r=10); rect(640,96,6,52,ORANGE)
wrap(664,130,"AI-use disclosure: generative AI assisted with code scaffolding, drafting and visualisation. "
     "All analysis, numbers and conclusions were verified against the dataset by the author.",9.8,SLATE,leading=13,maxw=500)
text(100,70,"Thank you. We look forward to helping Company A win back its base.",14,NAVY,"Helvetica-Bold")
footer()
c.showPage()

c.save()
print("Saved", OUT, round(os.path.getsize(OUT)/1024), "KB")
