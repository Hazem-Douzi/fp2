"""
Builds the 15-slide business-proposal PDF deck (16:9) for Company A telecom churn.
Output: deliverables/Company_A_Business_Proposal.pdf
"""
import os
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from PIL import Image as PILImage

W, H = 1280, 720
FIG = "/vercel/share/v0-project/deliverables/figs"
OUT = "/vercel/share/v0-project/deliverables/Company_A_Business_Proposal.pdf"

NAVY=HexColor("#1f3a5f"); BLUE=HexColor("#2563eb"); ORANGE=HexColor("#f97316")
TEAL=HexColor("#0d9488"); SLATE=HexColor("#475569"); LIGHT=HexColor("#f1f5f9")
WHITE=HexColor("#ffffff"); INK=HexColor("#0f172a"); MUTE=HexColor("#64748b")
LINE=HexColor("#e2e8f0"); GREEN=HexColor("#059669"); REDC=HexColor("#dc2626")

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
    text(100,H-80,title,26,NAVY,"Helvetica-Bold")
    text(W-80,H-78,f"{num} / 15",12,MUTE,"Helvetica",align="right")
    rect(80,H-100,W-160,1.4,LINE)

def footer(cite=None):
    rect(0,0,W,40,WHITE)
    text(80,16,"Company A  ·  Customer Retention PoC",10,MUTE)
    text(W-80,16,"GCI World 2026 — Final Assignment",10,MUTE,align="right")
    if cite:
        text(80,52,cite,9.5,MUTE,"Helvetica-Oblique")

def page_bg():
    rect(0,0,W,H,HexColor("#f8fafc"))

def chip(x,y,label,value,color,w=250,h=104):
    rect(x,y,w,h,WHITE,r=14)
    rect(x,y,6,h,color,r=0)
    text(x+24,y+h-30,value,30,color,"Helvetica-Bold")
    wrap(x+24,y+h-58,label,11.5,SLATE,"Helvetica",leading=14,maxw=w-40)

# ============================================================ SLIDE 1 — TITLE
rect(0,0,W,H,NAVY)
rect(0,0,W,8,ORANGE)
rect(80,H-180,70,6,ORANGE)
text(80,H-240,"Winning Back the Base",46,WHITE,"Helvetica-Bold")
text(80,H-292,"A Data-Driven Customer Retention Strategy for Company A",24,HexColor("#cbd5e1"),"Helvetica")
wrap(80,H-360,"Predicting and preventing telecom churn with machine learning — turning 100,000 customer records into a quantified, ROI-positive retention program.",16,HexColor("#94a3b8"),"Helvetica",leading=24,maxw=760)
# right metric panel
rect(900,H-470,300,300,HexColor("#274b73"),r=18)
text(925,H-225,"0.695",44,ORANGE,"Helvetica-Bold"); text(925,H-250,"XGBoost ROC-AUC",13,HexColor("#cbd5e1"))
text(925,H-300,"$22.6M",30,WHITE,"Helvetica-Bold"); text(925,H-323,"net benefit / yr (per 1M subs)",12,HexColor("#cbd5e1"))
text(925,H-370,"2.57x",30,TEAL,"Helvetica-Bold"); text(925,H-393,"return on retention spend",12,HexColor("#cbd5e1"))
text(80,70,"IT Consulting — Proof of Concept  ·  Prepared for Company A Executive Team",13,HexColor("#94a3b8"))
text(80,46,"Author: [Omnicampus Account Name]   ·   June 2026",11,HexColor("#64748b"))
c.showPage()

# ============================================================ SLIDE 2 — AGENDA
page_bg(); header("Agenda","WHAT WE WILL COVER",2)
items=[("01","Market analysis","Telecom churn economics & why retention wins"),
       ("02","The client & data","100k customers across usage + profile tables"),
       ("03","Exploratory analysis","What the data reveals about who leaves"),
       ("04","Problem & ML task","Framing churn as a prediction problem"),
       ("05","Models & results","Three models, named metrics and scores"),
       ("06","Business proposal","The solution and its quantified impact"),
       ("07","Roadmap & risks","How we deploy and de-risk the program")]
x0,y0=110,H-200; cw,ch=540,58
for i,(n,t,d) in enumerate(items):
    col=i//4; row=i%4
    x=x0+col*560; y=y0-row*70
    rect(x,y-ch+18,cw,ch,WHITE,r=12)
    rect(x,y-ch+18,46,ch,NAVY,r=0)
    text(x+12,y-22,n,20,WHITE,"Helvetica-Bold")
    text(x+62,y-12,t,16,NAVY,"Helvetica-Bold")
    text(x+62,y-34,d,11.5,SLATE)
footer()
c.showPage()

# ============================================================ SLIDE 3 — MARKET
page_bg(); header("A saturated market where growth means stealing share","MARKET ANALYSIS",3)
chip(80,H-230,"U.S. wireless market size (2025)","$344B",BLUE,w=250)
chip(345,H-230,"Postpaid monthly churn, best-in-class","~1.0%",TEAL,w=250)
chip(610,H-230,"Cost to acquire vs. retain a customer","5–25x",ORANGE,w=250)
chip(875,H-230,"Profit lift from +5% retention (Bain)","25–95%",NAVY,w=260)
y=H-300
y=wrap(80,y,"The U.S. wireless market is large but saturated: carriers compete for a fixed pool of subscribers, so almost all net growth comes from winning competitors' customers rather than new demand. In that environment, every customer who leaves must be re-acquired at 5–25x the cost of keeping them.",15,INK,"Helvetica",leading=23,maxw=1120)
y=wrap(80,y-6,"Industry research is unambiguous: small improvements in retention compound into large profit gains. A 5% increase in retention can lift profits 25–95% (Reichheld / Bain), and customer-level retention models have been shown to materially increase carrier profitability. Retention — not acquisition — is the highest-ROI growth lever available to Company A today.",15,INK,"Helvetica",leading=23,maxw=1120)
footer("Sources: IBISWorld U.S. Wireless Telecom 2025; Verizon/T-Mobile investor reports 2024-25; Reichheld & Bain & Co.; Harvard Business Review. Full list on slide 15.")
c.showPage()

# ============================================================ SLIDE 4 — CLIENT & DATA
page_bg(); header("The client and the data we were given","THE ENGAGEMENT",4)
y=wrap(80,H-150,"Company A is an anonymized telecommunications provider. It holds a large customer dataset but limited in-house analytics capacity. For this Proof of Concept we received two linked tables covering ~100,000 customers:",14.5,INK,"Helvetica",leading=22,maxw=1120)
rect(80,H-360,540,150,WHITE,r=14)
text(105,H-242,"Client.csv — who the customer is",16,NAVY,"Helvetica-Bold")
wrap(105,H-272,"Demographics & account profile: handset price, device age, credit class, tenure, plan attributes, household data. 50 columns.",12.5,SLATE,leading=18,maxw=490)
text(105,H-330,"~100,000 rows",20,BLUE,"Helvetica-Bold")
rect(650,H-360,550,150,WHITE,r=14)
text(675,H-242,"Record.csv — how they behave",16,NAVY,"Helvetica-Bold")
wrap(675,H-272,"Usage history: monthly revenue, minutes of use, recurring charges, overage, call quality — and the churn label. 51 columns.",12.5,SLATE,leading=18,maxw=500)
text(675,H-330,"+ the churn flag",20,ORANGE,"Helvetica-Bold")
rect(80,H-470,1120,84,LIGHT,r=14)
text(105,H-410,"Tables join on Customer_ID  →  one 100,000 × ~100 modeling table.",15,NAVY,"Helvetica-Bold")
wrap(105,H-438,"Real-world constraint: some columns are undocumented and many profile fields are sparse. Handling that ambiguity is part of the engagement — we lean on the well-populated, business-meaningful features.",12,SLATE,leading=17,maxw=1070)
footer("Source: Company A Dataset Overview (provided). Analysis: authors' merge of Client.csv + Record.csv.")
c.showPage()

# ============================================================ SLIDE 5 — EDA balance + arpu
page_bg(); header("Each customer is worth ~$59/month — and they don't stay long","EXPLORATORY DATA ANALYSIS (1/3)",5)
img(f"{FIG}/01_balance.png",80,180,360,400)
img(f"{FIG}/02_arpu_tenure.png",450,210,760,360)
rect(80,90,1120,70,LIGHT,r=12)
wrap(105,140,"Takeaway: the modeling sample is deliberately balanced 50/50 (not the real churn rate). Average revenue per user is $58.72/mo and median tenure is just 16 months — so losing a customer early destroys significant lifetime value. This is the value at stake.",12.5,NAVY,"Helvetica",leading=17,maxw=1070)
footer("Analysis: authors' EDA on merged dataset (n=100,000). ARPU = rev_Mean.")
c.showPage()

# ============================================================ SLIDE 6 — EDA drivers
page_bg(); header("Aging handsets drive churn; newer / pricier devices retain","EXPLORATORY DATA ANALYSIS (2/3)",6)
img(f"{FIG}/03_drivers.png",80,150,640,430)
y=H-200
text(760,y,"What the correlations tell us",16,NAVY,"Helvetica-Bold")
y-=34
for t in ["Days on current handset is the strongest churn-increasing signal.",
          "Handset price & monthly charge are the strongest churn-reducing signals.",
          "Higher engagement (minutes of use) tracks with staying.",
          "Customers with more/older equipment are flight risks."]:
    rect(760,y-4,10,10,ORANGE); y=wrap(782,y+4,t,13,INK,"Helvetica",leading=18,maxw=420); y-=4
rect(760,150,420,120,LIGHT,r=12)
wrap(782,250,"Hypothesis: customers stuck on old devices feel \u201cstale\u201d and are courted by rivals' upgrade offers. Device lifecycle is a lever Company A directly controls.",12.5,NAVY,leading=18,maxw=380)
footer("Analysis: authors' point-biserial correlations of numeric features with churn.")
c.showPage()

# ============================================================ SLIDE 7 — EDA key insight
page_bg(); header("The headline insight: churn rises monotonically with handset age","EXPLORATORY DATA ANALYSIS (3/3)",7)
img(f"{FIG}/04_eqpdays.png",80,170,700,410)
chip(820,H-250,"Churn, handsets < 6 months old","lowest",TEAL,w=360,h=92)
chip(820,H-352,"Churn, handsets 24+ months old","highest",ORANGE,w=360,h=92)
rect(820,150,360,150,LIGHT,r=12)
wrap(842,278,"Why it matters",13.5,NAVY,"Helvetica-Bold",leading=18)
wrap(842,252,"Equipment age is observable, predictable, and actionable. We can see a customer aging into the danger zone months ahead — and intervene with an upgrade offer before a competitor does.",12,SLATE,leading=17,maxw=320)
footer("Analysis: authors' EDA — churn rate by eqpdays (days on current handset) bins.")
c.showPage()

# ============================================================ SLIDE 8 — PROBLEM
page_bg(); header("From insight to a precise machine-learning question","PROBLEM DEFINITION",8)
rect(80,H-260,1120,86,WHITE,r=14)
rect(80,H-260,6,86,ORANGE)
wrap(110,H-205,"Business objective:  reduce voluntary churn by identifying at-risk, high-value customers early enough to act.",17,NAVY,"Helvetica-Bold",leading=24,maxw=1060)
cards=[("ML task","Supervised binary classification","Predict P(churn) for every customer.",BLUE),
       ("Target variable","churn  (1 = left, 0 = stayed)","Chosen from EDA + market: retention is the #1 profit lever.",NAVY),
       ("Primary metric","ROC-AUC","Ranks customers by risk, threshold-free, robust to the balanced sample.",TEAL),
       ("Operating metric","Lift / gains","Who to call first — the metric the campaign actually runs on.",ORANGE)]
x=80
for t,h,d,col in cards:
    rect(x,H-470,266,180,WHITE,r=14); rect(x,H-470,266,8,col)
    text(x+20,H-322,t.upper(),11,col,"Helvetica-Bold")
    wrap(x+20,H-348,h,15,NAVY,"Helvetica-Bold",leading=19,maxw=226)
    wrap(x+20,H-400,d,12,SLATE,leading=17,maxw=226)
    x+=284
footer("Rationale grounded in EDA (slides 5-7) and market analysis (slide 3).")
c.showPage()

# ============================================================ SLIDE 9 — APPROACH
page_bg(); header("How we built it — explained without the jargon","MODELING APPROACH",9)
steps=[("1","Clean & join","Merge the two tables on Customer_ID; remove IDs to avoid cheating."),
       ("2","Engineer features","Impute missing values; scale numbers; encode categories — 250+ model inputs."),
       ("3","Train 3 models","A simple baseline and two advanced tree models, on 75% of data."),
       ("4","Test fairly","Score the held-out 25% the model never saw — honest performance.")]
x=80
for n,t,d in steps:
    rect(x,H-330,266,140,WHITE,r=14)
    rect(x+20,H-250,40,40,NAVY,r=20); text(x+40,H-238,n,18,WHITE,"Helvetica-Bold",align="center")
    text(x+72,H-232,t,14.5,NAVY,"Helvetica-Bold")
    wrap(x+20,H-280,d,11.5,SLATE,leading=16,maxw=226)
    if x<900:
        text(x+274,H-262,"\u2192",22,MUTE)
    x+=284
rect(80,H-450,1120,90,LIGHT,r=14)
text(105,H-390,"Why these models?",14,NAVY,"Helvetica-Bold")
wrap(105,H-416,"Logistic Regression gives a transparent, explainable baseline. Random Forest and XGBoost capture non-linear interactions (e.g. \u201cold handset AND low usage\u201d) that drive churn. XGBoost is the industry standard for tabular data — so we let the data pick the winner rather than assuming one.",12.5,SLATE,leading=18,maxw=1070)
footer("Method: scikit-learn pipelines + XGBoost. Reproducible in the accompanying notebook. Ref: Chen & Guestrin 2016 (XGBoost).")
c.showPage()

# ============================================================ SLIDE 10 — RESULTS
page_bg(); header("Results: XGBoost is the best model — AUC 0.695","MODEL RESULTS  (NAME · METRIC · SCORE)",10)
img(f"{FIG}/05_roc.png",80,150,560,430)
# table
tx,ty=690,H-200; rw=470
rect(tx,ty-250,rw,300,WHITE,r=14)
cols=["Model","AUC","Precision","Recall","F1"]
cw=[170,80,80,75,65]
text(tx+18,ty-30,"Model performance (held-out test set)",13,NAVY,"Helvetica-Bold")
xx=tx+18
for i,h in enumerate(cols):
    text(xx,ty-58,h,11,MUTE,"Helvetica-Bold"); xx+=cw[i]
rows=[("Logistic Regression","0.631","0.592","0.591","0.592",False),
      ("Random Forest","0.671","0.612","0.639","0.625",False),
      ("XGBoost","0.695","0.632","0.646","0.639",True)]
ry=ty-86
for name,a,p,r,f,best in rows:
    if best: rect(tx+8,ry-8,rw-16,30,HexColor("#fff7ed"),r=8)
    vals=[name,a,p,r,f]; xx=tx+18
    for i,v in enumerate(vals):
        text(xx,ry,v,12.5,(ORANGE if best else INK),("Helvetica-Bold" if best else "Helvetica")); xx+=cw[i]
    ry-=34
text(tx+18,ry-2,"Winner: XGBoost — chosen for deployment.",12.5,GREEN,"Helvetica-Bold")
rect(690,150,470,120,LIGHT,r=12)
wrap(712,250,"How to read AUC:  0.5 = random guessing, 1.0 = perfect. At 0.695 the model reliably ranks a churner above a non-churner ~70% of the time — strong enough to target retention spend profitably.",12.5,NAVY,leading=18,maxw=430)
footer("Metric: ROC-AUC (primary) + precision/recall/F1. Scores from authors' notebook, 25% hold-out, seed=42.")
c.showPage()

# ============================================================ SLIDE 11 — LIFT + importance
page_bg(); header("The model tells us WHO to call — and WHY they're leaving","TURNING SCORES INTO ACTION",11)
img(f"{FIG}/07_lift.png",80,310,560,300)
img(f"{FIG}/06_importance.png",640,300,560,310)
rect(80,90,1120,180,LIGHT,r=14)
text(105,238,"Two model outputs the business can act on directly:",14,NAVY,"Helvetica-Bold")
wrap(105,210,"1)  LIFT  — Ranking customers by risk and contacting only the top 30% reaches 42% of all churners. That is up to ~1.6x more efficient than blanket outreach, so every retention dollar works harder.",12.5,INK,leading=18,maxw=1070)
wrap(105,150,"2)  DRIVERS  — The top predictors (handset age, new-vs-refurbished device, tenure, device web-capability) are exactly the levers Company A controls. The model doesn't just flag risk — it points to the fix: proactive device upgrades.",12.5,INK,leading=18,maxw=1070)
footer("Analysis: authors' lift/gains on test set; XGBoost feature importances.")
c.showPage()

# ============================================================ SLIDE 12 — PROPOSAL
page_bg(); header("The proposal: a model-driven proactive retention engine","BUSINESS PROPOSAL",12)
text(80,H-150,"Score every customer monthly, then act on the highest-risk, highest-value segment before they leave.",15,NAVY,"Helvetica-Bold")
props=[("Predict","Run the XGBoost model monthly across the full base to produce a churn-risk score for every customer.",BLUE),
       ("Prioritize","Rank by risk \u00d7 value. Focus retention budget on the top deciles — where 42% of churn concentrates.",NAVY),
       ("Intervene","Trigger the right offer: proactive handset upgrades for aging-device customers, loyalty perks for the rest.",ORANGE),
       ("Measure & learn","Run as a treatment-vs-holdout A/B test to prove incremental saves, then scale what works.",TEAL)]
x=80
for t,d,col in props:
    rect(x,H-430,266,210,WHITE,r=14); rect(x,H-430,266,8,col)
    text(x+20,H-260,t,17,col,"Helvetica-Bold")
    wrap(x+20,H-292,d,12.5,SLATE,leading=18,maxw=226)
    x+=284
rect(80,90,1120,100,NAVY,r=14)
text(105,150,"The core fix the data points to:",13,HexColor("#cbd5e1"),"Helvetica-Bold")
wrap(105,124,"Equipment age is the #1 churn driver AND a lever Company A owns. A targeted proactive-upgrade program attacks the root cause — not the symptom.",14.5,WHITE,leading=20,maxw=1070)
footer()
c.showPage()

# ============================================================ SLIDE 13 — IMPACT
page_bg(); header("Quantified impact: $22.6M net benefit per million subscribers","THE BUSINESS CASE",13)
img(f"{FIG}/08_waterfall.png",80,300,560,300)
img(f"{FIG}/09_scaling.png",640,300,560,300)
# assumptions box
rect(80,92,1120,185,WHITE,r=14)
text(105,250,"Assumptions (conservative, stated explicitly)",13,NAVY,"Helvetica-Bold")
col1=["Operating churn:  2.0% / month (challenged carrier)",
      "ARPU:  $58.72 / mo (measured)   ·   Gross margin:  50%",
      "Contribution CLV:  ~$1,468 per customer"]
col2=["Target top 30% risk  ·  captures 42% of churners",
      "Save rate:  25% of contacted would-be churners",
      "Program cost:  $4 per contacted customer"]
yy=224
for a,b in zip(col1,col2):
    rect(105,yy-4,9,9,ORANGE); text(122,yy,a,12,INK)
    rect(640,yy-4,9,9,TEAL); text(657,yy,b,12,INK); yy-=30
rect(105,100,520,40,HexColor("#ecfdf5"),r=8)
text(120,114,"Result: churn 2.0% \u2192 1.79%/mo  ·  ROI 2.57x  ·  payback < 1 yr",12.5,GREEN,"Helvetica-Bold")
rect(660,100,510,40,HexColor("#eff6ff"),r=8)
text(675,114,"Scales linearly: a 5M-sub carrier \u2248 $113M net benefit / yr",12.5,BLUE,"Helvetica-Bold")
footer("Authors' model; CLV = margin \u00f7 monthly churn. Retention-economics benchmarks: Bain/Reichheld; HBR. Full derivation in notebook \u00a76.")
c.showPage()

# ============================================================ SLIDE 14 — ROADMAP
page_bg(); header("Roadmap, risks, and how we de-risk the rollout","NEXT STEPS",14)
text(80,H-150,"A staged rollout that proves value before full investment.",14.5,NAVY,"Helvetica-Bold")
phases=[("Phase 1 — Validate","Wks 1-6","Lock the data feed, deploy scoring, design the A/B retention test on a single high-risk segment."),
        ("Phase 2 — Pilot","Wks 7-16","Run treatment-vs-holdout upgrade offers; measure true incremental save rate and ROI."),
        ("Phase 3 — Scale","Q3+","Automate monthly scoring across the base; integrate offers into CRM; add real-time triggers.")]
x=80
for t,w,d in phases:
    rect(x,H-330,360,150,WHITE,r=14); rect(x,H-330,360,8,BLUE)
    text(x+20,H-212,t,15,NAVY,"Helvetica-Bold"); text(x+20,H-234,w,11.5,ORANGE,"Helvetica-Bold")
    wrap(x+20,H-260,d,12,SLATE,leading=17,maxw=320)
    x+=380
# risks
rect(80,120,1120,210,LIGHT,r=14)
text(105,290,"Risks & mitigations",14,NAVY,"Helvetica-Bold")
risks=[("Model captures correlation, not causation","Validate lift with a randomized holdout before scaling spend."),
       ("Sparse / undocumented profile fields","Rely on well-populated behavioral features; refresh data dictionary with client."),
       ("Offer cannibalization (discounts to loyal customers)","Target by risk \u00d7 value; suppress low-risk customers via the score."),
       ("Model drift over time","Re-train quarterly; monitor AUC and realized save rate.")]
yy=256
for a,b in risks:
    rect(105,yy-4,9,9,REDC); text(122,yy,a,12.5,INK,"Helvetica-Bold")
    text(560,yy,"\u2192  "+b,12,SLATE); yy-=40
footer()
c.showPage()

# ============================================================ SLIDE 15 — REFERENCES
page_bg(); header("References","SOURCES & CITATIONS",15)
refs=[
 "1.  IBISWorld (2025). Wireless Telecommunications Carriers in the US — Market Size & Industry Statistics.",
 "2.  Verizon Communications (2025). Q4 2024 / FY2024 Earnings Report — postpaid churn & ARPU disclosures. verizon.com/about/investors",
 "3.  T-Mobile US (2025). Q4 2024 Investor Factbook — postpaid phone churn. t-mobile.com/news",
 "4.  Reichheld, F. & Sasser, W.E. (1990). \u201cZero Defections: Quality Comes to Services.\u201d Harvard Business Review.",
 "5.  Bain & Company / Reichheld, F. — Prescription for Cutting Costs (customer retention & profitability).",
 "6.  Gallo, A. (2014). \u201cThe Value of Keeping the Right Customers.\u201d Harvard Business Review.",
 "7.  Neslin, S. et al. (2006). \u201cDefection Detection: Measuring and Understanding Predictive Accuracy of Customer Churn Models.\u201d Journal of Marketing Research.",
 "8.  Chen, T. & Guestrin, C. (2016). \u201cXGBoost: A Scalable Tree Boosting System.\u201d KDD '16.",
 "9.  Pedregosa, F. et al. (2011). \u201cScikit-learn: Machine Learning in Python.\u201d JMLR 12.",
 "10. Company A Dataset Overview & Final Assignment README (provided course materials), GCI World 2026.",
 "11. Dataset: Cell2Cell-style telecom churn data (Client.csv + Record.csv), provided by the course.",
]
y=H-150
for r in refs:
    y=wrap(80,y,r,12.5,INK,"Helvetica",leading=16,maxw=1120); y-=6
text(80,70,"Generative AI (ChatGPT/v0) was used to assist drafting and diagramming; chat URL provided in the Omnicampus references field. All analysis, code, and figures are the authors' own.",10.5,MUTE,"Helvetica-Oblique")
footer()
c.showPage()

c.save()
print("WROTE", OUT)
sz=os.path.getsize(OUT)/1024
print(f"Size: {sz:.0f} KB")
