"""
Builds a slide-style landscape PDF `outputs/Company_A_Churn_Deck.pdf` from the
same results.json + figures used by the PPTX deck, so the two never drift.

Run:  python build_pdf.py
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
RES = json.loads((ROOT / "outputs" / "results" / "results.json").read_text())
OUT = ROOT / "outputs" / "Company_A_Churn_Deck.pdf"

# 16:9 landscape page
PW, PH = 13.333 * inch, 7.5 * inch
PAGE = (PW, PH)

INK = HexColor("#141B2E")
NAVY = HexColor("#1F3A5F")
TEAL = HexColor("#128A86")
AMBER = HexColor("#C97A1E")
SLATE = HexColor("#5B6677")
CLOUD = HexColor("#F2F4F7")
WHITE = HexColor("#FFFFFF")
RED = HexColor("#B13A3A")
LIGHTTEAL = HexColor("#9FD8D6")
MUTED = HexColor("#B9C4D4")

c = canvas.Canvas(str(OUT), pagesize=PAGE)


def fnum(x):
    return f"${x:,.0f}"


def wrap(text, font, size, max_w):
    """Greedy word wrap -> list of lines."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if stringWidth(trial, font, size) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def para(x, y, text, font, size, color, max_w, leading=None, align="left"):
    """Draw wrapped paragraph, return y after."""
    leading = leading or size * 1.3
    c.setFont(font, size)
    c.setFillColor(color)
    for ln in wrap(text, font, size, max_w):
        if align == "right":
            c.drawRightString(x + max_w, y, ln)
        elif align == "center":
            c.drawCentredString(x + max_w / 2, y, ln)
        else:
            c.drawString(x, y, ln)
        y -= leading
    return y


def header(kicker, title):
    c.setFillColor(TEAL)
    c.rect(0, PH - 0.18 * inch, PW, 0.18 * inch, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(TEAL)
    c.drawString(0.6 * inch, PH - 0.62 * inch, kicker.upper())
    c.setFillColor(INK)
    # title may wrap
    y = PH - 0.95 * inch
    for ln in wrap(title, "Helvetica-Bold", 24, PW - 1.2 * inch):
        c.drawString(0.6 * inch, y, ln)
        y -= 0.34 * inch
    c.setFillColor(AMBER)
    c.rect(0.6 * inch, y + 0.08 * inch, 1.1 * inch, 0.045 * inch, fill=1, stroke=0)
    return y


def img(name, x, y, w):
    """Place image with top-left at (x, y_top). y is the TOP edge."""
    p = FIG / name
    ir = ImageReader(str(p))
    iw, ih = ir.getSize()
    h = w * ih / iw
    c.drawImage(ir, x, y - h, width=w, height=h, mask="auto")
    return h


def panel(x, y_top, w, h, fill=CLOUD, bar=None):
    c.setFillColor(fill)
    c.rect(x, y_top - h, w, h, fill=1, stroke=0)
    if bar:
        c.setFillColor(bar)
        c.rect(x, y_top - 0.12 * inch, w, 0.12 * inch, fill=1, stroke=0)


d = RES
base = next(x for x in d["scenarios"] if x["name"] == "Base")

# ============================================================ 1. TITLE
c.setFillColor(NAVY)
c.rect(0, 0, PW, PH, fill=1, stroke=0)
c.setFillColor(TEAL)
c.rect(0, 2.5 * inch, PW, 0.07 * inch, fill=1, stroke=0)
c.setFillColor(WHITE)
c.setFont("Helvetica-Bold", 38)
c.drawString(0.9 * inch, PH - 2.1 * inch, "Reducing Customer Churn at Company A")
c.setFont("Helvetica", 19)
c.setFillColor(CLOUD)
c.drawString(0.9 * inch, PH - 2.9 * inch,
             "A profit-driven, calibrated, and fair churn model.")
c.setFillColor(LIGHTTEAL)
c.setFont("Helvetica-Bold", 13)
c.drawString(0.9 * inch, 1.9 * inch, "Telecom Retention Analytics")
c.setFillColor(MUTED)
c.setFont("Helvetica", 12)
c.drawString(0.9 * inch, 1.55 * inch,
             "100,000 customers   |   123 engineered behavioral features   |   "
             "demographics excluded by design")
c.showPage()

# ============================================================ 2. EXEC SUMMARY
header("Executive summary", "The bottom line")
panels = [
    ("Realistic model", f"AUC {d['calibration']['auc_calibrated']:.2f}",
     "Tuned + calibrated XGBoost. No leakage, no 0.99 fantasy.", TEAL),
    ("Targeting lift", f"{d['capture_at_20pct']['lift_evar_vs_random']:.2f}x",
     "vs. random outreach, ranking by expected value at risk.", NAVY),
    ("Base-case ROI", f"{base['roi']:.1f}x",
     f"Net {fnum(base['net_profit_population'])} at the profit-optimal "
     f"{int(d['profit_curve_base']['optimal_frac']*100)}% contact depth.", AMBER),
]
x = 0.6 * inch
for (k, big, sub, col) in panels:
    panel(x, PH - 2.1 * inch, 3.9 * inch, 2.6 * inch, CLOUD, col)
    c.setFillColor(col)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 0.3 * inch, PH - 2.55 * inch, k.upper())
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 36)
    c.drawString(x + 0.3 * inch, PH - 3.2 * inch, big)
    para(x + 0.3 * inch, PH - 3.7 * inch, sub, "Helvetica", 12.5, SLATE,
         3.3 * inch, leading=16)
    x += 4.25 * inch
y = PH - 5.1 * inch
c.setFillColor(INK)
c.setFont("Helvetica-Bold", 13.5)
c.drawString(0.6 * inch, y, "What changed from a naive first pass:")
para(0.6 * inch, y - 0.3 * inch,
     "Protected attributes removed, probabilities calibrated, scores re-weighted to "
     "the true ~2% churn rate, the decision threshold set on a profit curve, drivers "
     "explained with SHAP, and the business case shown as bear / base / bull with "
     "cited assumptions.", "Helvetica", 13.5, SLATE, PW - 1.2 * inch, leading=19)
c.showPage()

# ============================================================ 3. PROBLEM & DATA
ytop = header("Problem framing",
              "Churn is expensive - but not every customer is worth the same call")
y = PH - 2.1 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(0.6 * inch, y, "The task.")
y = para(0.6 * inch, y - 0.28 * inch,
         "Predict which subscribers will churn, explain why, and turn that into a "
         "retention plan that makes money.", "Helvetica", 13, SLATE, 6.0 * inch,
         leading=17)
y -= 0.15 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(0.6 * inch, y, "The data.")
y = para(0.6 * inch, y - 0.28 * inch,
         f"{d['data']['n_rows']:,} customers, {d['data']['n_raw_cols']} raw columns "
         "across account, usage, and demographic tables.", "Helvetica", 13, SLATE,
         6.0 * inch, leading=17)
y -= 0.15 * inch
c.setFillColor(AMBER)
c.setFont("Helvetica-Bold", 15)
c.drawString(0.6 * inch, y, "The honesty problem.")
para(0.6 * inch, y - 0.28 * inch,
     "The sample is balanced to ~50% churn for training, but real monthly churn is "
     "~2%. We train on the sample, then re-weight predictions back to reality so the "
     "dollars are credible.", "Helvetica", 13, SLATE, 6.0 * inch, leading=17)

panel(7.0 * inch, PH - 2.0 * inch, 5.7 * inch, 4.3 * inch, CLOUD)
c.setFillColor(TEAL)
c.setFont("Helvetica-Bold", 12)
c.drawString(7.3 * inch, PH - 2.45 * inch, "KEY NUMBERS")
kv = [
    ("ARPU (avg monthly revenue)", f"${d['descriptive']['arpu']:.0f}"),
    ("Median tenure", f"{d['descriptive']['median_tenure_months']:.0f} months"),
    ("Model features (engineered)", f"{d['data']['n_model_features']}"),
    ("Protected attributes dropped", f"{len(d['data']['dropped_protected'])}"),
    ("Assumed true monthly churn", f"{d['base_rate']['true_rate']*100:.0f}%"),
]
yy = PH - 2.95 * inch
for (k, v) in kv:
    c.setFillColor(INK)
    c.setFont("Helvetica", 13)
    c.drawString(7.3 * inch, yy, k)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 14)
    c.drawRightString(12.4 * inch, yy, v)
    yy -= 0.66 * inch
c.showPage()

# ============================================================ 4. FAIRNESS
header("Responsible modeling", "We deliberately excluded demographics")
para(0.6 * inch, PH - 2.0 * inch,
     "Targeting retention by ethnicity, income, marital status, or dwelling is a legal "
     "and ethical hazard - and it isn't actionable. All 20 such attributes are dropped "
     "before the model sees the data, with an automated guard that fails the build if "
     "one leaks through encoding.", "Helvetica", 14, SLATE, PW - 1.2 * inch, leading=19)
prot = d["data"]["dropped_protected"]
cols = [prot[i::4] for i in range(4)]
x = 0.6 * inch
for col in cols:
    panel(x, PH - 3.5 * inch, 2.95 * inch, 3.0 * inch, CLOUD)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + 0.25 * inch, PH - 3.85 * inch, "EXCLUDED")
    yy = PH - 4.2 * inch
    c.setFont("Helvetica", 12.5)
    c.setFillColor(SLATE)
    for a in col:
        c.drawString(x + 0.25 * inch, yy, "- " + a)
        yy -= 0.32 * inch
    x += 3.05 * inch
c.showPage()

# ============================================================ 5. METHOD PIPELINE
header("Approach", "One reproducible pipeline, end to end")
steps = [
    ("1  Engineer", "Merge usage + account tables; build behavioral features "
     "(equipment age, usage trend, care intensity, overage share)."),
    ("2  Model", "Compare Logistic Regression, Random Forest, XGBoost; tune XGBoost "
     "with randomized search + 5-fold CV."),
    ("3  Calibrate", "Isotonic calibration so scores are real probabilities; re-weight "
     "to the true 2% base rate."),
    ("4  Decide", "Rank by expected value at risk; choose contact depth on the profit "
     "curve, not an arbitrary 0.5 cutoff."),
    ("5  Explain & act", "SHAP drivers, K-means personas, and an uplift/persuadables "
     "view feed a bear/base/bull business case."),
]
yy = PH - 2.2 * inch
for (t, b) in steps:
    c.setFillColor(NAVY)
    c.rect(0.6 * inch, yy - 0.62 * inch, 2.5 * inch, 0.72 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(0.75 * inch, yy - 0.4 * inch, t)
    para(3.35 * inch, yy - 0.2 * inch, b, "Helvetica", 13, SLATE, 9.3 * inch,
         leading=15)
    yy -= 0.95 * inch
c.showPage()

# ============================================================ 6. MODEL COMPARISON
header("Model selection", "XGBoost wins - at a believable accuracy")
img("model_comparison.png", 0.6 * inch, PH - 1.95 * inch, 7.4 * inch)
xg = d["model_comparison"]["XGBoost"]
y = PH - 2.2 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(8.3 * inch, y, "Why this is credible")
y = para(8.3 * inch, y - 0.32 * inch,
         f"AUC {xg['auc']:.2f}, PR-AUC {xg['pr_auc']:.2f}. An AUC near 0.70 is normal "
         "for telecom churn on behavioral data - a 0.95+ score would signal target "
         "leakage.", "Helvetica", 13, SLATE, 4.4 * inch, leading=17)
y -= 0.1 * inch
y = para(8.3 * inch, y,
         f"5-fold CV: {d['cv']['mean_auc']:.3f} +/- {d['cv']['std_auc']:.3f}. Tiny "
         "variance across folds means the model is stable, not lucky.",
         "Helvetica", 13, SLATE, 4.4 * inch, leading=17)
y -= 0.1 * inch
para(8.3 * inch, y,
     "Brier score improves after calibration - the probabilities can be trusted for "
     "dollar math.", "Helvetica", 13, SLATE, 4.4 * inch, leading=17)
c.showPage()

# ============================================================ 7. CALIBRATION
header("Trustworthy probabilities", "Calibration + the base-rate correction")
img("calibration.png", 0.6 * inch, PH - 1.95 * inch, 7.0 * inch)
br = d["base_rate"]
y = PH - 2.2 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(7.9 * inch, y, "A score is not a probability")
y = para(7.9 * inch, y - 0.32 * inch,
         "Raw model scores cluster near 0.5 because the training sample is balanced. "
         "Isotonic calibration aligns predicted probability with observed frequency "
         "(the diagonal).", "Helvetica", 13, SLATE, 4.8 * inch, leading=17)
y -= 0.1 * inch
c.setFillColor(AMBER)
c.setFont("Helvetica-Bold", 14)
c.drawString(7.9 * inch, y, "Then we correct the prior:")
y = para(7.9 * inch, y - 0.28 * inch,
         f"mean P(churn) {br['mean_p_sample']:.2f} in the sample -> "
         f"{br['mean_p_true']:.3f} re-weighted to the real ~2% rate.",
         "Helvetica", 13, INK, 4.8 * inch, leading=17)
y -= 0.1 * inch
para(7.9 * inch, y,
     "Without this step, every revenue-at-risk and ROI figure would be inflated by "
     "~20x.", "Helvetica", 13, SLATE, 4.8 * inch, leading=17)
c.showPage()

# ============================================================ 8. SHAP
header("Why customers leave", "SHAP - the model's honest drivers")
img("shap_importance.png", 0.6 * inch, PH - 1.95 * inch, 7.4 * inch)
y = PH - 2.2 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(8.3 * inch, y, "All actionable, none demographic")
y -= 0.34 * inch
for head, body in [
    ("Equipment age (eqpdays)", " - the top driver. Old handsets churn; proactive "
     "upgrade offers are the lever."),
    ("Tenure & usage trend", " - declining minutes-of-use flags disengagement early."),
    ("Care intensity & dropped calls", " - service quality and support friction. "
     "Fixable operationally."),
]:
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(8.3 * inch, y, head)
    y = para(8.3 * inch, y - 0.22 * inch, body, "Helvetica", 13, SLATE, 4.4 * inch,
             leading=16)
    y -= 0.08 * inch
para(8.3 * inch, y,
     "Because demographics were excluded, every top driver maps to an action the "
     "business can actually take.", "Helvetica", 13, INK, 4.4 * inch, leading=16)
c.showPage()

# ============================================================ 9. HANDSET HONESTY
header("Honest segmentation", "Churn rises with equipment age")
img("handset_churn.png", 0.6 * inch, PH - 1.95 * inch, 7.4 * inch)
y = PH - 2.2 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(8.3 * inch, y, "Binned, not over-precise")
y = para(8.3 * inch, y - 0.32 * inch,
         "Equipment age is grouped into interpretable buckets rather than implying "
         "false day-level precision.", "Helvetica", 13, SLATE, 4.4 * inch, leading=17)
y -= 0.1 * inch
y = para(8.3 * inch, y,
         "The pattern is monotonic and clean: the longer a customer keeps an aging "
         "device, the more likely they leave.", "Helvetica", 13, INK, 4.4 * inch,
         leading=17)
y -= 0.1 * inch
c.setFillColor(AMBER)
c.setFont("Helvetica-Bold", 13)
c.drawString(8.3 * inch, y, "Action:")
para(8.3 * inch + 0.7 * inch, y,
     "trigger an upgrade offer as devices cross the high-risk age band.",
     "Helvetica", 13, SLATE, 3.7 * inch, leading=16)
c.showPage()

# ============================================================ 10. CAPTURE
header("Prioritize", "Call the highest-value risk first")
img("capture_curve.png", 0.6 * inch, PH - 1.95 * inch, 7.4 * inch)
cap = d["capture_at_20pct"]
y = PH - 2.2 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(8.3 * inch, y, "Expected value at risk")
y = para(8.3 * inch, y - 0.32 * inch,
         "We rank by P(churn) x margin x lifetime value, not probability alone - a "
         "$100 customer at 30% risk beats a $20 customer at 60%.",
         "Helvetica", 13, SLATE, 4.4 * inch, leading=17)
y -= 0.12 * inch
para(8.3 * inch, y,
     f"Contacting the top 20% by value captures {cap['by_evar']*100:.0f}% of churn "
     f"value - {cap['lift_evar_vs_random']:.2f}x better than random.",
     "Helvetica-Bold", 13, TEAL, 4.4 * inch, leading=17)
c.showPage()

# ============================================================ 11. PROFIT CURVE
header("Optimize", "How deep to go - the profit curve")
img("profit_curve.png", 0.6 * inch, PH - 1.95 * inch, 7.4 * inch)
pc = d["profit_curve_base"]
y = PH - 2.2 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(8.3 * inch, y, "Maximize profit, not recall")
y = para(8.3 * inch, y - 0.32 * inch,
         "Calling everyone wastes incentives; calling too few leaves value on the "
         "table. We sweep contact depth and pick the peak.",
         "Helvetica", 13, SLATE, 4.4 * inch, leading=17)
y -= 0.12 * inch
c.setFillColor(AMBER)
c.setFont("Helvetica-Bold", 15)
c.drawString(8.3 * inch, y, f"Optimal depth: {int(pc['optimal_frac']*100)}%")
para(8.3 * inch, y - 0.3 * inch,
     f"Net profit {fnum(pc['optimal_net_profit_population'])}  -  "
     f"ROI {pc['optimal_roi']:.1f}x  -  ~{pc['customers_saved_population']:.0f} "
     "customers saved (base case).", "Helvetica", 13, INK, 4.4 * inch, leading=17)
c.showPage()

# ============================================================ 12. PERSONAS
header("Who they are", "Four behavioral personas (K-means)")
img("personas.png", 0.6 * inch, PH - 1.95 * inch, 7.2 * inch)
cl = sorted(d["personas"]["clusters"], key=lambda c: -c["churn_rate"])[0]
y = PH - 2.2 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(8.1 * inch, y, "Behavioral, not demographic")
y = para(8.1 * inch, y - 0.32 * inch,
         "Clusters are built only from behavior - usage trend, equipment age, care "
         "intensity, overage, tenure.", "Helvetica", 13, SLATE, 4.6 * inch, leading=17)
y -= 0.12 * inch
c.setFillColor(RED)
c.setFont("Helvetica-Bold", 13)
c.drawString(8.1 * inch, y, "High-risk segment")
y = para(8.1 * inch, y - 0.24 * inch,
         f"churn rate {cl['churn_rate']*100:.0f}%, high care-call intensity and aging "
         "equipment - the clearest save target.", "Helvetica", 13, SLATE, 4.6 * inch,
         leading=16)
y -= 0.1 * inch
para(8.1 * inch, y,
     "A separate high-ARPU segment justifies premium, concierge-style retention "
     "offers.", "Helvetica", 13, INK, 4.6 * inch, leading=16)
c.showPage()

# ============================================================ 13. UPLIFT
header("Persuadables", "Treat customers a campaign can actually move")
img("uplift.png", 0.6 * inch, PH - 1.95 * inch, 7.2 * inch)
y = PH - 2.15 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 15)
c.drawString(8.1 * inch, y, "Beyond risk: responsiveness")
y = para(8.1 * inch, y - 0.32 * inch,
         "Some high-risk customers leave regardless; the money is in persuadables - "
         "those whose retention an offer actually changes (T-learner uplift).",
         "Helvetica", 13, SLATE, 4.6 * inch, leading=16)
y -= 0.1 * inch
c.setFillColor(AMBER)
c.setFont("Helvetica-Bold", 12.5)
c.drawString(8.1 * inch, y, "Honesty note")
y = para(8.1 * inch, y - 0.24 * inch,
         "This data is observational (no test/control), so the response here is a "
         "clearly-labeled simulation to demonstrate the method.",
         "Helvetica", 12.5, SLATE, 4.6 * inch, leading=15)
y -= 0.1 * inch
para(8.1 * inch, y,
     "Next step: a live A/B holdout to measure real uplift before scaling.",
     "Helvetica", 12.5, INK, 4.6 * inch, leading=15)
c.showPage()

# ============================================================ 14. BUSINESS CASE
header("The money", "Bear / base / bull - profit-positive in the base & bull")
img("scenarios.png", 0.6 * inch, PH - 1.95 * inch, 6.8 * inch)
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 14)
c.drawString(7.7 * inch, PH - 2.1 * inch, "Assumptions are cited, not invented")
para(7.7 * inch, PH - 2.4 * inch,
     "Each scenario varies save rate, offer cost, margin, and CLV horizon - all "
     "sourced in REFERENCES.md.", "Helvetica", 12.5, SLATE, 5.0 * inch, leading=15)
yy = PH - 3.1 * inch
for sc in d["scenarios"]:
    col = {"Bear": RED, "Base": NAVY, "Bull": TEAL}[sc["name"]]
    np_pop = sc["net_profit_population"]
    panel(7.7 * inch, yy, 5.0 * inch, 1.05 * inch, CLOUD)
    c.setFillColor(col)
    c.rect(7.7 * inch, yy - 1.05 * inch, 0.12 * inch, 1.05 * inch, fill=1, stroke=0)
    c.setFillColor(col)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(7.95 * inch, yy - 0.42 * inch, sc["name"])
    c.setFillColor(SLATE)
    c.setFont("Helvetica", 10.5)
    c.drawString(7.95 * inch, yy - 0.72 * inch,
                 f"save {int(sc['save_rate']*100)}% - ${sc['offer_cost']:.0f} offer")
    c.setFillColor(RED if np_pop < 0 else INK)
    c.setFont("Helvetica-Bold", 18)
    c.drawRightString(12.6 * inch, yy - 0.42 * inch, fnum(np_pop))
    c.setFillColor(SLATE)
    c.setFont("Helvetica", 10.5)
    c.drawRightString(12.6 * inch, yy - 0.72 * inch,
                      f"ROI {sc['roi']:.1f}x - ~{sc['customers_saved_population']:.0f} saved")
    yy -= 1.15 * inch
c.showPage()

# ============================================================ 15. RECOMMENDATIONS
c.setFillColor(NAVY)
c.rect(0, 0, PW, PH, fill=1, stroke=0)
c.setFillColor(TEAL)
c.rect(0, PH - 0.18 * inch, PW, 0.18 * inch, fill=1, stroke=0)
c.setFillColor(LIGHTTEAL)
c.setFont("Helvetica-Bold", 13)
c.drawString(0.7 * inch, PH - 0.7 * inch, "RECOMMENDATIONS & NEXT STEPS")
c.setFillColor(WHITE)
c.setFont("Helvetica-Bold", 25)
c.drawString(0.7 * inch, PH - 1.15 * inch,
             "Operationalize the model - and prove it with a test")
recs = [
    ("Deploy value-ranked scoring", "Score the base monthly; route the profit-optimal "
     f"top {int(d['profit_curve_base']['optimal_frac']*100)}% by value-at-risk to retention."),
    ("Trigger device-age offers", "Automate upgrade offers as handsets cross the "
     "high-risk age band - the #1 SHAP driver."),
    ("Fix service friction", "Feed dropped-call and care-intensity signals to network "
     "and support ops as leading indicators."),
    ("Run an A/B retention test", "Randomized treatment/control to measure REAL uplift "
     "and replace the simulated persuadables figures."),
    ("Re-validate assumptions", "Confirm save rate, offer cost, margin, and CLV with "
     "finance; refresh the bear/base/bull case quarterly."),
]
yy = PH - 2.05 * inch
for i, (t, b) in enumerate(recs, 1):
    c.setFillColor(TEAL)
    c.rect(0.7 * inch, yy - 0.5 * inch, 0.5 * inch, 0.5 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(0.95 * inch, yy - 0.37 * inch, str(i))
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1.45 * inch, yy - 0.18 * inch, t)
    c.setFillColor(MUTED)
    para(1.45 * inch, yy - 0.45 * inch, b, "Helvetica", 12.5, MUTED, 10.8 * inch,
         leading=15)
    yy -= 0.92 * inch
c.showPage()

c.save()
print(f"Saved {OUT}")
