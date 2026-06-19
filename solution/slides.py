"""Build the executive slide deck (16:9 PDF, <=15 slides) for the Company A churn PoC.

Renders each slide as a vector matplotlib page (crisp text, small file) and embeds the
pre-generated figures from figures/. Output: Hazem_Douzi.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.image as mpimg
import textwrap
import os

FIG = "/home/ubuntu/fp2/solution/figures"
OUT = "/home/ubuntu/fp2/solution/Hazem_Douzi.pdf"

NAVY   = "#1f3b73"
NAVY2  = "#2a4d8f"
RED    = "#d1495b"
TEAL   = "#15919b"
AMBER  = "#e9a000"
GREY   = "#5b6470"
LIGHT  = "#eef1f6"
INK    = "#1b2330"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.parse_math": False})

W, H = 13.333, 7.5  # 16:9


def new_slide(pdf, footer_idx):
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # footer
    ax.add_patch(Rectangle((0, 0), 1, 0.035, color=LIGHT, zorder=0))
    ax.text(0.012, 0.014, "Company A — Customer-Retention Proof of Concept", fontsize=8.5,
            color=GREY, va="center")
    ax.text(0.988, 0.014, f"{footer_idx}", fontsize=8.5, color=GREY, va="center", ha="right")
    return fig, ax


def header(ax, kicker, title, color=NAVY):
    ax.add_patch(Rectangle((0, 0.88), 1, 0.12, color=color, zorder=1))
    ax.add_patch(Rectangle((0, 0.875), 1, 0.006, color=AMBER, zorder=2))
    ax.text(0.035, 0.955, kicker.upper(), fontsize=11, color="#cdd8ee", va="center",
            fontweight="bold", zorder=3)
    ax.text(0.035, 0.915, title, fontsize=20, color="white", va="center",
            fontweight="bold", zorder=3)


def bullets(ax, items, x=0.05, y0=0.78, dy=0.075, size=13.5, w=0.9, color=INK):
    # Manual wrapping (avoids a matplotlib PDF-backend bug where wrap=True forces
    # math parsing on '$'). Advances vertically by the number of wrapped lines.
    cpl = max(18, int(w * 960 / (0.62 * size)))   # approx chars/line for the column width
    line_h = size / 540.0 * 1.5                    # line height in axes fraction
    gap = max(0.012, dy - 2 * line_h)              # inter-bullet gap derived from requested dy
    y = y0
    for it in items:
        txt = it[1] if isinstance(it, tuple) else it
        wrapped = textwrap.fill(txt, cpl)
        nlines = wrapped.count("\n") + 1
        ax.text(x, y, "▸", fontsize=size, color=TEAL, va="top", fontweight="bold")
        ax.text(x + 0.022, y, wrapped, fontsize=size, color=color, va="top",
                linespacing=1.5)
        y -= nlines * line_h + gap
    return y


def add_img(fig, path, rect):
    a = fig.add_axes(rect); a.axis("off")
    if os.path.exists(path):
        a.imshow(mpimg.imread(path))
    return a


def metric_card(ax, x, y, w, h, value, label, color=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=LIGHT, ec=color, lw=1.6, zorder=2))
    ax.text(x + w/2, y + h*0.60, value, fontsize=19, color=color, ha="center", va="center",
            fontweight="bold", zorder=3)
    ax.text(x + w/2, y + h*0.22, label, fontsize=9.5, color=GREY, ha="center", va="center", zorder=3)


pdf = PdfPages(OUT)

# ---------------- Slide 1 : Title ----------------
fig, ax = new_slide(pdf, 1)
ax.add_patch(Rectangle((0, 0), 1, 1, color=NAVY, zorder=0))
ax.add_patch(Rectangle((0, 0.46), 1, 0.012, color=AMBER, zorder=1))
ax.text(0.06, 0.70, "Turning Churn Data into Protected Revenue", fontsize=33, color="white",
        fontweight="bold", va="center")
ax.text(0.06, 0.60, "A data-driven retention engine for a US wireless carrier", fontsize=18,
        color="#cdd8ee", va="center")
ax.text(0.06, 0.36, "Proof of Concept  •  Market analysis · EDA · Machine learning · Business case",
        fontsize=13.5, color="white", va="center")
ax.text(0.06, 0.30, "Recommendation: a value-based ('Expected Value at Risk') targeted-retention program",
        fontsize=13.5, color="#9fb4e0", va="center")
ax.text(0.06, 0.14, "Hazem Douzi", fontsize=15, color="white", fontweight="bold", va="center")
ax.text(0.06, 0.095, "Final Assignment — June 2026", fontsize=12, color="#cdd8ee", va="center")
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 2 : Executive summary ----------------
fig, ax = new_slide(pdf, 2)
header(ax, "Executive summary", "The problem, the idea, and the payoff on one page")
bullets(ax, [
    "PROBLEM — In a saturated US wireless market (~2% monthly churn, ~22%/yr), the client loses high-value customers faster than it can profitably replace them; winning back a lost customer costs 5–10× more than keeping one.",
    "INSIGHT — Churn here is not random and not demographic. It is an equipment & engagement problem: aging handsets, no modern (web-capable) devices, low/declining usage, and overage bill-shock are the dominant, fixable drivers.",
    "SOLUTION — Score every customer monthly, then rank by Expected Value at Risk (EVaR = P(churn) × revenue) and route the top names into four persona-matched offers (device upgrade, right-plan, loyalty, re-engage).",
], y0=0.80, dy=0.10, size=14)
# metric strip
y = 0.18
metric_card(ax, 0.05, y, 0.205, 0.16, "$1.64M", "net benefit / yr\n(per 100k customers)", NAVY)
metric_card(ax, 0.28, y, 0.205, 0.16, "164%", "return on the\nretention budget", TEAL)
metric_card(ax, 0.51, y, 0.205, 0.16, "45%", "of revenue-at-risk\nin just the top 20%", AMBER)
metric_card(ax, 0.74, y, 0.205, 0.16, "$32.8M", "net / yr scaled to a\n2M-subscriber base", RED)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 3 : Market context ----------------
fig, ax = new_slide(pdf, 3)
header(ax, "Market analysis", "A saturated market where retention beats acquisition")
bullets(ax, [
    "The US wireless services market is ~$215–321B with 632M connections, but growing only ~2.4%/yr — a mature, share-of-wallet fight, not a land grab. [1][2]",
    "Carriers run ~0.86–0.92% postpaid churn per month (~22%/yr of the base must be re-won every year). [3][4]",
    "Acquiring a customer costs 5–10× more than retaining one; lifting retention 5% can raise profit 25–95%. [5][6]",
    "Each gross add costs ~$350 (CPGA); every avoided churn protects 12+ months of revenue and dodges that cost. [7]",
], y0=0.78, dy=0.105, size=14, w=0.55)
# right column metric cards
metric_card(ax, 0.62, 0.60, 0.33, 0.14, "$215–321B", "US wireless market size [1][2]", NAVY)
metric_card(ax, 0.62, 0.43, 0.33, 0.14, "~22% / yr", "of customers churn [3][4]", RED)
metric_card(ax, 0.62, 0.26, 0.33, 0.14, "5–10×", "costlier to acquire vs retain [5]", TEAL)
metric_card(ax, 0.62, 0.09, 0.33, 0.14, "+25–95%", "profit from +5% retention [6]", AMBER)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 4 : Data & quality ----------------
fig, ax = new_slide(pdf, 4)
header(ax, "Exploratory data analysis (1/3)", "100k customers, 100 fields — but trust behavior, not demographics")
bullets(ax, [
    "Merged Client.csv + Record.csv on Customer_ID → 100,000 customers × 100 features (usage, billing, equipment, call-quality, demographics + churn label).",
    "Target is balanced (~49.6% churn) — the sample is oversampled for modeling; we re-base to the real ~2%/mo rate in the business case.",
    "Third-party demographic fields are 25–49% missing (income, dwelling, #cars). Behavioral/billing fields are ≥99% complete.",
    ("•", "So the model — and every intervention — is built on reliable behavioral signals, not sparse demographics."),
], y0=0.80, dy=0.095, size=13, w=0.46)
add_img(fig, f"{FIG}/02_missingness.png", [0.52, 0.06, 0.46, 0.78])
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 5 : What moves churn ----------------
fig, ax = new_slide(pdf, 5)
header(ax, "Exploratory data analysis (2/3)", "What actually moves churn: equipment age leads")
add_img(fig, f"{FIG}/03_churn_correlations.png", [0.02, 0.07, 0.49, 0.77])
add_img(fig, f"{FIG}/04_eqpdays_churn.png", [0.52, 0.30, 0.46, 0.54])
bullets(ax, [
    "Equipment age (eqpdays) is the #1 driver: churn jumps from ~37–42% on newer phones to ~56–60% on older ones.",
    "Higher handset price & higher usage (mou) pull churn DOWN — engaged, well-equipped customers stay.",
], x=0.53, y0=0.26, dy=0.085, size=12.5, w=0.45)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 6 : Categorical levers ----------------
fig, ax = new_slide(pdf, 6)
header(ax, "Exploratory data analysis (3/3)", "Three interventionable levers stand out")
add_img(fig, f"{FIG}/05_categorical_levers.png", [0.04, 0.30, 0.92, 0.52])
bullets(ax, [
    ("•", "Spend-limit accounts (asl_flag=Y) churn ~10 pts LESS — a commitment signal we can nurture."),
    ("•", "Refurbished handsets churn MORE than new — device quality matters."),
    ("•", "Phones without modern web capability churn MOST — a direct upgrade trigger."),
], x=0.06, y0=0.25, dy=0.07, size=13)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 7 : Problem definition ----------------
fig, ax = new_slide(pdf, 7)
header(ax, "Problem definition", "From a vague 'reduce churn' goal to a precise, profitable decision")
bullets(ax, [
    "Business objective — reduce voluntary churn and the revenue it destroys, profitably (not churn for its own sake).",
    "ML task — binary classification: predict P(churn) in the 30–60 day window from the labeled churn outcome.",
    "Our twist — do NOT stop at probability. Rank by Expected Value at Risk: EVaR = P(churn) × customer revenue, so the budget protects the most REVENUE, not merely the most accounts.",
    "Success metrics — ROC-AUC (ranking quality) + lift / capture in the top decile-quintile + revenue-at-risk captured — the numbers that map straight to a campaign.",
    "Why this problem — its drivers (equipment, plan-fit, engagement) are observable AND fixable, so a prediction becomes an action.",
], y0=0.79, dy=0.13, size=14)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 8 : Models ----------------
fig, ax = new_slide(pdf, 8)
header(ax, "Machine learning — experiments", "Gradient boosting wins; the model agrees with the EDA")
add_img(fig, f"{FIG}/06_auc_comparison.png", [0.02, 0.40, 0.45, 0.44])
add_img(fig, f"{FIG}/08_feature_importance.png", [0.50, 0.40, 0.48, 0.44])
bullets(ax, [
    "Model & score — we compared 4 models on a held-out 30% test set. Selected model: XGBoost, evaluation metric ROC-AUC = 0.698 (5-fold CV 0.700 ± 0.003); accuracy 0.64, lift@20% 1.49. (Logistic 0.625, Random Forest 0.683, LightGBM 0.697.)",
    "Why this model & how built — gradient boosting captures non-linear interactions and handles mixed numeric/categorical fields & missing values without heavy preprocessing; built with median imputation + 5 engineered ratios, 600 trees (depth 5, learning-rate 0.03), validated by stratified 5-fold CV.",
    "Insight — top drivers (eqpdays, refurb_new, months, hnd_price, mou) mirror the EDA, so the model is accurate AND explainable; AUC ≈0.70 is the realistic ceiling for this classic dataset. [8]",
], x=0.05, y0=0.36, dy=0.062, size=12, w=0.9)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 9 : EVaR ----------------
fig, ax = new_slide(pdf, 9)
header(ax, "The differentiator", "Rank by revenue-at-risk, not just probability", color=TEAL)
add_img(fig, f"{FIG}/09b_revenue_capture.png", [0.50, 0.06, 0.48, 0.78])
bullets(ax, [
    "A churn-probability list over-weights cheap, low-value accounts.",
    "Multiplying probability × revenue (EVaR) re-prioritises toward customers whose loss actually hurts.",
    "Same campaign size (top 20%), far more value protected:",
], x=0.05, y0=0.78, dy=0.085, size=14, w=0.43)
metric_card(ax, 0.05, 0.40, 0.20, 0.14, "45%", "revenue-at-risk\ncaptured (EVaR)", TEAL)
metric_card(ax, 0.27, 0.40, 0.20, 0.14, "28%", "captured by naive\nchurn ranking", GREY)
ax.text(0.05, 0.30, "→ EVaR protects 1.6× more revenue than churn-ranking and 2.2× more than random,",
        fontsize=13, color=INK, va="top", fontweight="bold")
ax.text(0.05, 0.25, "    for the exact same number of customers contacted.", fontsize=13, color=INK, va="top",
        fontweight="bold")
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 10 : Personas ----------------
fig, ax = new_slide(pdf, 10)
header(ax, "From scores to action", "Four at-risk personas, each with a matched offer")
add_img(fig, f"{FIG}/10_personas.png", [0.03, 0.40, 0.94, 0.44])
bullets(ax, [
    ("•", "Aging-Handset (~67%) → proactive financed DEVICE UPGRADE — the single highest-ROI lever."),
    ("•", "Bill-Shock / Overage (~13%, ARPU $99) → RIGHT-PLAN move to unlimited + proactive overage alerts."),
    ("•", "Stable-Loyalty (~12%) → LOYALTY credit / contract-renewal nudge.   •  Disengaging (~8%) → RE-ENGAGE bundle / win-back."),
], x=0.05, y0=0.34, dy=0.075, size=12.5)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 11 : Business case ----------------
fig, ax = new_slide(pdf, 11)
header(ax, "Business case", "What it earns: ~$1.64M net per 100k customers / year")
add_img(fig, f"{FIG}/11_roi_waterfall.png", [0.48, 0.06, 0.50, 0.78])
bullets(ax, [
    "Per 100k customers, ~$15.5M of revenue is at risk each year (re-based to ~22%/yr real churn).",
    "Target the top 20% by EVaR → reach ~45% of that revenue; a 30% save rate protects ~$2.09M.",
    "Add ~$0.55M avoided reacquisition cost; subtract ~$1.0M campaign cost ($50 × 20k offers).",
], x=0.05, y0=0.78, dy=0.10, size=13.5, w=0.42)
metric_card(ax, 0.05, 0.16, 0.195, 0.16, "$1.64M", "net benefit / yr", NAVY)
metric_card(ax, 0.255, 0.16, 0.195, 0.16, "164%", "ROI on spend", TEAL)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 12 : Sensitivity & scale ----------------
fig, ax = new_slide(pdf, 12)
header(ax, "Is it robust?", "Profitable across every assumption — and it scales")
add_img(fig, f"{FIG}/12_sensitivity.png", [0.03, 0.08, 0.52, 0.76])
bullets(ax, [
    "Net benefit stays POSITIVE across every save-rate (20–40%) × offer-cost ($30–75) scenario tested.",
    "Even pessimistic (20% save, $75 offer) the program nets +$0.3M per 100k customers.",
    "Linear to scale: a 2M-subscriber base implies ~$32.8M net benefit per year.",
    "Only the population churn rate is an external assumption; the model's lift was measured on held-out data.",
], x=0.57, y0=0.78, dy=0.115, size=13, w=0.40)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 13 : Roadmap & risks ----------------
fig, ax = new_slide(pdf, 13)
header(ax, "How we de-risk & deploy", "A staged, measured rollout")
ax.text(0.05, 0.80, "90-day pilot", fontsize=15, color=NAVY, fontweight="bold", va="top")
bullets(ax, [
    "Weeks 1–2 — deploy monthly scoring, build EVaR list, assign personas, design offers.",
    "Weeks 3–8 — pilot in one region with a randomized CONTROL group to measure true incremental save.",
    "Weeks 9–12 — scale winners, tune offer mix per persona, stand up a monthly scoring pipeline.",
], x=0.05, y0=0.73, dy=0.075, size=12.5, w=0.9)
ax.text(0.05, 0.42, "Risks & mitigations", fontsize=15, color=RED, fontweight="bold", va="top")
bullets(ax, [
    "Balanced sample ≠ live 2% base → recalibrate probabilities before budgeting.",
    "Offer cannibalization (paying people who'd stay anyway) → control group + uplift modeling (treat the persuadable).",
    "Model drift → monthly retrain & monitoring; Fairness → exclude protected demographics (already weak), audit offers.",
], x=0.05, y0=0.35, dy=0.075, size=12.5, w=0.9)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 14 : Takeaways ----------------
fig, ax = new_slide(pdf, 14)
header(ax, "Key takeaways", "Why this proposal wins")
bullets(ax, [
    "Churn here is a fixable equipment & engagement problem — the data points straight at the levers.",
    "EVaR targeting is the differentiator: it protects 45% of revenue-at-risk from the top 20% — 1.6× a naive churn list.",
    "Persona-matched offers turn a score into specific actions; two-thirds of the at-risk pool is one upgrade program.",
    "The economics are compelling and robust: ~$1.64M net / 100k customers (164% ROI), positive in every scenario, ~$32.8M at 2M scale.",
    "Honest & deployable: explainable model, stated assumptions, control-group pilot, fairness & drift safeguards.",
], y0=0.78, dy=0.135, size=14.5)
pdf.savefig(fig); plt.close(fig)

# ---------------- Slide 15 : References ----------------
fig, ax = new_slide(pdf, 15)
header(ax, "References & data sources", "Citations", color=GREY)
refs = [
    "[1] MarketLine (2024). Wireless Telecommunication Services in the United States — Market size & growth.",
    "[2] IBISWorld (2024). Wireless Telecommunications Carriers in the US — Industry report.",
    "[3] T-Mobile US (2025). Q4 2024 results — postpaid phone churn 0.92%.",
    "[4] Verizon Communications (2025). Q4/FY2024 earnings — churn ~0.89%, ARPA $139.77.",
    "[5] SAS / Harvard Business Review — Cost of acquisition vs retention (5–10×); ~22%/yr attrition.",
    "[6] Bain & Company / Reichheld; Simon-Kucher — +5% retention → +25–95% profit.",
    "[7] Investopedia (2006). Cost Per Gross Add (CPGA) — ~$350 acquisition cost.",
    "[8] Course tutorial notebook & Cell2Cell telecom churn benchmark (AUC ~0.70 ceiling).",
    "[9] Dataset: Client.csv + Record.csv (provided with assignment), 100,000 customers.",
]
y = 0.80
for r in refs:
    ax.text(0.05, y, r, fontsize=12.5, color=INK, va="top"); y -= 0.072
ax.text(0.05, 0.14, "[10] Generative-AI assistance (Devin) — session / chat history:", fontsize=11, color=INK, va="top")
ax.text(0.07, 0.105, "https://app.devin.ai/sessions/513c994718b44c95a14cdce101e30bdb", fontsize=10.5, color=NAVY2, va="top")
ax.text(0.05, 0.06, "AI was used for code scaffolding and prose editing; all analysis, metrics and figures are reproduced by the accompanying notebook (Hazem_Douzi.ipynb).",
        fontsize=9.5, color=GREY, va="top", style="italic")
pdf.savefig(fig); plt.close(fig)

pdf.close()
print("Saved deck:", OUT, "(", os.path.getsize(OUT)//1024, "KB )")
