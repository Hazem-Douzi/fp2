"""
Business value layer: EVaR targeting, profit-curve threshold optimization,
uplift / persuadables modeling, and bear/base/bull scenario analysis.

All dollar assumptions are explicit and individually cited in references.md.
Nothing here is a hidden constant.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from . import config as C


# ---------------------------------------------------------------------------
# Scenario assumptions (each value is cited in references.md)
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    save_rate: float        # P(retain | contacted & would-have-churned)
    offer_cost: float       # $ retention offer per accepted save
    contact_cost: float     # $ outreach cost per customer contacted
    gross_margin: float     # margin on retained revenue
    clv_months: int         # horizon for customer lifetime value
    annual_churn: float     # population annual churn (context only)


# Benchmarks: win-back/save rates in telecom retention programs typically run
# 10-30% (bear/base/bull). CLV horizon ~ inverse of churn. See references.md.
SCENARIOS = [
    Scenario("Bear",  save_rate=0.12, offer_cost=35, contact_cost=4,
             gross_margin=0.55, clv_months=18, annual_churn=0.24),
    Scenario("Base",  save_rate=0.20, offer_cost=30, contact_cost=3,
             gross_margin=0.58, clv_months=24, annual_churn=0.24),
    Scenario("Bull",  save_rate=0.28, offer_cost=25, contact_cost=2,
             gross_margin=0.60, clv_months=30, annual_churn=0.24),
]


# ---------------------------------------------------------------------------
# Expected Value at Risk (EVaR) = P(churn) * customer lifetime value
# ---------------------------------------------------------------------------
def expected_value_at_risk(p_churn: np.ndarray, monthly_rev: np.ndarray,
                           margin: float, clv_months: int) -> np.ndarray:
    clv = np.clip(monthly_rev, 0, None) * margin * clv_months
    return p_churn * clv


# ---------------------------------------------------------------------------
# Capture curve: what fraction of *actual* churners is captured by ranking the
# population on a given score and contacting the top-k%.
# ---------------------------------------------------------------------------
def capture_curve(score: np.ndarray, y_true: np.ndarray,
                  grid: np.ndarray | None = None) -> pd.DataFrame:
    if grid is None:
        grid = np.linspace(0.02, 1.0, 50)
    order = np.argsort(-score)
    y_sorted = np.asarray(y_true)[order]
    total_churn = y_sorted.sum()
    rows = []
    n = len(y_sorted)
    for frac in grid:
        k = max(1, int(frac * n))
        captured = y_sorted[:k].sum()
        rows.append({"frac_contacted": frac,
                     "frac_churn_captured": captured / total_churn})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Profit curve: net $ benefit of contacting the top-k% ranked by EVaR.
# ---------------------------------------------------------------------------
def profit_curve(p_churn: np.ndarray, monthly_rev: np.ndarray,
                 y_true: np.ndarray, sc: Scenario,
                 grid: np.ndarray | None = None) -> pd.DataFrame:
    """For each contact fraction, compute expected net profit.

    For a contacted customer who *would* churn (prob p), a campaign with
    save_rate s retains them with prob s, yielding margin*rev*clv_months minus
    the offer cost; every contacted customer costs contact_cost.
    We score on EVaR so the highest-value at-risk customers are contacted first.
    """
    if grid is None:
        grid = np.linspace(0.02, 1.0, 50)
    clv = np.clip(monthly_rev, 0, None) * sc.gross_margin * sc.clv_months
    evar = p_churn * clv
    order = np.argsort(-evar)
    p_sorted = np.asarray(p_churn)[order]
    clv_sorted = clv[order]
    n = len(p_sorted)
    rows = []
    for frac in grid:
        k = max(1, int(frac * n))
        idx = slice(0, k)
        # Expected saves = sum over contacted of p_churn * save_rate
        exp_saves = (p_sorted[idx] * sc.save_rate).sum()
        gross_saved_value = (p_sorted[idx] * sc.save_rate * clv_sorted[idx]).sum()
        offer_costs = exp_saves * sc.offer_cost
        contact_costs = k * sc.contact_cost
        net = gross_saved_value - offer_costs - contact_costs
        rows.append({
            "frac_contacted": frac,
            "n_contacted": k,
            "exp_customers_saved": float(exp_saves),
            "gross_value_saved": float(gross_saved_value),
            "campaign_cost": float(offer_costs + contact_costs),
            "net_profit": float(net),
            "roi": float(gross_saved_value / (offer_costs + contact_costs))
            if (offer_costs + contact_costs) > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def optimal_operating_point(pc: pd.DataFrame) -> dict:
    """Pick the contact fraction that maximizes net profit."""
    best = pc.loc[pc["net_profit"].idxmax()]
    return best.to_dict()


# ---------------------------------------------------------------------------
# Uplift / persuadables (T-learner).
# ---------------------------------------------------------------------------
def t_learner_uplift(X: pd.DataFrame, treatment: np.ndarray,
                     outcome_retained: np.ndarray) -> np.ndarray:
    """Two-model uplift: model P(retain) under treatment vs control, predict
    the difference (the treatment effect / persuadability) for everyone.

    Returns per-customer uplift = P(retain|treat) - P(retain|control).
    """
    t = np.asarray(treatment).astype(bool)
    base = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                random_state=C.RANDOM_STATE, eval_metric="logloss",
                tree_method="hist")
    m_treat = XGBClassifier(**base).fit(X[t], outcome_retained[t])
    m_ctrl = XGBClassifier(**base).fit(X[~t], outcome_retained[~t])
    return m_treat.predict_proba(X)[:, 1] - m_ctrl.predict_proba(X)[:, 1]


def t_learner_uplift_fit_predict(X_train: pd.DataFrame, treatment: np.ndarray,
                                 outcome_retained: np.ndarray,
                                 X_pred: pd.DataFrame) -> np.ndarray:
    """Fit the two-model uplift learner on training rows, predict uplift on a
    held-out frame. uplift = P(retain|treat) - P(retain|control)."""
    t = np.asarray(treatment).astype(bool)
    base = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                random_state=C.RANDOM_STATE, eval_metric="logloss",
                tree_method="hist")
    m_treat = XGBClassifier(**base).fit(X_train[t], outcome_retained[t])
    m_ctrl = XGBClassifier(**base).fit(X_train[~t], outcome_retained[~t])
    return m_treat.predict_proba(X_pred)[:, 1] - m_ctrl.predict_proba(X_pred)[:, 1]


def scenario_summary(p_churn: np.ndarray, monthly_rev: np.ndarray,
                     y_true: np.ndarray, frac: float) -> list[dict]:
    """Run all three scenarios at a fixed contact fraction; return summaries."""
    out = []
    for sc in SCENARIOS:
        pc = profit_curve(p_churn, monthly_rev, y_true, sc,
                          grid=np.array([frac]))
        row = pc.iloc[0].to_dict()
        row.update({"scenario": sc.name, **asdict(sc)})
        out.append(row)
    return out
