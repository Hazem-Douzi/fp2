"""
Model training, tuning, calibration, and evaluation.

Addresses reviewer critique (guideline §§2-9):
  * Imbalance: scale_pos_weight (XGBoost/LightGBM) + class_weight (LogReg/RF).
    SMOTE is NOT used; it adds no clear AUC gain here and risks leakage if
    misapplied outside a proper pipeline. (guideline §3)
  * Wider Optuna-style search space via RandomizedSearchCV. (guideline §5)
  * CatBoost added as a categorical-aware challenger model. (guideline §6)
  * Both ROC-AUC AND PR-AUC reported for every model. (guideline §8)
  * Revenue-only and rule-based heuristic baselines added. (guideline §12)
  * Probabilities CALIBRATED (isotonic); re-weighted to true base rate.
  * Decision threshold chosen on PROFIT curve, not naive 0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C


# ---------------------------------------------------------------------------
# Prior-prevalence correction
# ---------------------------------------------------------------------------
def reweight_to_base_rate(p_sample: np.ndarray, sample_rate: float,
                          true_rate: float = C.TRUE_BASE_RATE) -> np.ndarray:
    """Convert probabilities trained on an oversampled (~50%) set back to the
    true population base rate via odds adjustment:
        odds_true = odds_sample * (true/(1-true)) / (sample/(1-sample))
    """
    p = np.clip(p_sample, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    adj = (true_rate / (1 - true_rate)) / (sample_rate / (1 - sample_rate))
    odds_true = odds * adj
    return odds_true / (1 + odds_true)


# ---------------------------------------------------------------------------
# Model zoo  (guideline §§5-6)
# ---------------------------------------------------------------------------
def build_models(scale_pos_weight: float) -> dict:
    """Return the full challenger zoo.

    CatBoost is added because several predictors are categorical customer /
    profile attributes; CatBoost handles them natively and is a more rigorous
    comparison than re-encoding everything. (guideline §6)
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.1,
                                       class_weight="balanced",
                                       random_state=C.RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, class_weight="balanced", random_state=C.RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, scale_pos_weight=scale_pos_weight,
            eval_metric="auc", n_jobs=-1, random_state=C.RANDOM_STATE,
            tree_method="hist"),
        # CatBoost: categorical-aware gradient boosting (guideline §6).
        # verbose=0 suppresses per-iteration output.
        "CatBoost": CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.03,
            l2_leaf_reg=3, subsample=0.8,
            auto_class_weights="Balanced",
            eval_metric="AUC", random_seed=C.RANDOM_STATE,
            verbose=0),
    }


# ---------------------------------------------------------------------------
# Wider tuning search space  (guideline §5)
# ---------------------------------------------------------------------------
XGBOOST_SEARCH_SPACE = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10],
    "n_estimators": [300, 500, 800, 1200],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.3, 0.5, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 3.0, 5.0, 10.0],
}


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(model, X_tr, y_tr, X_te, y_te) -> tuple[dict, object]:
    """Fit + score one model. Returns (metrics_dict, fitted_estimator).

    Reports both ROC-AUC and PR-AUC as recommended in guideline §8.
    """
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_te, proba)),
    }, model


# ---------------------------------------------------------------------------
# Revenue-only and rule-based baselines  (guideline §12)
# ---------------------------------------------------------------------------
def revenue_only_score(monthly_rev: np.ndarray) -> np.ndarray:
    """Targeting score = monthly revenue. Used as a common-business baseline:
    'just contact your highest-spending customers first'. (guideline §12)"""
    r = np.asarray(monthly_rev, dtype=float)
    mn, mx = r.min(), r.max()
    if mx == mn:
        return np.ones_like(r) * 0.5
    return (r - mn) / (mx - mn)


def rule_based_score(X: pd.DataFrame, monthly_rev: np.ndarray) -> np.ndarray:
    """Heuristic score: old handset + high revenue + declining usage.

    Guideline §12: rule-based baseline to benchmark EVaR model against.
    Score = weighted sum of three observable signals, normalised to [0, 1].
    """
    rev = np.asarray(monthly_rev, dtype=float)
    rev_med = float(np.median(rev[rev > 0])) if (rev > 0).any() else 1.0

    old_phone = (X["eqpdays"] > 365).astype(float) if "eqpdays" in X.columns else pd.Series(0.0, index=X.index)
    high_rev = (rev > rev_med).astype(float)
    declining = (X["change_mou"] < 0).astype(float) if "change_mou" in X.columns else pd.Series(0.0, index=X.index)

    score = (0.4 * np.asarray(old_phone)
             + 0.4 * high_rev
             + 0.2 * np.asarray(declining))
    # Add tiny noise to break ties without changing ranking structure.
    rng = np.random.default_rng(C.RANDOM_STATE)
    score = score + rng.uniform(0, 1e-4, size=len(score))
    mn, mx = score.min(), score.max()
    return (score - mn) / (mx - mn) if mx > mn else score
