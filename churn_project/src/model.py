"""
Model training, tuning, calibration, and evaluation.

Addresses prior-version critique:
  * Imbalance handled via scale_pos_weight (though sample is ~50/50, this keeps
    the pipeline correct if re-run on the true distribution).
  * Probabilities CALIBRATED (isotonic) so P(churn) is a real probability.
  * Calibrated scores RE-WEIGHTED to the true population base rate so EVaR and
    the business case live on a realistic scale (prior-prevalence correction).
  * Decision threshold chosen on the PROFIT curve, not a naive 0.5.
  * Honest model comparison: LogReg, RandomForest, XGBoost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
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
    true population base rate using the standard prior-correction formula.

        p_true = (p * tau) / ((p * tau) + (1 - p) * (1 - tau) * ... )

    Implemented via odds adjustment:
        odds_true = odds_sample * (true/(1-true)) / (sample/(1-sample))
    """
    p = np.clip(p_sample, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    adj = (true_rate / (1 - true_rate)) / (sample_rate / (1 - sample_rate))
    odds_true = odds * adj
    return odds_true / (1 + odds_true)


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def build_models(scale_pos_weight: float) -> dict:
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
    }


def evaluate(model, X_tr, y_tr, X_te, y_te) -> dict:
    """Fit + score one model. Returns metrics dict and fitted estimator."""
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
