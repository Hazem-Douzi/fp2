"""
End-to-end, deterministic pipeline. Run from the project root:

    python -m src.run_pipeline

Produces:
    outputs/results/results.json      -- every number used in the deck
    outputs/figures/*.png             -- every chart used in the deck
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import business as B
from . import config as C
from . import data as D
from . import model as M
from . import plots as P

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(C.RANDOM_STATE)


def main() -> dict:
    results: dict = {}

    # ---------------------------------------------------------------- data
    X, y, meta = D.build_feature_frame()
    results["data"] = meta
    sample_rate = meta["sample_churn_rate"]

    # Keep raw monthly revenue aligned to X rows for the value layer.
    monthly_rev = D.load_raw().loc[X.index, "rev_Mean"].clip(lower=0).fillna(0).to_numpy()

    X_tr, X_te, y_tr, y_te, rev_tr, rev_te = train_test_split(
        X, y, monthly_rev, test_size=C.TEST_SIZE, stratify=y,
        random_state=C.RANDOM_STATE)

    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    # ------------------------------------------------ model comparison
    print("[v0] Training model zoo...")
    models = M.build_models(scale_pos_weight=spw)
    comparison = {}
    fitted = {}
    for name, mdl in models.items():
        metrics, fit = M.evaluate(mdl, X_tr, y_tr, X_te, y_te)
        comparison[name] = metrics
        fitted[name] = fit
        print(f"[v0]   {name}: AUC={metrics['auc']:.4f}")
    results["model_comparison"] = comparison

    # ------------------------------------------------ tuning (XGBoost)
    print("[v0] Tuning XGBoost (RandomizedSearchCV)...")
    search_space = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "n_estimators": [300, 500, 700],
        "min_child_weight": [3, 5, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    base_xgb = XGBClassifier(scale_pos_weight=spw, eval_metric="auc",
                             n_jobs=-1, random_state=C.RANDOM_STATE,
                             tree_method="hist")
    search = RandomizedSearchCV(
        base_xgb, search_space, n_iter=12,
        scoring="roc_auc", cv=3, random_state=C.RANDOM_STATE, n_jobs=-1)
    search.fit(X_tr, y_tr)
    best_xgb = search.best_estimator_
    results["tuning"] = {
        "best_params": search.best_params_,
        "best_cv_auc": float(search.best_score_),
    }
    print(f"[v0]   Best CV AUC={search.best_score_:.4f} params={search.best_params_}")

    # Cross-val stability of tuned model.
    cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    cv_scores = cross_val_score(best_xgb, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    results["cv"] = {"mean_auc": float(cv_scores.mean()),
                     "std_auc": float(cv_scores.std()),
                     "folds": [float(s) for s in cv_scores]}

    # ------------------------------------------------ calibration
    print("[v0] Calibrating probabilities (isotonic)...")
    calibrated = CalibratedClassifierCV(best_xgb, method="isotonic", cv=3)
    calibrated.fit(X_tr, y_tr)
    proba_sample = calibrated.predict_proba(X_te)[:, 1]

    # Reliability curve before/after.
    raw_proba = best_xgb.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    frac_pos_raw, mean_pred_raw = calibration_curve(y_te, raw_proba, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_te, proba_sample, n_bins=10)
    from sklearn.metrics import brier_score_loss, roc_auc_score
    results["calibration"] = {
        "brier_raw": float(brier_score_loss(y_te, raw_proba)),
        "brier_calibrated": float(brier_score_loss(y_te, proba_sample)),
        "auc_calibrated": float(roc_auc_score(y_te, proba_sample)),
    }

    # Reweight calibrated probs to the true base rate for the value layer.
    p_true = M.reweight_to_base_rate(proba_sample, sample_rate, C.TRUE_BASE_RATE)
    results["base_rate"] = {"sample_rate": sample_rate,
                            "true_rate": C.TRUE_BASE_RATE,
                            "mean_p_sample": float(proba_sample.mean()),
                            "mean_p_true": float(p_true.mean())}

    # ------------------------------------------------ capture curves
    print("[v0] Building capture & profit curves...")
    evar = B.expected_value_at_risk(p_true, rev_te, B.SCENARIOS[1].gross_margin,
                                    B.SCENARIOS[1].clv_months)
    cap_evar = B.capture_curve(evar, y_te.to_numpy())
    cap_score = B.capture_curve(proba_sample, y_te.to_numpy())
    cap_rand = pd.DataFrame({"frac_contacted": cap_evar["frac_contacted"],
                             "frac_churn_captured": cap_evar["frac_contacted"]})
    # Capture at top-20%.
    def at(df, f):
        return float(df.iloc[(df["frac_contacted"] - f).abs().idxmin()]["frac_churn_captured"])
    results["capture_at_20pct"] = {
        "by_evar": at(cap_evar, 0.20),
        "by_score": at(cap_score, 0.20),
        "random": 0.20,
        "lift_evar_vs_random": at(cap_evar, 0.20) / 0.20,
    }

    # ------------------------------------------------ profit curve
    base_sc = B.SCENARIOS[1]
    pc = B.profit_curve(p_true, rev_te, y_te.to_numpy(), base_sc)
    opt = B.optimal_operating_point(pc)
    # Scale test-set economics up to the full population.
    scale = len(X) / len(X_te)
    results["profit_curve_base"] = {
        "optimal_frac": float(opt["frac_contacted"]),
        "optimal_net_profit_testset": float(opt["net_profit"]),
        "optimal_net_profit_population": float(opt["net_profit"] * scale),
        "optimal_roi": float(opt["roi"]),
        "customers_saved_population": float(opt["exp_customers_saved"] * scale),
    }

    # ------------------------------------------------ scenarios (bear/base/bull)
    scen = B.scenario_summary(p_true, rev_te, y_te.to_numpy(),
                              float(opt["frac_contacted"]))
    for s in scen:
        s["net_profit_population"] = s["net_profit"] * scale
        s["customers_saved_population"] = s["exp_customers_saved"] * scale
    results["scenarios"] = scen

    # ------------------------------------------------ SHAP
    print("[v0] Computing SHAP values...")
    import shap
    sample_idx = RNG.choice(len(X_te), size=min(2000, len(X_te)), replace=False)
    X_shap = X_te.iloc[sample_idx]
    explainer = shap.TreeExplainer(best_xgb)
    shap_vals = explainer.shap_values(X_shap)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    shap_imp = (pd.Series(mean_abs, index=X_te.columns)
                .sort_values(ascending=False).head(15))
    results["shap_top_features"] = {k: float(v) for k, v in shap_imp.items()}

    # ------------------------------------------------ uplift (T-learner)
    # NOTE: dataset is observational (no treatment/control). We SIMULATE a
    # retention campaign under documented response assumptions to demonstrate
    # the persuadables framework and size the segment. Clearly labeled as a
    # what-if simulation; a real RCT is recommended (see deck "next steps").
    print("[v0] Uplift simulation (T-learner)...")
    uplift_res = run_uplift_simulation(X_tr, X_te, y_tr, y_te, proba_sample)
    results["uplift"] = uplift_res

    # ------------------------------------------------ K-means personas
    print("[v0] K-means personas...")
    personas = run_kmeans_personas(X_te, y_te, p_true, rev_te)
    results["personas"] = personas

    # ------------------------------------------------ descriptive stats
    raw = D.load_raw()
    results["descriptive"] = {
        "arpu": float(raw["rev_Mean"].clip(lower=0).mean()),
        "median_tenure_months": float(raw["months"].median()),
        "eqp_corr_churn": float(pd.Series(raw["eqpdays"]).corr(raw[C.TARGET])),
    }

    # ------------------------------------------------ FIGURES
    print("[v0] Rendering figures...")
    P.fig_model_comparison(comparison)
    P.fig_calibration(mean_pred_raw, frac_pos_raw, mean_pred_cal, frac_pos_cal)
    P.fig_capture(cap_evar, cap_score, cap_rand)
    P.fig_profit_curve(pc, opt)
    P.fig_shap(shap_imp)
    P.fig_handset_churn(raw)
    P.fig_scenarios(scen)
    P.fig_uplift(uplift_res)
    P.fig_personas(personas)

    # ------------------------------------------------ persist
    out_path = C.RES_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"[v0] Wrote {out_path}")
    return results


def run_uplift_simulation(X_tr, X_te, y_tr, y_te, proba_sample) -> dict:
    """Simulate a campaign with heterogeneous treatment effect, then recover
    persuadables with a T-learner. Illustrative, documented as a simulation."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    n = len(X_tr)
    # Random 50/50 treatment assignment (as a real RCT would).
    treat = rng.integers(0, 2, size=n).astype(bool)
    # Baseline churn prob from the model on the training rows.
    base_xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             n_jobs=-1, random_state=C.RANDOM_STATE,
                             eval_metric="logloss", tree_method="hist")
    p_churn_tr = base_xgb.fit(X_tr, y_tr).predict_proba(X_tr)[:, 1]
    # Heterogeneous persuadability: customers with mid eqp age & high care calls
    # respond most. Build a latent responsiveness in [0, 0.4].
    eqp = X_tr["eqp_months"].to_numpy() if "eqp_months" in X_tr else np.zeros(n)
    care = X_tr["care_intensity"].to_numpy() if "care_intensity" in X_tr else np.zeros(n)
    resp = 0.30 * np.exp(-((eqp - 12) ** 2) / (2 * 6 ** 2))
    resp += 0.10 * (care > np.quantile(care, 0.7))
    resp = np.clip(resp, 0, 0.4)
    # Simulated retention outcome (1 = retained).
    churn_prob = p_churn_tr.copy()
    churn_prob[treat] = np.clip(churn_prob[treat] - resp[treat], 0, 1)
    churned = rng.random(n) < churn_prob
    retained = (~churned).astype(int)
    # T-learner recovers per-customer uplift.
    uplift_te = B.t_learner_uplift_fit_predict(X_tr, treat, retained, X_te)
    # Rank persuadables on the test set.
    order = np.argsort(-uplift_te)
    top20 = order[: int(0.20 * len(order))]
    return {
        "method": "T-learner (two-model), SIMULATED treatment response",
        "note": ("Observational data has no treatment/control; response is "
                 "simulated under documented assumptions to demonstrate the "
                 "persuadables framework. A live A/B holdout is recommended."),
        "mean_uplift_top20": float(uplift_te[top20].mean()),
        "mean_uplift_overall": float(uplift_te.mean()),
        "persuadable_share_positive_uplift": float((uplift_te > 0.01).mean()),
        "uplift_hist": np.histogram(uplift_te, bins=30)[0].tolist(),
        "uplift_edges": np.histogram(uplift_te, bins=30)[1].tolist(),
    }


def run_kmeans_personas(X_te, y_te, p_true, rev_te) -> dict:
    """Real K-means on standardized behavioral features; profile each cluster."""
    feats = [c for c in ["eqp_months", "mou_trend", "care_intensity",
                          "overage_share", "rev_per_mou", "drop_rate", "months"]
             if c in X_te.columns]
    Z = StandardScaler().fit_transform(X_te[feats].fillna(0))
    km = KMeans(n_clusters=4, n_init=10, random_state=C.RANDOM_STATE)
    labels = km.fit_predict(Z)
    df = X_te[feats].copy()
    df["churn"] = y_te.to_numpy()
    df["p_true"] = p_true
    df["rev"] = rev_te
    df["cluster"] = labels
    clusters = []
    for k in range(4):
        m = df[df["cluster"] == k]
        clusters.append({
            "cluster": int(k),
            "size": int(len(m)),
            "share": float(len(m) / len(df)),
            "churn_rate": float(m["churn"].mean()),
            "avg_rev": float(m["rev"].mean()),
            "avg_eqp_months": float(m["eqp_months"].mean()) if "eqp_months" in m else 0.0,
            "avg_care": float(m["care_intensity"].mean()) if "care_intensity" in m else 0.0,
        })
    return {"features_used": feats, "clusters": clusters}


if __name__ == "__main__":
    main()
