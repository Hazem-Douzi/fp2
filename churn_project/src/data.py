"""
Data loading, cleaning, and feature engineering for the Company A churn model.

Design principles (addressing prior-version critique):
  * ONE pipeline, deterministic, relative paths -> reproducible anywhere.
  * Protected/sensitive attributes are dropped BEFORE modeling (fairness).
  * Real behavioral feature engineering from the rich usage data, not raw dump.
  * Honest handling of missingness; junk columns removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Load Client.csv + Record.csv and merge on Customer_ID (1:1)."""
    client = pd.read_csv(C.CLIENT_CSV)
    record = pd.read_csv(C.RECORD_CSV)
    df = client.merge(record, on=C.ID_COL, how="inner", suffixes=("", "_rec"))
    # Drop any accidental duplicate columns from the suffix collision.
    df = df.loc[:, ~df.columns.str.endswith("_rec")]
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise num/den, guarding divide-by-zero -> 0."""
    den = den.replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create documented behavioral features that capture *change* and
    *friction* signals -- the things that actually precede churn."""
    out = df.copy()

    # Equipment age in MONTHS (real units -- the prior version faked these).
    out["eqp_months"] = out["eqpdays"].clip(lower=0) / 30.0

    # Usage trend: recent vs. lifetime. Negative => declining engagement.
    out["mou_trend"] = _safe_ratio(out["change_mou"], out["mou_Mean"].abs() + 1)
    out["rev_trend"] = _safe_ratio(out["change_rev"], out["rev_Mean"].abs() + 1)
    out["mou_3m_vs_life"] = _safe_ratio(out.get("avg3mou", 0), out.get("avgmou", 0) + 1)
    out["rev_3m_vs_life"] = _safe_ratio(out.get("avg3rev", 0), out.get("avgrev", 0) + 1)

    # Network friction: share of calls that drop/block/go unanswered.
    attempts = out.get("attempt_Mean", pd.Series(0, index=out.index)).fillna(0) + 1
    out["drop_rate"] = _safe_ratio(out.get("drop_blk_Mean", 0), attempts)
    out["dropvce_rate"] = _safe_ratio(out.get("drop_vce_Mean", 0), attempts)
    out["unanvce_rate"] = _safe_ratio(out.get("unan_vce_Mean", 0), attempts)
    out["incompletes"] = _safe_ratio(
        out.get("attempt_Mean", 0) - out.get("complete_Mean", 0), attempts
    )

    # Overage / bill-shock share of revenue.
    out["overage_share"] = _safe_ratio(out.get("ovrrev_Mean", 0), out["rev_Mean"].abs() + 1)

    # Customer-care intensity (frustration proxy).
    out["care_intensity"] = _safe_ratio(out.get("custcare_Mean", 0), out["mou_Mean"].abs() + 1)

    # Revenue per minute (price sensitivity proxy).
    out["rev_per_mou"] = _safe_ratio(out["rev_Mean"], out["mou_Mean"].abs() + 1)

    # Household/account structure (NON-protected: subscription mechanics only).
    out["inactive_subs"] = (out.get("uniqsubs", 0) - out.get("actvsubs", 0)).clip(lower=0)

    # Tenure buckets for narrative (kept numeric `months` for the model).
    return out


# ---------------------------------------------------------------------------
# Cleaning / column selection
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop protected attributes, junk columns, and obviously invalid rows."""
    out = df.copy()

    # Fairness: remove protected & sensitive attributes entirely.
    drop_cols = [c for c in C.PROTECTED_ATTRS if c in out.columns]
    # Data quality: remove high-missing / undocumented columns.
    drop_cols += [c for c in C.HIGH_MISSING_DROP if c in out.columns and c not in drop_cols]
    out = out.drop(columns=drop_cols, errors="ignore")

    # A handful of revenue/usage fields have a few NaNs or negatives from
    # billing adjustments. Clip impossible negatives on level (not change) cols.
    for col in ["rev_Mean", "mou_Mean", "totmrc_Mean", "eqpdays", "hnd_price"]:
        if col in out.columns:
            out[col] = out[col].clip(lower=0)

    return out


def build_feature_frame() -> tuple[pd.DataFrame, pd.Series, dict]:
    """End-to-end: returns (X, y, meta).

    X excludes the ID and target; protected attributes are already gone.
    Categorical columns are one-hot encoded. Numeric NaNs -> median.
    """
    raw = load_raw()
    meta = {
        "n_rows": int(len(raw)),
        "n_raw_cols": int(raw.shape[1]),
        "sample_churn_rate": float(raw[C.TARGET].mean()),
        "dropped_protected": [c for c in C.PROTECTED_ATTRS if c in raw.columns],
    }

    feat = engineer_features(raw)
    feat = clean(feat)

    y = feat[C.TARGET].astype(int)
    X = feat.drop(columns=[C.TARGET, C.ID_COL], errors="ignore")

    # Split column types robustly (object OR string dtype -> categorical).
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Numeric: median impute.
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Categorical: cast to string, fill NaN, limit cardinality, then one-hot.
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("MISSING")
        top = X[c].value_counts().nlargest(15).index
        X[c] = np.where(X[c].isin(top), X[c], "OTHER")
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    # Guard: ensure no protected attribute leaked through encoding.
    bad = [c for c in X.columns if any(c.lower().startswith(p) for p in
           ["ethnic", "marital", "income", "creditcd", "crclscod"])]
    assert not bad, f"Protected attribute leaked into features: {bad}"

    meta["n_model_features"] = int(X.shape[1])
    meta["n_numeric"] = len(num_cols)
    meta["n_categorical_raw"] = len(cat_cols)
    return X, y, meta


if __name__ == "__main__":
    X, y, meta = build_feature_frame()
    import json
    print(json.dumps(meta, indent=2, default=str))
    print("X shape:", X.shape)
    print("Sample features:", list(X.columns)[:20])
