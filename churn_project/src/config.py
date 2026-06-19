"""
Central configuration for the Company A churn project.

All paths are RELATIVE to the project root (the `churn_project/` directory),
so the pipeline runs unchanged on any machine / grader environment.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (relative & portable)
# ----------------------------------------------------------------------------
# config.py lives in churn_project/src/, so project root is its parent's parent.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "telecom"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"

CLIENT_CSV = DATA_DIR / "Client.csv"
RECORD_CSV = DATA_DIR / "Record.csv"

for _d in (FIG_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ----------------------------------------------------------------------------
# Domain / business constants
# ----------------------------------------------------------------------------
# The modeling sample is oversampled to ~50% churn. Company A's real-world
# monthly churn for US wireless post-paid is far lower. We reweight model
# outputs to this true base rate so probabilities and EVaR are on a real scale.
# Source: see references.md (US wireless monthly churn benchmarks).
TRUE_BASE_RATE = 0.02  # ~2% monthly churn (24% annualized) -- documented assumption

# ----------------------------------------------------------------------------
# FAIRNESS: protected / sensitive attributes that MUST be excluded from the
# model. We deliberately drop these so targeting cannot be driven by them.
# This is an ethics requirement, not an accuracy decision.
# ----------------------------------------------------------------------------
PROTECTED_ATTRS = [
    "ethnic",     # Ethnicity roll-up code  -- protected
    "marital",    # Marital status          -- protected
    "income",     # Estimated income        -- proxy for socioeconomic status
    "creditcd",   # Credit card indicator   -- socioeconomic proxy
    "crclscod",   # Credit class code       -- socioeconomic proxy
    "kid0_2", "kid3_5", "kid6_10", "kid11_15", "kid16_17",  # household children
    "adults",     # number of adults        -- household composition
    "ownrent",    # home owner/renter        -- socioeconomic proxy
    "dwlltype", "dwllsize",  # dwelling type/size -- socioeconomic proxy
    "lor",        # length of residence
    "HHstatin",   # premier household status indicator
    "infobase",   # third-party demographic match
    "numbcars",   # known number of vehicles
    "truck", "rv",  # vehicle ownership
]

# Columns with very high missingness or no documented business meaning that we
# drop for data-quality reasons (separate from the fairness drops above).
HIGH_MISSING_DROP = ["numbcars", "dwllsize", "HHstatin", "infobase", "lor"]

# Identifier (never a feature)
ID_COL = "Customer_ID"
TARGET = "churn"
