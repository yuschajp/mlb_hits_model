"""
calibration.py

Calibration diagnostics for binary prop predictions (hits, home runs),
plus Platt scaling correction for systematic over/underconfidence.

prob_col/outcome_col default to the hits schema ("p_hit"/"actual_hit") so
existing callers work unchanged; pass prob_col="p_hr", outcome_col=
"actual_hr" for the home run ledger.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIT_CALIBRATION_PATH = ROOT / "data" / "hit_calibration.json"


def brier_score(graded_df, prob_col="p_hit", outcome_col="actual_hit"):
    """Binary Brier score: mean((prob - outcome)^2). 0 = perfect, 0.25 = always guessing 50%."""
    df = graded_df.dropna(subset=[outcome_col])
    if df.empty:
        return None
    return float(((df[prob_col] - df[outcome_col]) ** 2).mean())


def calibration_table(graded_df, n_bins=5, prob_col="p_hit", outcome_col="actual_hit"):
    """Buckets predicted probability into bins and compares average prediction to actual frequency."""
    df = graded_df.dropna(subset=[outcome_col])
    if df.empty:
        return pd.DataFrame(columns=["PredRange", "N", "AvgPredicted", "ActualFrequency"])

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(df[prob_col].to_numpy(), bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "PredRange": f"{bins[b]:.1f}-{bins[b + 1]:.1f}",
            "N": int(mask.sum()),
            "AvgPredicted": df[prob_col].to_numpy()[mask].mean(),
            "ActualFrequency": df[outcome_col].to_numpy()[mask].mean(),
        })
    return pd.DataFrame(rows)


# ── Platt scaling ─────────────────────────────────────────────────────────────

def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def load_calibration_params(path=None):
    """
    Load saved Platt scaling parameters from disk.
    Returns (slope, intercept) or (1.0, 0.0) if no params file exists
    (identity transform -- no correction applied).
    """
    path = Path(path) if path else HIT_CALIBRATION_PATH
    if not path.exists():
        return 1.0, 0.0
    params = json.loads(path.read_text())
    return float(params["slope"]), float(params["intercept"])


def apply_calibration(raw_prob, slope=1.0, intercept=0.0):
    """
    Apply Platt scaling to a single raw probability or an array of them.
    With slope=1.0, intercept=0.0 this is the identity (no correction).
    """
    return float(_sigmoid(slope * _logit(raw_prob) + intercept))
