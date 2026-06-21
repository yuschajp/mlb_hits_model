"""
calibration.py

Calibration diagnostics for binary prop predictions (hits, home runs).
Same philosophy as the soccer-trading-engine repo: always check whether
predicted probabilities actually match realized frequencies, not just
whether picks "won" -- a model can look profitable on a lucky stretch
while being badly overconfident, and this is the check that would catch it.

prob_col/outcome_col default to the hits schema ("p_hit"/"actual_hit") so
existing callers work unchanged; pass prob_col="p_hr", outcome_col=
"actual_hr" for the home run ledger.
"""

import numpy as np
import pandas as pd


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
