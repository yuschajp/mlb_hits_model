"""
calibration.py

Calibration diagnostics for the binary "did they get a hit" prediction.
Same philosophy as the soccer-trading-engine repo: always check whether
predicted probabilities actually match realized frequencies, not just
whether picks "won" -- a model can look profitable on a lucky stretch
while being badly overconfident, and this is the check that would catch it.
"""

import numpy as np
import pandas as pd


def brier_score(graded_df):
    """Binary Brier score: mean((p_hit - actual_hit)^2). 0 = perfect, 0.25 = always guessing 50%."""
    df = graded_df.dropna(subset=["actual_hit"])
    if df.empty:
        return None
    return float(((df["p_hit"] - df["actual_hit"]) ** 2).mean())


def calibration_table(graded_df, n_bins=5):
    """Buckets p_hit into bins and compares average predicted probability to actual hit rate."""
    df = graded_df.dropna(subset=["actual_hit"])
    if df.empty:
        return pd.DataFrame(columns=["PredRange", "N", "AvgPredicted", "ActualFrequency"])

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(df["p_hit"].to_numpy(), bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "PredRange": f"{bins[b]:.1f}-{bins[b + 1]:.1f}",
            "N": int(mask.sum()),
            "AvgPredicted": df["p_hit"].to_numpy()[mask].mean(),
            "ActualFrequency": df["actual_hit"].to_numpy()[mask].mean(),
        })
    return pd.DataFrame(rows)
