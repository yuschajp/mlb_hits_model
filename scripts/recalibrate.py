"""
recalibrate.py

Fits a Platt scaling correction on the graded hit prediction ledger and
saves the calibration parameters to data/hit_calibration.json.

Platt scaling fits a logistic regression on the logit of the raw model
probabilities vs actual outcomes:

    logit(p_calibrated) = slope * logit(p_raw) + intercept

This is the standard way to correct systematic overconfidence or
underconfidence in a model's probability outputs. The result is two
numbers -- slope and intercept -- that get applied to every future
prediction before it's logged or compared against odds.

Run with: python3 scripts/recalibrate.py

After running, restart run_daily.py and find_value.py -- they will
automatically pick up the new calibration parameters.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.calibration import brier_score, calibration_table
from src.ledger import load_ledger

ROOT         = Path(__file__).resolve().parents[1]
HITS_LEDGER  = ROOT / "data" / "ledger" / "predictions_log.csv"
PARAMS_OUT   = ROOT / "data" / "hit_calibration.json"
MIN_SAMPLES  = 200  # refuse to fit on fewer than this many graded predictions


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_platt_scaling(probs, outcomes):
    """
    Fits slope and intercept via gradient descent on binary cross-entropy.
    No sklearn dependency -- pure numpy, consistent with the rest of this
    project's zero-extra-dependency philosophy.
    """
    X = logit(np.array(probs, dtype=float))
    y = np.array(outcomes, dtype=float)

    # Initialize: slope=1 (identity), intercept=0
    slope = 1.0
    intercept = 0.0
    lr = 0.01
    n_iter = 2000

    for _ in range(n_iter):
        p_hat = sigmoid(slope * X + intercept)
        error = p_hat - y
        grad_slope = float(np.mean(error * X))
        grad_intercept = float(np.mean(error))
        slope -= lr * grad_slope
        intercept -= lr * grad_intercept

    return slope, intercept


def apply_calibration(raw_probs, slope, intercept):
    X = logit(np.array(raw_probs, dtype=float))
    return sigmoid(slope * X + intercept)


def main():
    df = load_ledger(HITS_LEDGER)
    graded = df[df["graded"] == True].dropna(subset=["actual_hit"])  # noqa: E712

    print(f"Graded predictions available: {len(graded)}")

    if len(graded) < MIN_SAMPLES:
        print(f"Need at least {MIN_SAMPLES} graded predictions to fit calibration. "
              f"Keep running the model daily and try again.")
        return

    probs    = graded["p_hit"].values
    outcomes = graded["actual_hit"].values

    # --- Before calibration ---
    score_before = brier_score(graded)
    cal_before   = calibration_table(graded)
    print(f"\nBefore calibration:")
    print(f"  Brier score: {score_before:.4f}")
    print(cal_before[["PredRange", "N", "AvgPredicted", "ActualFrequency"]].to_string(index=False))

    # --- Fit ---
    slope, intercept = fit_platt_scaling(probs, outcomes)
    print(f"\nFitted Platt scaling: slope={slope:.4f}, intercept={intercept:.4f}")

    # --- After calibration ---
    calibrated_probs = apply_calibration(probs, slope, intercept)
    graded_cal = graded.copy()
    graded_cal["p_hit_calibrated"] = calibrated_probs

    score_after = brier_score(graded_cal, prob_col="p_hit_calibrated")
    cal_after   = calibration_table(graded_cal, prob_col="p_hit_calibrated")
    print(f"\nAfter calibration:")
    print(f"  Brier score: {score_after:.4f}  (improvement: {score_before - score_after:+.4f})")
    print(cal_after[["PredRange", "N", "AvgPredicted", "ActualFrequency"]].to_string(index=False))

    # --- What this means in plain English ---
    sample_raw  = [0.60, 0.65, 0.70, 0.75, 0.80]
    sample_cal  = apply_calibration(sample_raw, slope, intercept)
    print(f"\nCalibration correction (what changes in practice):")
    print(f"  {'Raw':>8}  {'Calibrated':>12}  {'Shift':>8}")
    for r, c in zip(sample_raw, sample_cal):
        print(f"  {r:>8.1%}  {c:>12.1%}  {c-r:>+8.1%}")

    # --- Save ---
    params = {
        "slope": slope,
        "intercept": intercept,
        "fitted_on_n": len(graded),
        "brier_before": round(score_before, 4),
        "brier_after": round(score_after, 4),
        "note": (
            "Platt scaling correction fitted on graded hit predictions. "
            "Apply via: p_calibrated = sigmoid(slope * logit(p_raw) + intercept). "
            "Refit after every ~200 new graded predictions."
        ),
    }
    PARAMS_OUT.write_text(json.dumps(params, indent=2))
    print(f"\nCalibration parameters saved to {PARAMS_OUT}")
    print("run_daily.py and find_value.py will pick these up automatically on next run.")


if __name__ == "__main__":
    main()
