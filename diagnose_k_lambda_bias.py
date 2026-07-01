"""
diagnose_k_lambda_bias.py

Checks whether lambda_k itself is biased -- as opposed to the Poisson vs.
Negative Binomial distribution-shape question fit_k_dispersion.py answers.

Why this is a different question:
    fit_k_dispersion.py tests whether, GIVEN a lambda, the shape of the
    count distribution around it is right (Poisson vs NB). It found NB
    doesn't help -- and in fact NB moves P(over) in the wrong direction
    for the low-lambda bucket, which is a strong hint the problem isn't
    distribution shape at all.

    This script tests the more basic thing underneath that: is lambda_k
    itself systematically too high or too low, and does that bias depend
    on where lambda sits (i.e. on which pitchers it's calculated for)?
    That's a mean-calibration question, answerable directly by comparing
    lambda_k to actual_ks -- no distributional assumption required.

What it does:
    1. Loads graded rows from k_predictions_log.csv
    2. Reports overall correlation and a simple OLS fit of
       actual_ks ~ intercept + slope * lambda_k
       - slope < 1 and/or intercept > 0 in a way that makes the fitted
         line cross below the actual data at low lambda is the signature
         of shrinkage pulling weak-K pitchers' lambda up too aggressively.
    3. Buckets predictions by lambda_k range and shows avg lambda_k vs
       avg actual_ks per bucket -- a direct calibration-of-the-mean
       check, independent of Poisson/NB.
    4. Splits low-lambda predictions by season_k_per_9 to check whether
       the bias specifically concentrates in low-true-rate pitchers (the
       shrinkage-toward-league-average hypothesis) vs. something else
       (e.g. opponent factor or innings estimate).
    5. If bias is present, prints a suggested linear recalibration
       (lambda_corrected = a + b * lambda_k) fit to zero out the bias in
       an OLS sense -- same idea as hit_model.py's existing Platt scaling,
       just applied to lambda instead of a probability.

Run with: python3 diagnose_k_lambda_bias.py path/to/k_predictions_log.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def ols_fit(x, y):
    """Simple least-squares fit of y = a + b*x. Returns (a, b, r)."""
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    var_x = sum((xi - x_mean) ** 2 for xi in x)
    var_y = sum((yi - y_mean) ** 2 for yi in y)
    b = cov / var_x if var_x > 0 else 0.0
    a = y_mean - b * x_mean
    r = cov / (var_x * var_y) ** 0.5 if var_x > 0 and var_y > 0 else 0.0
    return a, b, r


def bucket_report(df, col, buckets, label):
    print(f"\n{label}")
    print(f"{'Bucket':<14}{'N':>5}{'AvgLambda':>12}{'AvgActual':>12}{'Bias(pred-act)':>16}")
    for lo, hi in buckets:
        sub = df[(df[col] >= lo) & (df[col] < hi)]
        if len(sub) == 0:
            continue
        avg_lam = sub["lambda_k"].mean()
        avg_act = sub["actual_ks"].mean()
        print(f"{f'{lo}-{hi}':<14}{len(sub):>5}{avg_lam:>12.2f}{avg_act:>12.2f}{avg_lam - avg_act:>16.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger_csv", help="Path to k_predictions_log.csv")
    args = parser.parse_args()

    path = Path(args.ledger_csv)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    df = df[df["graded"].astype(str).isin(["True", "true", "1", "1.0"])]
    df = df.dropna(subset=["lambda_k", "actual_ks", "season_k_per_9"])

    if len(df) < 20:
        print(f"Warning: only {len(df)} graded rows -- treat results as provisional.")

    lam = df["lambda_k"].astype(float).tolist()
    actual = df["actual_ks"].astype(float).tolist()

    print(f"Graded games: {len(df)}")
    print(f"Mean lambda_k: {sum(lam)/len(lam):.3f}   Mean actual_ks: {sum(actual)/len(actual):.3f}")
    overall_bias = sum(lam)/len(lam) - sum(actual)/len(actual)
    print(f"Overall bias (avg predicted - avg actual): {overall_bias:+.3f}")

    a, b, r = ols_fit(lam, actual)
    print(f"\nOLS fit: actual_ks = {a:.3f} + {b:.3f} * lambda_k   (r = {r:.3f})")
    if b < 0.9:
        print(f"  Slope < 1 -- predictions swing more than reality does across the "
              f"lambda range. Consistent with lambda being pulled toward a central "
              f"value (shrinkage) more than it should be.")
    if a > 0.3:
        print(f"  Positive intercept of {a:.2f} -- even at lambda=0 the fit implies "
              f"actual Ks above 0, which combined with slope<1 means low-lambda "
              f"predictions sit BELOW the true relationship: lambda is too low there? "
              f"(check against the bucket table below to see which direction it "
              f"actually breaks)")

    # Mean-calibration by predicted lambda range
    lam_buckets = [(0, 3), (3, 4.5), (4.5, 6), (6, 7.5), (7.5, 10), (10, 15)]
    bucket_report(df, "lambda_k", lam_buckets, "Calibration by lambda_k range (predicted mean vs actual mean):")

    # Check whether bias concentrates in genuinely low-true-rate pitchers
    # (season_k_per_9 low) vs. lambda being low for other reasons
    # (opponent factor, short expected innings, etc.)
    print("\nLow-lambda predictions (lambda_k < 5.5) split by season_k_per_9:")
    low_lam = df[df["lambda_k"] < 5.5]
    if len(low_lam) > 0:
        k9_buckets = [(0, 7), (7, 8.5), (8.5, 10), (10, 15)]
        bucket_report(low_lam, "season_k_per_9", k9_buckets,
                      "  (if bias concentrates in low season_k_per_9 rows, that points "
                      "at the shrinkage prior in stabilized_k_per_9 pulling weak-K "
                      "pitchers up too hard)")
    else:
        print("  No rows with lambda_k < 5.5 -- skip.")

    # Suggested linear recalibration of lambda itself
    print("\n--- Suggested lambda recalibration ---")
    print(f"If you want to correct lambda directly (keep Poisson, fix the input "
          f"instead of the distribution): lambda_corrected = a + b * lambda_k")
    print(f"  a = {a:.4f}")
    print(f"  b = {b:.4f}")
    print(f"This is the same idea as hit_model.py's existing Platt-scaling "
          f"calibration (apply_calibration/load_calibration_params), just applied "
          f"to lambda instead of the final probability. Only apply this once you've "
          f"confirmed via the bucket tables above that the bias direction makes "
          f"sense (it should shrink lambda toward actual for the buckets where "
          f"AvgLambda > AvgActual, and vice versa) -- don't apply blindly off the "
          f"single overall slope/intercept if the bias direction flips across "
          f"buckets, since that would mean a linear correction is too crude and "
          f"the real fix is in a specific input (e.g. the shrinkage prior) rather "
          f"than a global rescale.")


if __name__ == "__main__":
    main()
