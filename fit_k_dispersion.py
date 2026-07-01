"""
fit_k_dispersion.py

Fits the Negative Binomial dispersion parameter (alpha) for the strikeout
model from graded predictions in k_predictions_log.csv.

Why this exists:
    The current k_model.py assumes strikeouts-per-start are Poisson
    (variance == mean). Rolling calibration on 68 graded games shows the
    0.0-0.2 predicted-probability bucket runs hot (14% predicted P(over)
    vs 5% actual over rate) while 0.4-0.6 is dead-on. That pattern -- bias
    concentrated in the tails, fine in the middle -- is the signature of
    overdispersion: the real variance of Ks-per-start is larger than a
    Poisson with the same mean would predict (start-to-start variance in
    command, bullpen hooks, etc.). A Negative Binomial with the same mean
    (lambda) but a fitted variance = lambda + alpha * lambda^2 corrects
    this without needing to change how lambda itself is computed.

What this script does:
    1. Loads k_predictions_log.csv
    2. Keeps only graded rows (graded == True, actual_ks not null)
    3. Fits alpha by maximum likelihood: for each graded (lambda_k,
       actual_ks) pair, find the alpha that maximizes the sum of NB
       log-likelihoods across all pairs. lambda_k is taken as given
       (already fit/blended elsewhere) -- only alpha is estimated here.
    4. Prints the fitted alpha and a before/after calibration comparison
       table (same bucket structure as your grading script), so you can
       see directly whether NB tightens the 0.0-0.2 and 0.6-0.8 buckets
       vs. the Poisson baseline.

Run with: python3 fit_k_dispersion.py path/to/k_predictions_log.csv

Output: prints the fitted alpha to paste into k_model.py's K_DISPERSION
constant. Also writes k_dispersion_fit.json with the value + fit metadata
if you want to load it programmatically instead of hardcoding it.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


def nb_log_pmf(k, mu, alpha):
    """Log PMF of Negative Binomial parameterized by mean (mu) and
    dispersion (alpha), where variance = mu + alpha * mu^2.
    alpha -> 0 recovers Poisson."""
    if mu <= 0:
        return 0.0 if k == 0 else float("-inf")
    if alpha <= 1e-8:
        # Poisson limit
        return -mu + k * math.log(mu) - math.lgamma(k + 1)
    r = 1.0 / alpha
    p = r / (r + mu)
    return (
        math.lgamma(k + r)
        - math.lgamma(r)
        - math.lgamma(k + 1)
        + r * math.log(p)
        + k * math.log(1 - p)
    )


def nb_pmf(k, mu, alpha):
    return math.exp(nb_log_pmf(k, mu, alpha))


def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def neg_log_likelihood(alpha, pairs):
    """pairs: list of (lambda_k, actual_ks)"""
    total = 0.0
    for lam, k in pairs:
        total -= nb_log_pmf(int(round(k)), lam, alpha)
    return total


def fit_alpha(pairs, lo=1e-6, hi=5.0, tol=1e-6, max_iter=200):
    """Golden-section search for the alpha minimizing negative log-likelihood.
    Simple and dependency-free (no scipy needed)."""
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = neg_log_likelihood(c, pairs)
    fd = neg_log_likelihood(d, pairs)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = neg_log_likelihood(c, pairs)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = neg_log_likelihood(d, pairs)
    return (a + b) / 2


def bucket_calibration(pairs, alpha, line_lookup):
    """
    Compare predicted P(over line) vs actual over-rate, in buckets,
    for both Poisson (alpha=0) and fitted-NB, using the same buckets as
    the daily grading script (0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0).
    """
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    rows = []
    for lo, hi in buckets:
        poisson_preds, nb_preds, actuals = [], [], []
        for lam, actual_ks, line in line_lookup:
            threshold = math.floor(line)
            p_over_poisson = 1 - sum(poisson_pmf(k, lam) for k in range(threshold + 1))
            p_over_nb = 1 - sum(nb_pmf(k, lam, alpha) for k in range(threshold + 1))
            if lo <= p_over_poisson < hi or (hi == 1.0 and p_over_poisson == 1.0):
                poisson_preds.append(p_over_poisson)
                nb_preds.append(p_over_nb)
                actuals.append(1.0 if actual_ks > line else 0.0)
        if actuals:
            rows.append({
                "bucket": f"{lo}-{hi}",
                "n": len(actuals),
                "poisson_avg_pred": sum(poisson_preds) / len(poisson_preds),
                "nb_avg_pred": sum(nb_preds) / len(nb_preds),
                "actual_freq": sum(actuals) / len(actuals),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger_csv", help="Path to k_predictions_log.csv")
    parser.add_argument("--out", default="k_dispersion_fit.json",
                         help="Where to write the fitted alpha + metadata")
    args = parser.parse_args()

    path = Path(args.ledger_csv)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    df = df[df["graded"].astype(str).isin(["True", "true", "1", "1.0"])]
    df = df.dropna(subset=["lambda_k", "actual_ks", "line"])

    if len(df) < 20:
        print(f"Warning: only {len(df)} graded rows -- alpha estimate will be noisy. "
              f"Treat as provisional until you have 100+.")

    pairs = list(zip(df["lambda_k"].astype(float), df["actual_ks"].astype(float)))
    line_lookup = list(zip(df["lambda_k"].astype(float), df["actual_ks"].astype(float), df["line"].astype(float)))

    alpha = fit_alpha(pairs)

    nll_poisson = neg_log_likelihood(1e-8, pairs)
    nll_nb = neg_log_likelihood(alpha, pairs)

    print(f"Graded games used: {len(pairs)}")
    print(f"Fitted alpha (NB dispersion): {alpha:.4f}")
    print(f"  (alpha=0 is exactly Poisson; higher alpha = fatter tails / more overdispersion)")
    print(f"Negative log-likelihood -- Poisson: {nll_poisson:.2f}   NB: {nll_nb:.2f}")
    if nll_nb < nll_poisson:
        print(f"  NB fits better by {nll_poisson - nll_nb:.2f} log-likelihood units "
              f"(lower is better fit).")
    else:
        print(f"  Poisson still fits as well or better -- overdispersion may not be "
              f"the issue, or sample is too small to tell.")

    print("\nCalibration comparison (Poisson lambda-bucket vs NB, same lambda):")
    print(f"{'Bucket':<10}{'N':>5}{'PoissonPred':>13}{'NB_Pred':>10}{'ActualFreq':>12}")
    for row in bucket_calibration(pairs, alpha, line_lookup):
        print(f"{row['bucket']:<10}{row['n']:>5}{row['poisson_avg_pred']:>13.3f}"
              f"{row['nb_avg_pred']:>10.3f}{row['actual_freq']:>12.3f}")

    with open(args.out, "w") as f:
        json.dump({
            "alpha": alpha,
            "n_games": len(pairs),
            "nll_poisson": nll_poisson,
            "nll_nb": nll_nb,
            "note": "variance = lambda + alpha * lambda^2. Paste `alpha` into "
                    "k_model.py's K_DISPERSION constant.",
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
