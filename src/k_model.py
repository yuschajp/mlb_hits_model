"""
k_model.py

Predicts P(pitcher records Over X strikeouts).

    lambda = blended_k_per_9 * opponent_k_rate_factor * expected_innings / 9
    P(over line) = 1 - sum(NB or Poisson PMF for k=0 to floor(line))

Three inputs blended via empirical-Bayes shrinkage (same philosophy as
hit_model.py and hr_model.py):

  1. Season K/9 -- stabilized rate, most data but reflects early-season
     rust, lineup changes, and injury recovery
  2. Recent K/9 (last 5 starts) -- captures current stuff/form, less
     stable but more timely
  3. Opponent K rate -- some lineups swing and miss far more than others;
     adjusts the expected lambda up or down

Expected innings is the fourth input -- the ceiling on how many Ks a
pitcher can accumulate. Uses the starter's season average innings/start,
shrunk toward 5.5 (the rough modern-era average for starters who take
the mound).

--- Distribution: Poisson vs Negative Binomial ---

Originally this used a pure Poisson model, on the reasoning that each
batter faced is roughly an independent trial with constant K probability.
That's a fine approximation for lambda itself, but rolling calibration
(68 graded games, ledger since 2026-06-27) showed a specific pattern:

    PredRange  N  AvgPredicted  ActualFrequency
      0.0-0.2 22          0.14             0.05
      0.2-0.4 32          0.27             0.22
      0.4-0.6 14          0.48             0.50

Bias concentrated in the low bucket, roughly fine in the middle -- that's
the signature of overdispersion: start-to-start variance in Ks (command
variance, bullpen hooks, umpire zone, matchup-specific whiff rates) is
larger than Poisson's variance=mean assumption allows for. Poisson
underweights the chance of a real blowout/bust game and compensates by
smearing probability mass toward the middle, which inflates P(over) for
pitchers whose true lambda is low.

The fix: Negative Binomial with the same mean (lambda) but a fitted
dispersion parameter alpha, where variance = lambda + alpha * lambda^2.
alpha=0 recovers Poisson exactly, so this is a strict generalization, not
a different model family.

K_DISPERSION below defaults to 0.0 (pure Poisson, original behavior) so
nothing changes until you set it. Fit alpha from graded games with
fit_k_dispersion.py, then paste the result in here:

    python3 fit_k_dispersion.py data/ledger/k_predictions_log.csv

Re-fit periodically (e.g. every ~50 new graded games) as the sample grows
and as roster/rule changes shift the true variance.
"""

import math

LEAGUE_AVG_K_PER_9 = 8.9          # rough 2024-era MLB average for starters
LEAGUE_AVG_INNINGS = 5.5           # average innings per start, modern era
LEAGUE_AVG_OPP_K_RATE = 0.227      # league-average team K rate (K/PA)

# Negative Binomial dispersion parameter for the strikeout count distribution.
# variance = lambda + K_DISPERSION * lambda^2
# 0.0 = pure Poisson (original behavior, variance == mean).
# Fit this from graded games with fit_k_dispersion.py -- do not guess a
# value by hand, the MLE fit is what makes this correction trustworthy.
K_DISPERSION = 0.0


def stabilized_k_per_9(ks, innings_pitched, prior_innings=50.0):
    """
    Empirical-Bayes shrinkage of observed K/9 toward league average.
    prior_innings=50 means a pitcher needs ~50 innings before we fully
    trust their observed K rate -- less than hits (prior_ab=200) because
    K rate stabilizes faster than batting average.
    """
    if innings_pitched < 0 or ks < 0:
        raise ValueError("ks and innings_pitched must be non-negative")
    if innings_pitched == 0:
        return LEAGUE_AVG_K_PER_9

    observed_k_per_9 = (ks / innings_pitched) * 9
    weight = innings_pitched / (innings_pitched + prior_innings)
    return weight * observed_k_per_9 + (1 - weight) * LEAGUE_AVG_K_PER_9


def blended_k_per_9(season, recent, weights=(0.60, 0.40)):
    """
    Blend season and recent K/9 rates, each shrunk toward league average.

    season: (ks, innings_pitched) for the full season
    recent: (ks, innings_pitched) for the last 5 starts (~30 days)
    weights: (season_weight, recent_weight), must sum to 1
    """
    w_season, w_recent = weights
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")

    r_season = stabilized_k_per_9(*season, prior_innings=50.0)
    r_recent = stabilized_k_per_9(*recent, prior_innings=20.0)

    return w_season * r_season + w_recent * r_recent


def opponent_k_rate_factor(opp_team_k_rate):
    """
    Adjusts expected Ks based on how much the opposing lineup strikes out.

    opp_team_k_rate: team's K rate (K per plate appearance) this season.
    Returns a multiplicative factor centered at 1.0 for a league-average
    lineup, clipped to ±30% to prevent extreme adjustments from small
    samples.

    Example: a lineup with 28% K rate vs league-average 22.7% gets a
    factor of 0.28/0.227 = 1.23 -- the pitcher should expect ~23% more
    strikeouts than against an average lineup.
    """
    if opp_team_k_rate <= 0:
        return 1.0
    factor = opp_team_k_rate / LEAGUE_AVG_OPP_K_RATE
    return max(0.70, min(1.30, factor))


def stabilized_innings(avg_innings_per_start, n_starts, prior_starts=10.0):
    """
    Shrinks a pitcher's observed average innings/start toward the league
    average. A pitcher with only 3 starts shouldn't be extrapolated from.
    """
    if n_starts <= 0:
        return LEAGUE_AVG_INNINGS
    weight = n_starts / (n_starts + prior_starts)
    return weight * avg_innings_per_start + (1 - weight) * LEAGUE_AVG_INNINGS


def _poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _nb_pmf(k, mu, alpha):
    """
    Negative Binomial PMF parameterized by mean (mu) and dispersion
    (alpha), where variance = mu + alpha * mu^2. Falls back to exact
    Poisson when alpha is ~0, so this is safe to call unconditionally.
    """
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    if alpha <= 1e-8:
        return _poisson_pmf(k, mu)
    r = 1.0 / alpha
    p = r / (r + mu)
    log_pmf = (
        math.lgamma(k + r)
        - math.lgamma(r)
        - math.lgamma(k + 1)
        + r * math.log(p)
        + k * math.log(1 - p)
    )
    return math.exp(log_pmf)


def _count_pmf(k, lam, dispersion):
    """Single switch point for the count distribution used everywhere
    below -- change K_DISPERSION, not this function, unless you're
    testing a different distribution family entirely."""
    return _nb_pmf(k, lam, dispersion)


def k_over_probability(season, recent, opp_k_rate,
                        avg_innings, n_starts, line,
                        weights=(0.60, 0.40), dispersion=None):
    """
    Full pipeline: blended K/9 → opponent adjustment → expected innings
    → lambda → P(over line) via Negative Binomial (or Poisson if
    dispersion is 0).

    season: (ks, innings_pitched) season-to-date
    recent: (ks, innings_pitched) last ~5 starts
    opp_k_rate: opposing team's K rate (K/PA) this season
    avg_innings: pitcher's average innings per start this season
    n_starts: number of starts made this season (for innings stabilization)
    line: the sportsbook's strikeout total (e.g. 6.5, 7.5)
    dispersion: NB dispersion parameter. Defaults to module-level
        K_DISPERSION if not passed explicitly -- pass this in only if you
        want to override it for a specific call (e.g. A/B testing a new
        fit before committing it to the module constant).

    Returns (p_over, p_under, expected_ks)
    """
    if dispersion is None:
        dispersion = K_DISPERSION

    base_k_per_9 = blended_k_per_9(season, recent, weights)
    opp_factor    = opponent_k_rate_factor(opp_k_rate)
    adjusted_k9   = base_k_per_9 * opp_factor
    exp_innings   = stabilized_innings(avg_innings, n_starts)

    # Expected Ks = K/9 rate × innings / 9 innings-per-game
    lam = adjusted_k9 * exp_innings / 9.0
    lam = max(0.5, min(15.0, lam))  # sanity bounds

    # P(over line) = 1 - P(<=floor(line))
    threshold = math.floor(line)
    p_under_or_eq = sum(_count_pmf(k, lam, dispersion) for k in range(threshold + 1))
    p_over  = 1 - p_under_or_eq
    p_under = p_under_or_eq

    return round(p_over, 4), round(p_under, 4), round(lam, 3)


def expected_k_distribution(lam, max_k=20, dispersion=None):
    """
    Full probability distribution of K totals for a given lambda.
    Useful for pricing any Over/Under line from a single model run.
    Returns {k: probability} for k in 0..max_k.
    """
    if dispersion is None:
        dispersion = K_DISPERSION
    return {k: round(_count_pmf(k, lam, dispersion), 5) for k in range(max_k + 1)}
