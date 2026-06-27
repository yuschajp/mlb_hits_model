"""
k_model.py

Predicts P(pitcher records Over X strikeouts) using a Poisson model.

    lambda = blended_k_per_9 * opponent_k_rate_factor * expected_innings / 9
    P(over line) = 1 - sum(Poisson PMF for k=0 to floor(line))

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

Why Poisson works here: pitcher strikeout counts per start are well-
approximated by a Poisson process -- each batter faced is an independent
trial with a roughly constant K probability. This is more valid for Ks
than for hits (where batter quality varies more across lineup slots) and
HR (where park/weather effects are larger). The main violation is that
K rate varies batter-to-batter through the order, but the aggregate
per-game total is Poisson-distributed to a good approximation.
"""

import math

LEAGUE_AVG_K_PER_9 = 8.9          # rough 2024-era MLB average for starters
LEAGUE_AVG_INNINGS = 5.5           # average innings per start, modern era
LEAGUE_AVG_OPP_K_RATE = 0.227      # league-average team K rate (K/PA)


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


def k_over_probability(season, recent, opp_k_rate,
                        avg_innings, n_starts, line,
                        weights=(0.60, 0.40)):
    """
    Full pipeline: blended K/9 → opponent adjustment → expected innings
    → Poisson lambda → P(over line).

    season: (ks, innings_pitched) season-to-date
    recent: (ks, innings_pitched) last ~5 starts
    opp_k_rate: opposing team's K rate (K/PA) this season
    avg_innings: pitcher's average innings per start this season
    n_starts: number of starts made this season (for innings stabilization)
    line: the sportsbook's strikeout total (e.g. 6.5, 7.5)

    Returns (p_over, p_under, expected_ks)
    """
    base_k_per_9 = blended_k_per_9(season, recent, weights)
    opp_factor    = opponent_k_rate_factor(opp_k_rate)
    adjusted_k9   = base_k_per_9 * opp_factor
    exp_innings   = stabilized_innings(avg_innings, n_starts)

    # Expected Ks = K/9 rate × innings / 9 innings-per-game
    lam = adjusted_k9 * exp_innings / 9.0
    lam = max(0.5, min(15.0, lam))  # sanity bounds

    # P(over line) = 1 - P(<=floor(line))
    threshold = math.floor(line)
    p_under_or_eq = sum(_poisson_pmf(k, lam) for k in range(threshold + 1))
    p_over  = 1 - p_under_or_eq
    p_under = p_under_or_eq

    return round(p_over, 4), round(p_under, 4), round(lam, 3)


def expected_k_distribution(lam, max_k=20):
    """
    Full probability distribution of K totals for a given lambda.
    Useful for pricing any Over/Under line from a single model run.
    Returns {k: probability} for k in 0..max_k.
    """
    return {k: round(_poisson_pmf(k, lam), 5) for k in range(max_k + 1)}
