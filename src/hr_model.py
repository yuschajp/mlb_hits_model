"""
hr_model.py

Model for P(batter hits at least one home run in today's game).

Home runs are rare enough per at-bat (roughly 1 in 30-35, vs. roughly 1 in
4 for any hit) that a Poisson model is both simpler and more appropriate
than the binomial approach used in hit_model.py:

    lambda = adjusted_hr_rate * expected_AB
    P(>=1 HR) = 1 - e^(-lambda)

This also sidesteps the rounding-to-integer-AB inconsistency that showed
up in hit_model.py's over/under extension -- Poisson's lambda is
continuous, so there's no separate "rounded" code path needed; the same
formula extends cleanly to any home-run-count threshold (see
hr_over_under_probability() below), and "at least 1" is mathematically
guaranteed to agree with the over/under version at line=0.5 because they
share the same underlying formula rather than two different ones.

Same core methodology as hit_model.py otherwise: blend season, recent,
and platoon-split rates (each shrunk toward league average), adjust for
the opposing pitcher and the ballpark, then convert to a per-game
probability via expected at-bats for the lineup slot.
"""

import math

LEAGUE_AVG_HR_RATE = 0.032  # rough modern-era MLB home runs per at-bat -- refresh yearly

EXPECTED_AB_BY_SLOT = {
    1: 4.3, 2: 4.2, 3: 4.1, 4: 4.0, 5: 3.9,
    6: 3.8, 7: 3.7, 8: 3.6, 9: 3.5,
}


def stabilized_hr_rate(homers, at_bats, league_rate=LEAGUE_AVG_HR_RATE, prior_ab=400):
    """
    Empirical-Bayes shrinkage of an observed HR rate toward league average.
    prior_ab is much larger than hit_model.py's equivalent (200) because HR
    rate is a rarer, noisier signal -- a guy who's gone deep twice in 15
    at-bats is not actually a 13% HR hitter, and needs a stronger pull
    toward league average than a hot batting-average stretch would.
    """
    if at_bats < 0 or homers < 0:
        raise ValueError("homers and at_bats must be non-negative")
    if homers > at_bats:
        raise ValueError("homers cannot exceed at_bats")
    return (homers + league_rate * prior_ab) / (at_bats + prior_ab)


def blended_hr_rate(season, recent, vs_hand, weights=(0.5, 0.3, 0.2)):
    """Combine season/recent/platoon-split HR rates, each shrunk with its own prior_ab."""
    w_season, w_recent, w_split = weights
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")

    r_season = stabilized_hr_rate(*season, prior_ab=400)
    r_recent = stabilized_hr_rate(*recent, prior_ab=150)
    r_split = stabilized_hr_rate(*vs_hand, prior_ab=300)

    return w_season * r_season + w_recent * r_recent + w_split * r_split


def pitcher_hr_adjustment(pitcher_hr_allowed, pitcher_ab_faced, league_rate=LEAGUE_AVG_HR_RATE):
    """
    Ratio of the opposing starter's (stabilized) HR-rate-allowed to league
    average. Clipped wider than hit_model.py's pitcher adjustment (0.5-1.8
    vs 0.7-1.3) since flyball vs. groundball pitching style genuinely
    creates bigger, more real differences in HR rate allowed than it does
    for plain batting average against.
    """
    rate_allowed = stabilized_hr_rate(pitcher_hr_allowed, pitcher_ab_faced, league_rate=league_rate, prior_ab=300)
    ratio = rate_allowed / league_rate
    return max(0.5, min(1.8, ratio))


def _poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def hr_probability(season, recent, vs_hand, lineup_spot,
                    pitcher_hr_allowed, pitcher_ab_faced,
                    park_factor=1.0, weights=(0.5, 0.3, 0.2)):
    """
    Full pipeline: blended/shrunk HR rate -> pitcher adjustment -> park
    adjustment -> per-game probability via Poisson with expected at-bats
    for the lineup slot. Returns (p_hr, adjusted_rate, expected_ab).
    """
    base_rate = blended_hr_rate(season, recent, vs_hand, weights)
    adj = pitcher_hr_adjustment(pitcher_hr_allowed, pitcher_ab_faced)
    adjusted_rate = base_rate * adj * park_factor
    adjusted_rate = max(0.005, min(0.20, adjusted_rate))  # sanity bounds

    expected_ab = EXPECTED_AB_BY_SLOT.get(lineup_spot, 3.8)
    lam = adjusted_rate * expected_ab
    p_hr = 1 - math.exp(-lam)
    return p_hr, adjusted_rate, expected_ab


def hr_over_under_probability(adjusted_rate, expected_ab, line):
    """
    For a standard sportsbook home-run-total line (a half-integer like
    0.5, 1.5), returns (p_over, p_under) using the same Poisson formula as
    hr_probability() -- at line=0.5 this is mathematically guaranteed to
    match hr_probability()'s own output exactly, unlike the analogous
    situation in hit_model.py, since there's no integer-rounding step here.
    """
    lam = adjusted_rate * expected_ab
    threshold = math.floor(line) + 1  # e.g. Over 1.5 means >=2 HR
    p_under_or_equal = sum(_poisson_pmf(k, lam) for k in range(threshold))
    p_over = 1 - p_under_or_equal
    return p_over, 1 - p_over
