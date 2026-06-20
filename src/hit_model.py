"""
hit_model.py

Model for P(batter records at least one hit in today's game) -- the
standard "to record a hit" prop.

Core idea: estimate a "true" per-at-bat hit probability for this specific
matchup (season rate, blended with recent form and the platoon split vs.
the opposing starter's throwing hand, each shrunk toward league average so
small samples don't dominate), adjust for the opposing pitcher's quality
and the ballpark, then convert to a per-game probability using the
batter's expected number of at-bats for his lineup spot:

    P(>=1 hit) = 1 - (1 - p_adjusted) ** expected_AB

v1 simplification worth flagging: expected_AB is treated as a fixed point
estimate per lineup slot rather than a full distribution (extra innings,
early pinch-hit removal, rain delays etc. all move the real number around).
Good enough to start; refining this is on the roadmap.
"""

LEAGUE_AVG_BA = 0.245  # rough modern-era MLB league average -- refresh yearly

# Approximate at-bats per game by batting-order slot. These are typical
# league-wide averages, not team-specific. A real refinement: pull each
# team's actual historical AB-per-slot instead of one shared table.
EXPECTED_AB_BY_SLOT = {
    1: 4.3, 2: 4.2, 3: 4.1, 4: 4.0, 5: 3.9,
    6: 3.8, 7: 3.7, 8: 3.6, 9: 3.5,
}


def stabilized_rate(hits, at_bats, league_rate=LEAGUE_AVG_BA, prior_ab=200):
    """
    Empirical-Bayes shrinkage of an observed batting average toward the
    league mean. prior_ab sets how much weight the league average carries:
    a player with at_bats << prior_ab gets pulled heavily toward league
    average; at_bats >> prior_ab leaves them close to their own observed
    rate. 200 is a reasonable starting point given how slowly batting
    average actually stabilizes in reality -- tune once real data is in.
    """
    if at_bats < 0 or hits < 0:
        raise ValueError("hits and at_bats must be non-negative")
    if hits > at_bats:
        raise ValueError("hits cannot exceed at_bats")
    return (hits + league_rate * prior_ab) / (at_bats + prior_ab)


def blended_hit_rate(season, recent, vs_hand, weights=(0.5, 0.3, 0.2)):
    """
    Combine three rates into one batter-quality estimate: season-to-date,
    last-30-days form, and the platoon split vs. today's opposing starter's
    throwing hand. Each input is a (hits, at_bats) tuple; weights must sum
    to 1. Each rate is shrunk with its own prior_ab since the three signals
    stabilize at different speeds (recent form moves fastest, platoon
    splits stabilize slowest).
    """
    w_season, w_recent, w_split = weights
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")

    r_season = stabilized_rate(*season, prior_ab=200)
    r_recent = stabilized_rate(*recent, prior_ab=80)
    r_split = stabilized_rate(*vs_hand, prior_ab=150)

    return w_season * r_season + w_recent * r_recent + w_split * r_split


def pitcher_adjustment(pitcher_hits_allowed, pitcher_ab_faced, league_rate=LEAGUE_AVG_BA):
    """
    Ratio of the opposing starter's (stabilized) batting-average-against to
    league average. >1 means hitters do better than average against this
    pitcher; <1 means the pitcher suppresses hits. Clipped so a small,
    noisy pitcher sample can't swing the estimate too far in either
    direction (e.g. a starter's first two outings of the season).
    """
    baa = stabilized_rate(pitcher_hits_allowed, pitcher_ab_faced, league_rate=league_rate, prior_ab=150)
    ratio = baa / league_rate
    return max(0.7, min(1.3, ratio))


def hit_probability(season, recent, vs_hand, lineup_spot,
                     pitcher_hits_allowed, pitcher_ab_faced,
                     park_factor=1.0, weights=(0.5, 0.3, 0.2)):
    """
    Full pipeline: blended/shrunk batter rate -> pitcher adjustment -> park
    adjustment -> per-game probability via expected at-bats for the slot.

    season, recent, vs_hand: (hits, at_bats) tuples for the batter.
    lineup_spot: 1-9, today's confirmed batting order position.
    pitcher_hits_allowed, pitcher_ab_faced: the opposing starter's season
        hits-allowed and at-bats-faced (for batting-average-against).
    park_factor: multiplicative venue adjustment, 1.0 = league-neutral park.

    Returns (p_hit, adjusted_ba, expected_ab) -- the intermediate values
    are returned too so a daily report can show *why* a number is high or
    low, not just the final probability.
    """
    base_rate = blended_hit_rate(season, recent, vs_hand, weights)
    adj = pitcher_adjustment(pitcher_hits_allowed, pitcher_ab_faced)
    adjusted_ba = base_rate * adj * park_factor
    adjusted_ba = max(0.05, min(0.55, adjusted_ba))  # sanity bounds

    expected_ab = EXPECTED_AB_BY_SLOT.get(lineup_spot, 3.8)
    p_hit = 1 - (1 - adjusted_ba) ** expected_ab
    return p_hit, adjusted_ba, expected_ab
