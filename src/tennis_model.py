"""
tennis_model.py

Predicts ATP/WTA match outcomes using surface-specific Elo ratings.

Elo update formula (standard chess-style, tuned for tennis):
    expected_a = 1 / (1 + 10^((elo_b - elo_a) / 400))
    new_elo_a  = elo_a + K * (actual_a - expected_a)

K-factor decreases as more matches are played (more stable rating),
similar in spirit to the shrinkage priors used in the MLB models --
new players' ratings move more per match than established players'.

Surface adjustment: a player's overall Elo is blended with their
surface-specific Elo, weighted by how many matches they've played on
that surface. A grass-court specialist with limited grass match history
still gets meaningful credit for their surface Elo once they've played
a handful of matches there.
"""

import math
from collections import defaultdict

INITIAL_ELO = 1500.0
BASE_K      = 32.0
MIN_K       = 10.0
K_DECAY_MATCHES = 30.0  # matches after which K stabilizes near MIN_K

SURFACE_WEIGHT_FULL = 20.0  # matches on surface before fully trusting surface Elo


def expected_score(elo_a, elo_b):
    """Standard Elo expected score formula."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def k_factor(matches_played):
    """K-factor decays as a player accumulates match history."""
    if matches_played <= 0:
        return BASE_K
    decay = math.exp(-matches_played / K_DECAY_MATCHES)
    return MIN_K + (BASE_K - MIN_K) * decay


def update_elo(elo_a, elo_b, matches_a, actual_a):
    """
    Updates player A's Elo after a match.

    actual_a: 1.0 if A won, 0.0 if A lost
    Returns new_elo_a
    """
    exp_a = expected_score(elo_a, elo_b)
    k     = k_factor(matches_a)
    return elo_a + k * (actual_a - exp_a)


def compute_elo_ratings(matches, surface=None):
    """
    Computes overall and surface-specific Elo ratings from a list of
    historical matches, processed in chronological order.

    matches: list of dicts, each with:
        {"winner": str, "loser": str, "surface": str, "date": str}

    surface: if provided, only matches on this surface contribute to
             the surface-specific rating (overall rating uses all matches
             regardless of this filter)

    Returns:
        overall_elo: {player: rating}
        surface_elo: {player: rating}  (only meaningful if surface given)
        match_counts: {player: total_matches}
        surface_match_counts: {player: matches_on_this_surface}
    """
    overall_elo = defaultdict(lambda: INITIAL_ELO)
    surface_elo = defaultdict(lambda: INITIAL_ELO)
    match_counts = defaultdict(int)
    surface_match_counts = defaultdict(int)

    # Sort chronologically
    sorted_matches = sorted(matches, key=lambda m: m.get("date", ""))

    for m in sorted_matches:
        winner = m["winner"]
        loser  = m["loser"]
        m_surface = m.get("surface", "")

        # Overall Elo update (every match counts)
        w_elo, l_elo = overall_elo[winner], overall_elo[loser]
        new_w = update_elo(w_elo, l_elo, match_counts[winner], actual_a=1.0)
        new_l = update_elo(l_elo, w_elo, match_counts[loser],  actual_a=0.0)
        overall_elo[winner] = new_w
        overall_elo[loser]  = new_l
        match_counts[winner] += 1
        match_counts[loser]  += 1

        # Surface-specific Elo update (only matches on the target surface)
        if surface and m_surface == surface:
            sw_elo, sl_elo = surface_elo[winner], surface_elo[loser]
            new_sw = update_elo(sw_elo, sl_elo, surface_match_counts[winner], actual_a=1.0)
            new_sl = update_elo(sl_elo, sw_elo, surface_match_counts[loser],  actual_a=0.0)
            surface_elo[winner] = new_sw
            surface_elo[loser]  = new_sl
            surface_match_counts[winner] += 1
            surface_match_counts[loser]  += 1

    return dict(overall_elo), dict(surface_elo), dict(match_counts), dict(surface_match_counts)


def blended_elo(player, overall_elo, surface_elo, surface_matches,
                 full_weight_matches=SURFACE_WEIGHT_FULL):
    """
    Blends overall Elo with surface-specific Elo based on how much
    surface match history the player has.

    A player with 0 surface matches gets 100% overall Elo.
    A player with full_weight_matches+ surface matches gets ~100% surface Elo.
    """
    n = surface_matches.get(player, 0)
    overall = overall_elo.get(player, INITIAL_ELO)
    surf    = surface_elo.get(player, INITIAL_ELO)

    weight = min(1.0, n / full_weight_matches)
    return weight * surf + (1 - weight) * overall


def match_win_probability(elo_a, elo_b):
    """
    P(player A wins) from Elo difference.
    Same formula as expected_score -- exposed separately for clarity
    in calling code.
    """
    return round(expected_score(elo_a, elo_b), 4)


def predict_match(player_a, player_b, overall_elo, surface_elo, surface_matches):
    """
    Full pipeline: blended Elo for both players → win probability.

    Returns dict with elo_a, elo_b, p_a_wins, p_b_wins.
    """
    elo_a = blended_elo(player_a, overall_elo, surface_elo, surface_matches)
    elo_b = blended_elo(player_b, overall_elo, surface_elo, surface_matches)

    p_a = match_win_probability(elo_a, elo_b)
    p_b = round(1 - p_a, 4)

    return {
        "player_a": player_a,
        "player_b": player_b,
        "elo_a":    round(elo_a, 1),
        "elo_b":    round(elo_b, 1),
        "p_a_wins": p_a,
        "p_b_wins": p_b,
    }
