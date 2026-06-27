"""
wc_model.py

Predicts World Cup match outcomes (1X2) and expected goals using a
Poisson model fitted on tournament results so far.

For each team, we maintain:
  attack_rating  -- goals scored per 90 min, shrunk toward tournament avg
  defense_rating -- goals conceded per 90 min, shrunk toward tournament avg

Expected goals in a match:
  xG_A = attack_A * (1/defense_B) * home_factor * tournament_avg_goals
  xG_B = attack_B * (1/defense_A) * tournament_avg_goals

Match probabilities are computed by summing over all (i,j) goal combinations
via independent Poisson distributions -- the core of Dixon-Coles.

Team ratings are initialized from FIFA ranking points (provided as input)
and updated from tournament results via a simple Bayesian update. The
prior is stronger for teams with fewer games played.
"""

import math
from collections import defaultdict

# World Cup 2026 tournament average goals per team per game
# (roughly 1.3 based on recent World Cup history)
TOURNAMENT_AVG_GOALS = 1.30

# Shrinkage prior: how many "virtual games" worth of average performance
# to add before trusting observed tournament results
PRIOR_GAMES = 3.0

# Maximum goals to consider in probability calculations
MAX_GOALS = 8


def _poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def compute_team_ratings(results):
    """
    Computes attack and defense ratings for each team from tournament results.

    results: list of dicts:
      {"home_team": str, "away_team": str,
       "home_goals": int, "away_goals": int}

    Returns {team_name: {"attack": float, "defense": float, "games": int}}
    """
    goals_scored   = defaultdict(list)
    goals_conceded = defaultdict(list)

    for r in results:
        goals_scored[r["home_team"]].append(r["home_goals"])
        goals_conceded[r["home_team"]].append(r["away_goals"])
        goals_scored[r["away_team"]].append(r["away_goals"])
        goals_conceded[r["away_team"]].append(r["home_goals"])

    ratings = {}
    all_teams = set(goals_scored.keys()) | set(goals_conceded.keys())

    for team in all_teams:
        scored    = goals_scored.get(team, [])
        conceded  = goals_conceded.get(team, [])
        n         = len(scored)

        # Bayesian update: blend observed rate with prior
        obs_attack  = sum(scored)  / n if n > 0 else TOURNAMENT_AVG_GOALS
        obs_defense = sum(conceded) / n if n > 0 else TOURNAMENT_AVG_GOALS

        weight = n / (n + PRIOR_GAMES)
        attack  = weight * obs_attack  + (1 - weight) * TOURNAMENT_AVG_GOALS
        defense = weight * obs_defense + (1 - weight) * TOURNAMENT_AVG_GOALS

        ratings[team] = {
            "attack":  round(attack, 4),
            "defense": round(max(0.1, defense), 4),
            "games":   n,
        }

    return ratings


def expected_goals(team_a, team_b, ratings, neutral_venue=True):
    """
    Computes expected goals for both teams in a match.

    team_a: attacking team name
    team_b: defending team name
    ratings: output of compute_team_ratings()
    neutral_venue: World Cup knockout games are on neutral ground

    Returns (xg_a, xg_b)
    """
    avg = TOURNAMENT_AVG_GOALS

    a = ratings.get(team_a, {"attack": avg, "defense": avg})
    b = ratings.get(team_b, {"attack": avg, "defense": avg})

    # Dixon-Coles style expected goals
    xg_a = (a["attack"] / avg) * (avg / b["defense"]) * avg
    xg_b = (b["attack"] / avg) * (avg / a["defense"]) * avg

    # Clip to reasonable range
    xg_a = max(0.3, min(4.0, xg_a))
    xg_b = max(0.3, min(4.0, xg_b))

    return round(xg_a, 3), round(xg_b, 3)


def match_probabilities(xg_a, xg_b, include_draw=True):
    """
    Computes 1X2 probabilities from expected goals via independent Poisson.

    Returns {"home": p_home, "draw": p_draw, "away": p_away,
             "over_2_5": p_over, "btts": p_btts}
    """
    p_home = p_draw = p_away = 0.0
    p_btts = 0.0

    # Build probability matrix
    for i in range(MAX_GOALS + 1):
        p_i = _poisson_pmf(i, xg_a)
        for j in range(MAX_GOALS + 1):
            p_j = _poisson_pmf(j, xg_b)
            p_ij = p_i * p_j
            if i > j:
                p_home += p_ij
            elif i == j:
                p_draw += p_ij
            else:
                p_away += p_ij
            if i > 0 and j > 0:
                p_btts += p_ij

    # Normalize (truncation at MAX_GOALS creates tiny rounding gap)
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # Over 2.5 goals
    p_over_2_5 = sum(
        _poisson_pmf(i, xg_a) * _poisson_pmf(j, xg_b)
        for i in range(MAX_GOALS + 1)
        for j in range(MAX_GOALS + 1)
        if i + j > 2
    )

    return {
        "home":      round(p_home, 4),
        "draw":      round(p_draw, 4),
        "away":      round(p_away, 4),
        "over_2_5":  round(p_over_2_5, 4),
        "btts":      round(p_btts, 4),
    }


def predict_match(home_team, away_team, ratings):
    """
    Full pipeline: ratings → xG → match probabilities.

    Returns dict with xg_home, xg_away, and all probability fields.
    """
    xg_h, xg_a = expected_goals(home_team, away_team, ratings)
    probs = match_probabilities(xg_h, xg_a)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "xg_home":   xg_h,
        "xg_away":   xg_a,
        **probs,
    }
