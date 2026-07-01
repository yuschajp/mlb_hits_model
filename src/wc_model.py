"""
wc_model.py

Predicts World Cup match outcomes (1X2) and expected goals using a
Poisson model fitted on tournament results so far.

For each team, we maintain:
  attack_rating  -- goals scored per 90 min, shrunk toward tournament avg
  defense_rating -- goals conceded per 90 min, shrunk toward tournament avg
                    (LOWER = stronger defense, since it's goals *conceded*)

Expected goals in a match:
  xG_A = attack_A * (defense_B / tournament_avg_goals)
  xG_B = attack_B * (defense_A / tournament_avg_goals)

--- Bug fix: inverted defense term ---

The previous version computed:
    xg_a = attack_a * avg / defense_b

Since defense_b is "goals conceded per game" (lower = better defense),
dividing by it is backwards: it means facing a STRONGER defense (lower
defense_b) produced a HIGHER expected goals for the attacking team, and
facing a WEAKER defense (higher defense_b) suppressed it. That's the
opposite of reality.

This was caught on the Belgium vs Senegal Round of 32 prediction: Belgium
had conceded only 2 goals in 3 group games (defense=0.98, well below the
1.30 tournament average -- a strong defense), while Senegal had conceded
6 goals in 3 games including two losses (defense=1.65, a leaky defense).
The buggy formula divided Senegal's attack by Belgium's low defense
number, inflating Senegal's expected goals specifically BECAUSE Belgium's
defense was good -- and produced Away 65.7% / Home 17.1%, with Senegal
favored despite a losing group-stage record. The corrected formula
(multiplying by defense_b/avg instead of dividing by defense_b) flips
this to Home 51.3% / Away 27.6%, which matches the actual underlying
results far better.

This bug likely explains why Home Brier (0.318) had been running worse
than the naive 0.222 baseline while Away Brier (0.148) looked artificially
good on the 7 graded matches so far: home teams were being systematically
underrated whenever they had a genuinely strong defense, and away teams
were being overrated when facing one. Re-grade and recheck calibration
once more matches accumulate under the corrected formula.

Match probabilities are computed by summing over all (i,j) goal combinations
via independent Poisson distributions.

--- Note: this is NOT full Dixon-Coles, despite earlier claims ---

True Dixon-Coles adds a correlation correction term (tau) for low
scorelines (0-0, 1-0, 0-1, 1-1) to correct for the empirically-observed
fact that real match results are more correlated at low scores than
independent Poisson predicts -- mainly, more draws than independent
Poisson alone would produce. This model does NOT implement that
correction; it's plain independent bivariate Poisson. This is a likely
contributor to the known issue of draws being underweighted (Draw Brier
0.217, barely beating the 0.222 naive baseline). Fixing this properly
would mean implementing the actual tau adjustment, which is a separate,
larger change from the defense-term bug fixed here -- flagging so it
isn't mistaken for solved.

Team ratings are updated from tournament results via a simple Bayesian
shrinkage toward the tournament average. The prior is stronger for teams
with fewer games played (PRIOR_GAMES=3.0 means a team with 3 games gets
a 50/50 blend of observed rate and tournament average -- looser shrinkage
than the K/HR models use, worth revisiting if ratings look unstable
early in the tournament).
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
    neutral_venue: World Cup knockout games are on neutral ground. No
        home-field-advantage term is currently modeled either way (this
        parameter was previously accepted but silently unused -- flagging
        that explicitly here rather than pretending it does something).
        If you want home-field advantage for group-stage matches with a
        genuine host-nation edge, that needs to be added as a real
        multiplicative term, not implied by this flag.

    Returns (xg_a, xg_b)
    """
    avg = TOURNAMENT_AVG_GOALS

    a = ratings.get(team_a, {"attack": avg, "defense": avg})
    b = ratings.get(team_b, {"attack": avg, "defense": avg})

    # Expected goals = attacker's scoring rate, scaled by how the
    # opponent's defense compares to average. A defense number ABOVE
    # average (leaky defense, concedes more than average) should SCALE
    # UP the attacker's expected goals; a defense number BELOW average
    # (stingy defense) should scale it DOWN. That means multiplying by
    # (opponent_defense / avg), not dividing by opponent_defense -- see
    # module docstring for the bug this replaces and why it mattered.
    xg_a = a["attack"] * (b["defense"] / avg)
    xg_b = b["attack"] * (a["defense"] / avg)

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
