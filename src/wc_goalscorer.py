"""
wc_goalscorer.py

Predicts P(player scores at least 1 goal) in today's World Cup matches
using a Poisson model fitted on tournament goalscorer data from
openfootball/worldcup.json.

Model:
    lambda = shrunk_goals_per_90 * expected_minutes / 90 * opp_defense_factor
    P(scores) = 1 - exp(-lambda)   [Poisson P(>=1)]

Data source: openfootball worldcup.json (already used for match predictions)
-- goalscorer names and minutes are embedded in each completed match.

--- Bug fix: minutes tracked per goal instead of per match appearance ---

The original version incremented `minutes` by EXPECTED_MINUTES every time
a goal was recorded, inside the per-goal loop. That meant minutes was
always exactly `goals * EXPECTED_MINUTES` for every player -- so
`observed = (goals / minutes) * 90` reduced to a constant
(90 / EXPECTED_MINUTES) for every player who had scored at least once,
completely independent of how many goals they'd actually scored. This
was caught because it produced identical lambda/P(scores) values across
unrelated players (defenders and attackers alike) who happened to share a
goal count -- e.g. every 1-goal scorer got the exact same probability.

It also double-counted minutes within a single match: a brace or
hat-trick added 160 or 240 minutes of "playing time" for one 90-minute
game, since the loop increments minutes once per goal event rather than
once per match.

Fix: track minutes per unique match the player is known to have appeared
in (via goals1/goals2 entries), not per goal. A hat-trick in one match
now correctly contributes one match's worth of minutes, and goals/minutes
varies properly across players instead of being a fixed identity.

Remaining limitation (unchanged from before, and NOT fixable from this
data source alone): openfootball only tells us which matches a player
scored in, not which matches they played in without scoring. A player
who scored in 1 of 5 appearances will still be scored as if they only
played 1 game, inflating their apparent rate relative to a similarly
-skilled player who happened to play more scoreless minutes. Fixing this
properly requires real appearance/lineup data per player across all
matches (scored or not), not just goalscorer events -- flagging this so
it isn't mistaken for solved. Treat P(scores) outputs as directional
until that data is sourced.

Limitations (original, still applicable):
    - Small sample (only tournament games, 2-3 per player max)
    - No position data (strikers vs defenders treated equally)
    - Expected minutes defaulted to 80 (assumes starter)
    - No xG or shot data -- pure goals-per-game rate

This is intentionally simple -- the value comes from the odds comparison,
not model sophistication.
"""

import math
from collections import defaultdict

LEAGUE_AVG_GOALS_PER_90 = 0.25   # rough tournament average across all outfield players
PRIOR_MINUTES           = 180.0  # ~2 full games before we trust observed rate
EXPECTED_MINUTES        = 80.0   # assumed playing time for a starter


def extract_goalscorers(completed_matches):
    """
    Extracts player goal counts and minutes played from completed matches.

    completed_matches: list of match dicts from wc_data_client.get_all_completed_matches()
    but we need the raw data with goals1/goals2 arrays.

    Minutes are tracked per unique match appearance (via a match key, not
    goal count) so a multi-goal game only contributes one match's worth
    of minutes -- see module docstring for why this matters.

    Returns:
        player_stats: {player_name: {"goals": int, "minutes": float, "team": str}}
        team_defense: {team_name: {"goals_conceded": int, "matches": int}}
    """
    player_stats  = defaultdict(lambda: {"goals": 0, "minutes": 0.0, "team": ""})
    team_defense  = defaultdict(lambda: {"goals_conceded": 0, "matches": 0})
    player_matches_seen = defaultdict(set)  # name -> set of match keys already credited minutes

    for match_idx, m in enumerate(completed_matches):
        home  = m.get("home_team", "")
        away  = m.get("away_team", "")
        hg    = m.get("home_goals", 0) or 0
        ag    = m.get("away_goals", 0) or 0

        # Use a stable per-match key; fall back to index if no explicit id
        match_key = m.get("match_id", match_idx)

        # Defense ratings
        team_defense[home]["goals_conceded"] += ag
        team_defense[home]["matches"]         += 1
        team_defense[away]["goals_conceded"]  += hg
        team_defense[away]["matches"]          += 1

        # Goals1 = home scorers, goals2 = away scorers
        for scorer in m.get("goals1", []):
            name = scorer.get("name", "").strip()
            if not name:
                continue
            player_stats[name]["goals"] += 1
            player_stats[name]["team"]   = home
            # Only credit minutes once per match, no matter how many
            # goals this player scored in it (fixes brace/hat-trick
            # double-counting and the goals-cancel-out identity bug)
            if match_key not in player_matches_seen[name]:
                player_stats[name]["minutes"] += EXPECTED_MINUTES
                player_matches_seen[name].add(match_key)

        for scorer in m.get("goals2", []):
            name = scorer.get("name", "").strip()
            if not name:
                continue
            player_stats[name]["goals"] += 1
            player_stats[name]["team"]   = away
            if match_key not in player_matches_seen[name]:
                player_stats[name]["minutes"] += EXPECTED_MINUTES
                player_matches_seen[name].add(match_key)

        # Players who played but didn't score also need minutes tracked
        # We can't know who played from openfootball without lineup data,
        # so we only track confirmed scorers -- this means non-scorers
        # default to the league average prior (correct Bayesian behavior).
        # NOTE: this also means a scorer's minutes only reflect matches
        # where they scored, not all matches they appeared in -- see
        # module docstring "Remaining limitation" for why this still
        # understates minutes for players who scored in only some of
        # their appearances.

    return dict(player_stats), dict(team_defense)


def shrunk_goals_per_90(goals, minutes, prior_minutes=PRIOR_MINUTES):
    """
    Empirical-Bayes shrinkage of observed goals/90 toward league average.
    """
    if minutes <= 0:
        return LEAGUE_AVG_GOALS_PER_90
    observed = (goals / minutes) * 90
    weight   = minutes / (minutes + prior_minutes)
    return weight * observed + (1 - weight) * LEAGUE_AVG_GOALS_PER_90


def opponent_defense_factor(opp_team, team_defense):
    """
    Multiplicative factor based on how many goals the opponent has conceded.
    Weak defense (many goals allowed) → factor > 1 → more goals expected.
    """
    if opp_team not in team_defense:
        return 1.0
    d = team_defense[opp_team]
    if d["matches"] == 0:
        return 1.0
    goals_per_game = d["goals_conceded"] / d["matches"]
    # League average goals conceded per game ≈ LEAGUE_AVG_GOALS_PER_90 * 11 outfield players
    league_avg_conceded = 1.30  # tournament avg goals per team per game
    factor = goals_per_game / league_avg_conceded
    return max(0.5, min(2.0, factor))


def p_scores(goals, minutes, opp_team, team_defense,
             expected_minutes=EXPECTED_MINUTES):
    """
    Full pipeline: observed stats → shrunk rate → Poisson P(>=1 goal).
    """
    rate    = shrunk_goals_per_90(goals, minutes)
    defense = opponent_defense_factor(opp_team, team_defense)
    lam     = rate * (expected_minutes / 90) * defense
    lam     = max(0.01, min(3.0, lam))
    p_over  = 1 - math.exp(-lam)
    return round(p_over, 4), round(lam, 4)


def predict_match_scorers(home_team, away_team,
                           player_stats, team_defense,
                           min_p=0.10):
    """
    Returns predicted scorer probabilities for both teams in a match.
    Only returns players with p >= min_p to avoid cluttering output.

    Returns list of dicts sorted by p_scores descending.
    """
    results = []

    for player, stats in player_stats.items():
        team = stats.get("team", "")
        if team not in (home_team, away_team):
            continue

        opp = away_team if team == home_team else home_team
        p, lam = p_scores(
            stats["goals"], stats["minutes"],
            opp_team=opp,
            team_defense=team_defense,
        )
        if p >= min_p:
            results.append({
                "player":   player,
                "team":     team,
                "opponent": opp,
                "goals":    stats["goals"],
                "minutes":  stats["minutes"],
                "lambda":   lam,
                "p_scores": p,
            })

    # Also add "unknown" scorers at league average for teams with no scorers tracked
    for team, opp in [(home_team, away_team), (away_team, home_team)]:
        team_players = [r for r in results if r["team"] == team]
        if not team_players:
            # No scorers tracked for this team -- add a generic entry
            p, lam = p_scores(0, 0, opp_team=opp, team_defense=team_defense)
            results.append({
                "player":   f"({team} scorer)",
                "team":     team,
                "opponent": opp,
                "goals":    0,
                "minutes":  0,
                "lambda":   lam,
                "p_scores": p,
            })

    return sorted(results, key=lambda r: -r["p_scores"])
