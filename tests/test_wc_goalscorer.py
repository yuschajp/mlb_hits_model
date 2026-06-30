"""Tests for wc_goalscorer.py -- pure math, no network required."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.wc_goalscorer import (
    extract_goalscorers,
    shrunk_goals_per_90,
    opponent_defense_factor,
    p_scores,
    predict_match_scorers,
    LEAGUE_AVG_GOALS_PER_90,
)

SAMPLE_MATCHES = [
    {
        "home_team": "Brazil", "away_team": "Serbia",
        "home_goals": 2, "away_goals": 0,
        "goals1": [{"name": "Vinicius Jr", "minute": "12"},
                   {"name": "Richarlison", "minute": "73"}],
        "goals2": [],
    },
    {
        "home_team": "Brazil", "away_team": "Cameroon",
        "home_goals": 1, "away_goals": 0,
        "goals1": [{"name": "Richarlison", "minute": "45"}],
        "goals2": [],
    },
    {
        "home_team": "Argentina", "away_team": "Mexico",
        "home_goals": 2, "away_goals": 0,
        "goals1": [{"name": "Messi", "minute": "64"},
                   {"name": "Fernandez", "minute": "87"}],
        "goals2": [],
    },
    {
        "home_team": "France", "away_team": "Denmark",
        "home_goals": 2, "away_goals": 1,
        "goals1": [{"name": "Mbappe", "minute": "61"},
                   {"name": "Mbappe", "minute": "86"}],
        "goals2": [{"name": "Christensen", "minute": "68"}],
    },
]


def test_extract_goalscorers_counts_goals():
    stats, _ = extract_goalscorers(SAMPLE_MATCHES)
    assert stats["Richarlison"]["goals"] == 2
    assert stats["Mbappe"]["goals"] == 2
    assert stats["Messi"]["goals"] == 1


def test_extract_goalscorers_assigns_teams():
    stats, _ = extract_goalscorers(SAMPLE_MATCHES)
    assert stats["Richarlison"]["team"] == "Brazil"
    assert stats["Mbappe"]["team"] == "France"
    assert stats["Messi"]["team"] == "Argentina"


def test_extract_defense_tracks_conceded():
    _, defense = extract_goalscorers(SAMPLE_MATCHES)
    # Serbia conceded 2 goals to Brazil
    assert defense["Serbia"]["goals_conceded"] == 2
    # Brazil conceded 0 in two games
    assert defense["Brazil"]["goals_conceded"] == 0


def test_shrunk_goals_per_90_pulls_small_samples():
    # 1 goal in 80 minutes = 1.125 per 90 -- should shrink toward avg
    rate = shrunk_goals_per_90(goals=1, minutes=80.0)
    assert LEAGUE_AVG_GOALS_PER_90 < rate < 1.125


def test_shrunk_goals_per_90_zero_minutes_returns_avg():
    rate = shrunk_goals_per_90(goals=0, minutes=0.0)
    assert rate == LEAGUE_AVG_GOALS_PER_90


def test_shrunk_goals_per_90_trusts_large_samples():
    # 10 goals in 900 minutes = 1.0 per 90, with prior=180 weight=900/1080=0.833
    # result ≈ 0.833*1.0 + 0.167*0.25 = 0.875 -- closer to observed than to prior
    rate = shrunk_goals_per_90(goals=10, minutes=900.0)
    assert rate > LEAGUE_AVG_GOALS_PER_90  # pulled toward observation
    assert rate < 1.0                       # but still some shrinkage


def test_opponent_defense_factor_weak_defense_boosts():
    _, defense = extract_goalscorers(SAMPLE_MATCHES)
    # Serbia conceded 2 in 1 game = weak defense, factor should be > 1
    factor = opponent_defense_factor("Serbia", defense)
    assert factor > 1.0


def test_opponent_defense_factor_strong_defense_reduces():
    _, defense = extract_goalscorers(SAMPLE_MATCHES)
    # Brazil conceded 0, strong defense, factor should be < 1
    factor = opponent_defense_factor("Brazil", defense)
    assert factor < 1.0


def test_opponent_defense_factor_unknown_team_returns_one():
    _, defense = extract_goalscorers(SAMPLE_MATCHES)
    factor = opponent_defense_factor("Unknown FC", defense)
    assert factor == 1.0


def test_p_scores_prolific_scorer_higher_probability():
    stats, defense = extract_goalscorers(SAMPLE_MATCHES)
    # Mbappe scored 2 goals -- should have higher P than Messi (1 goal)
    p_mbappe, _ = p_scores(stats["Mbappe"]["goals"], stats["Mbappe"]["minutes"],
                            "Serbia", defense)
    p_messi, _  = p_scores(stats["Messi"]["goals"],  stats["Messi"]["minutes"],
                            "Serbia", defense)
    assert p_mbappe > p_messi


def test_p_scores_returns_valid_probability():
    stats, defense = extract_goalscorers(SAMPLE_MATCHES)
    p, lam = p_scores(1, 80.0, "Serbia", defense)
    assert 0 < p < 1
    assert lam > 0


def test_predict_match_scorers_returns_correct_teams():
    stats, defense = extract_goalscorers(SAMPLE_MATCHES)
    preds = predict_match_scorers("Brazil", "Argentina", stats, defense)
    teams = {r["team"] for r in preds}
    assert "Brazil" in teams
    assert "Argentina" in teams


def test_predict_match_scorers_sorted_by_probability():
    stats, defense = extract_goalscorers(SAMPLE_MATCHES)
    preds = predict_match_scorers("Brazil", "France", stats, defense)
    probs = [r["p_scores"] for r in preds]
    assert probs == sorted(probs, reverse=True)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
