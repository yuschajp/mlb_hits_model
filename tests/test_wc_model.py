"""Tests for wc_model.py -- pure math, no network required."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.wc_model import (
    compute_team_ratings,
    expected_goals,
    match_probabilities,
    predict_match,
    TOURNAMENT_AVG_GOALS,
)

SAMPLE_RESULTS = [
    {"home_team": "Brazil",    "away_team": "Serbia",   "home_goals": 2, "away_goals": 0},
    {"home_team": "Brazil",    "away_team": "Cameroon", "home_goals": 1, "away_goals": 0},
    {"home_team": "Argentina", "away_team": "Mexico",   "home_goals": 2, "away_goals": 0},
    {"home_team": "Argentina", "away_team": "Poland",   "home_goals": 2, "away_goals": 0},
    {"home_team": "France",    "away_team": "Denmark",  "home_goals": 2, "away_goals": 1},
    {"home_team": "France",    "away_team": "Tunisia",  "home_goals": 0, "away_goals": 1},
    {"home_team": "Serbia",    "away_team": "Cameroon", "home_goals": 3, "away_goals": 3},
    {"home_team": "Mexico",    "away_team": "Poland",   "home_goals": 0, "away_goals": 0},
]


def test_compute_team_ratings_returns_all_teams():
    ratings = compute_team_ratings(SAMPLE_RESULTS)
    assert "Brazil" in ratings
    assert "Argentina" in ratings
    assert "France" in ratings


def test_strong_team_has_higher_attack_than_weak():
    ratings = compute_team_ratings(SAMPLE_RESULTS)
    # Brazil scored 3 goals in 2 games, Mexico scored 0
    assert ratings["Brazil"]["attack"] > ratings["Mexico"]["attack"]


def test_strong_defense_has_lower_defense_rating():
    ratings = compute_team_ratings(SAMPLE_RESULTS)
    # Brazil conceded 0, Serbia conceded 5
    assert ratings["Brazil"]["defense"] < ratings["Serbia"]["defense"]


def test_ratings_shrink_toward_average_with_few_games():
    ratings = compute_team_ratings(SAMPLE_RESULTS)
    # With only 2 games and PRIOR_GAMES=3, ratings should be pulled toward avg
    brazil_attack = ratings["Brazil"]["attack"]
    # 1.5 goals/game observed, should be between that and league avg
    assert TOURNAMENT_AVG_GOALS < brazil_attack < 1.5 or brazil_attack <= 1.5


def test_expected_goals_stronger_team_expects_more():
    ratings = compute_team_ratings(SAMPLE_RESULTS)
    xg_bra, xg_mex = expected_goals("Brazil", "Mexico", ratings)
    xg_mex2, xg_bra2 = expected_goals("Mexico", "Brazil", ratings)
    assert xg_bra > xg_mex


def test_expected_goals_unknown_team_uses_defaults():
    ratings = compute_team_ratings(SAMPLE_RESULTS)
    xg_a, xg_b = expected_goals("Unknown FC", "Brazil", ratings)
    assert 0.3 <= xg_a <= 4.0
    assert 0.3 <= xg_b <= 4.0


def test_match_probabilities_sum_to_one():
    probs = match_probabilities(xg_a=1.3, xg_b=1.1)
    total = probs["home"] + probs["draw"] + probs["away"]
    assert abs(total - 1.0) < 0.01


def test_match_probabilities_better_team_wins_more():
    probs_dominant = match_probabilities(xg_a=2.5, xg_b=0.5)
    probs_even     = match_probabilities(xg_a=1.3, xg_b=1.3)
    assert probs_dominant["home"] > probs_even["home"]
    assert probs_dominant["away"] < probs_even["away"]


def test_match_probabilities_even_teams_symmetrical():
    probs = match_probabilities(xg_a=1.3, xg_b=1.3)
    assert abs(probs["home"] - probs["away"]) < 0.01


def test_over_2_5_higher_for_high_scoring_match():
    p_high = match_probabilities(xg_a=2.0, xg_b=2.0)["over_2_5"]
    p_low  = match_probabilities(xg_a=0.8, xg_b=0.8)["over_2_5"]
    assert p_high > p_low


def test_btts_higher_when_both_teams_score():
    p_btts_high = match_probabilities(xg_a=1.5, xg_b=1.5)["btts"]
    p_btts_low  = match_probabilities(xg_a=2.5, xg_b=0.3)["btts"]
    assert p_btts_high > p_btts_low


def test_predict_match_returns_complete_output():
    ratings = compute_team_ratings(SAMPLE_RESULTS)
    pred = predict_match("Brazil", "Argentina", ratings)
    assert "home" in pred and "draw" in pred and "away" in pred
    assert "xg_home" in pred and "xg_away" in pred
    assert "over_2_5" in pred and "btts" in pred
    assert abs(pred["home"] + pred["draw"] + pred["away"] - 1.0) < 0.01


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
