"""Tests for tennis_model.py -- pure math, no network required."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.tennis_model import (
    INITIAL_ELO,
    expected_score,
    k_factor,
    update_elo,
    compute_elo_ratings,
    blended_elo,
    match_win_probability,
    predict_match,
)

SAMPLE_MATCHES = [
    {"date": "20260601", "winner": "Djokovic", "loser": "Nadal",   "surface": "Hard"},
    {"date": "20260605", "winner": "Djokovic", "loser": "Federer", "surface": "Grass"},
    {"date": "20260610", "winner": "Alcaraz",  "loser": "Djokovic","surface": "Clay"},
    {"date": "20260615", "winner": "Alcaraz",  "loser": "Nadal",   "surface": "Grass"},
    {"date": "20260620", "winner": "Alcaraz",  "loser": "Federer", "surface": "Grass"},
    {"date": "20260625", "winner": "Federer",  "loser": "Nadal",   "surface": "Grass"},
]


def test_expected_score_equal_elo_is_fifty_fifty():
    p = expected_score(1500, 1500)
    assert abs(p - 0.5) < 0.001


def test_expected_score_higher_elo_favored():
    p_high = expected_score(1700, 1500)
    p_low  = expected_score(1500, 1700)
    assert p_high > 0.5
    assert p_low < 0.5
    assert abs(p_high + p_low - 1.0) < 0.001


def test_k_factor_decreases_with_experience():
    k_new = k_factor(0)
    k_experienced = k_factor(100)
    assert k_new > k_experienced


def test_k_factor_never_below_minimum():
    k = k_factor(1000)
    assert k >= 10.0 - 0.01


def test_update_elo_winner_gains_rating():
    new_elo = update_elo(1500, 1500, matches_a=10, actual_a=1.0)
    assert new_elo > 1500


def test_update_elo_loser_loses_rating():
    new_elo = update_elo(1500, 1500, matches_a=10, actual_a=0.0)
    assert new_elo < 1500


def test_update_elo_upset_win_gains_more():
    # Beating a much higher-rated opponent should gain more Elo
    # than beating an equal opponent
    gain_vs_equal  = update_elo(1500, 1500, matches_a=10, actual_a=1.0) - 1500
    gain_vs_better = update_elo(1500, 1700, matches_a=10, actual_a=1.0) - 1500
    assert gain_vs_better > gain_vs_equal


def test_compute_elo_ratings_returns_all_players():
    overall, surface, counts, surf_counts = compute_elo_ratings(SAMPLE_MATCHES, surface="Grass")
    assert "Djokovic" in overall
    assert "Alcaraz" in overall
    assert "Nadal" in overall
    assert "Federer" in overall


def test_compute_elo_ratings_winner_has_higher_overall_elo():
    overall, _, _, _ = compute_elo_ratings(SAMPLE_MATCHES, surface="Grass")
    # Alcaraz won 3 matches including beating Djokovic -- should be highly rated
    assert overall["Alcaraz"] > INITIAL_ELO


def test_compute_elo_ratings_surface_filter_only_counts_grass():
    _, _, _, surf_counts = compute_elo_ratings(SAMPLE_MATCHES, surface="Grass")
    # 4 grass matches in sample: Djok-Fed, Alcaraz-Nadal, Alcaraz-Fed, Fed-Nadal
    assert surf_counts.get("Federer", 0) == 3  # played in 3 grass matches
    assert surf_counts.get("Djokovic", 0) == 1  # only 1 grass match


def test_blended_elo_no_surface_history_uses_overall():
    overall = {"PlayerX": 1600}
    surface = {"PlayerX": 1500}
    surf_counts = {}  # no surface matches
    blend = blended_elo("PlayerX", overall, surface, surf_counts)
    assert abs(blend - 1600) < 1.0


def test_blended_elo_full_surface_history_uses_surface():
    overall = {"PlayerX": 1600}
    surface = {"PlayerX": 1500}
    surf_counts = {"PlayerX": 50}  # well beyond full_weight_matches
    blend = blended_elo("PlayerX", overall, surface, surf_counts)
    assert abs(blend - 1500) < 1.0


def test_blended_elo_partial_history_interpolates():
    overall = {"PlayerX": 1600}
    surface = {"PlayerX": 1500}
    surf_counts = {"PlayerX": 10}  # half of full_weight_matches (20)
    blend = blended_elo("PlayerX", overall, surface, surf_counts)
    assert 1500 < blend < 1600


def test_match_win_probability_matches_expected_score():
    p1 = match_win_probability(1600, 1500)
    p2 = expected_score(1600, 1500)
    assert p1 == round(p2, 4)


def test_predict_match_returns_valid_probabilities():
    overall, surface, counts, surf_counts = compute_elo_ratings(SAMPLE_MATCHES, surface="Grass")
    pred = predict_match("Alcaraz", "Nadal", overall, surface, surf_counts)
    assert 0 < pred["p_a_wins"] < 1
    assert abs(pred["p_a_wins"] + pred["p_b_wins"] - 1.0) < 0.001


def test_predict_match_unrated_players_default_to_fifty_fifty():
    pred = predict_match("Unknown A", "Unknown B", {}, {}, {})
    assert abs(pred["p_a_wins"] - 0.5) < 0.01


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
