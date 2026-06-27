"""Tests for k_model.py -- pure math, no network required."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.k_model import (
    LEAGUE_AVG_K_PER_9,
    blended_k_per_9,
    expected_k_distribution,
    k_over_probability,
    opponent_k_rate_factor,
    stabilized_innings,
    stabilized_k_per_9,
)


def test_stabilized_k_per_9_pulls_small_samples_toward_league_average():
    # 5 Ks in 10 IP = 4.5 K/9, should be pulled strongly toward 8.9
    r = stabilized_k_per_9(ks=5, innings_pitched=10.0, prior_innings=50.0)
    assert abs(r - LEAGUE_AVG_K_PER_9) < 2.0


def test_stabilized_k_per_9_trusts_large_samples():
    # 200 Ks in 180 IP = 10 K/9, large sample should stay close to that
    r = stabilized_k_per_9(ks=200, innings_pitched=180.0, prior_innings=50.0)
    assert abs(r - 10.0) < 1.0


def test_stabilized_k_per_9_zero_innings_returns_league_average():
    r = stabilized_k_per_9(ks=0, innings_pitched=0.0)
    assert r == LEAGUE_AVG_K_PER_9


def test_stabilized_k_per_9_rejects_invalid():
    try:
        stabilized_k_per_9(ks=-1, innings_pitched=10.0)
        assert False
    except ValueError:
        pass


def test_blended_k_per_9_weights_must_sum_to_one():
    try:
        blended_k_per_9(season=(100, 100), recent=(20, 20), weights=(0.5, 0.6))
        assert False
    except ValueError:
        pass


def test_opponent_k_rate_factor_high_strikeout_lineup_boosts_lambda():
    factor_high = opponent_k_rate_factor(0.28)  # high K lineup
    factor_avg  = opponent_k_rate_factor(0.227)  # league average
    factor_low  = opponent_k_rate_factor(0.18)   # contact lineup
    assert factor_high > factor_avg > factor_low


def test_opponent_k_rate_factor_is_clipped():
    factor_extreme_high = opponent_k_rate_factor(0.50)
    factor_extreme_low  = opponent_k_rate_factor(0.05)
    assert factor_extreme_high <= 1.30
    assert factor_extreme_low  >= 0.70


def test_k_over_probability_higher_line_has_lower_p_over():
    common = dict(
        season=(150, 150), recent=(25, 25),
        opp_k_rate=0.227, avg_innings=6.0, n_starts=15,
    )
    p_over_5, _, _ = k_over_probability(line=5.5, **common)
    p_over_7, _, _ = k_over_probability(line=7.5, **common)
    p_over_9, _, _ = k_over_probability(line=9.5, **common)
    assert p_over_5 > p_over_7 > p_over_9


def test_k_over_probability_returns_valid_probabilities():
    p_over, p_under, lam = k_over_probability(
        season=(150, 150), recent=(25, 25),
        opp_k_rate=0.227, avg_innings=6.0, n_starts=15,
        line=6.5,
    )
    assert 0 < p_over < 1
    assert 0 < p_under < 1
    assert abs(p_over + p_under - 1.0) < 1e-6
    assert lam > 0


def test_k_over_probability_high_k_pitcher_favors_over():
    # High K pitcher: 12 K/9 over 6 innings vs average lineup
    # Expected Ks ≈ 7.0 -- should favor Over 6.5 (p_over > 50%)
    p_over, _, lam = k_over_probability(
        season=(180, 135), recent=(30, 22.5),  # ~12 K/9
        opp_k_rate=0.227, avg_innings=6.0, n_starts=15,
        line=6.5,
    )
    assert p_over > 0.50
    assert lam > 6.5


def test_k_over_probability_low_k_pitcher_favors_under():
    # Contact pitcher: 5 K/9 in 6 innings -- expected ~3.3 Ks
    p_over, _, lam = k_over_probability(
        season=(60, 108), recent=(8, 14.4),  # ~5 K/9
        opp_k_rate=0.227, avg_innings=6.0, n_starts=15,
        line=6.5,
    )
    assert p_over < 0.20
    assert lam < 6.5


def test_expected_k_distribution_sums_to_one():
    dist = expected_k_distribution(lam=7.0)
    total = sum(dist.values())
    assert abs(total - 1.0) < 0.001


def test_stabilized_innings_shrinks_small_samples():
    ip = stabilized_innings(avg_innings_per_start=4.0, n_starts=2)
    assert ip > 4.0  # pulled toward league average of 5.5


def test_stabilized_innings_trusts_large_samples():
    ip = stabilized_innings(avg_innings_per_start=7.0, n_starts=30)
    assert abs(ip - 7.0) < 0.5


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
