"""
Sanity tests for hit_model.py -- no network required, pure math checks.
Run with: pytest tests/  (or run directly with python3, see bottom of file)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hit_model import (
    LEAGUE_AVG_BA,
    blended_hit_rate,
    hit_probability,
    hits_count_distribution,
    over_under_probability,
    pitcher_adjustment,
    stabilized_rate,
)


def test_stabilized_rate_pulls_small_samples_toward_league_average():
    # 1-for-1 (1.000 BA) on a tiny sample should land nowhere near 1.000
    # after shrinkage -- it should sit close to the league average.
    r = stabilized_rate(hits=1, at_bats=1, prior_ab=200)
    assert abs(r - LEAGUE_AVG_BA) < 0.01


def test_stabilized_rate_trusts_large_samples():
    # A real, full-season sample should barely move from the observed rate.
    r = stabilized_rate(hits=180, at_bats=600, prior_ab=200)  # .300 hitter
    assert abs(r - 0.300) < 0.02


def test_stabilized_rate_rejects_invalid_input():
    try:
        stabilized_rate(hits=10, at_bats=5)  # more hits than at-bats: impossible
        assert False, "should have raised"
    except ValueError:
        pass


def test_blended_hit_rate_weights_must_sum_to_one():
    try:
        blended_hit_rate((50, 200), (10, 40), (15, 60), weights=(0.5, 0.5, 0.5))
        assert False, "should have raised"
    except ValueError:
        pass


def test_pitcher_adjustment_is_clipped():
    # An extremely small, extremely low-contact sample shouldn't be allowed
    # to swing the multiplier past the clip bounds.
    adj = pitcher_adjustment(pitcher_hits_allowed=0, pitcher_ab_faced=3)
    assert 0.7 <= adj <= 1.3


def test_hit_probability_monotonic_in_lineup_spot():
    # Holding everything else fixed, a leadoff hitter (more expected AB)
    # should have a higher hit probability than a #9 hitter.
    common_kwargs = dict(
        season=(80, 320), recent=(12, 45), vs_hand=(20, 75),
        pitcher_hits_allowed=140, pitcher_ab_faced=580, park_factor=1.0,
    )
    p_leadoff, _, _ = hit_probability(lineup_spot=1, **common_kwargs)
    p_ninth, _, _ = hit_probability(lineup_spot=9, **common_kwargs)
    assert p_leadoff > p_ninth


def test_hit_probability_responds_to_park_factor():
    common_kwargs = dict(
        season=(80, 320), recent=(12, 45), vs_hand=(20, 75), lineup_spot=3,
        pitcher_hits_allowed=140, pitcher_ab_faced=580,
    )
    p_coors, _, _ = hit_probability(park_factor=1.13, **common_kwargs)
    p_petco, _, _ = hit_probability(park_factor=0.92, **common_kwargs)
    assert p_coors > p_petco


def test_hit_probability_is_a_valid_probability():
    p, adjusted_ba, expected_ab = hit_probability(
        season=(80, 320), recent=(12, 45), vs_hand=(20, 75), lineup_spot=4,
        pitcher_hits_allowed=140, pitcher_ab_faced=580, park_factor=1.0,
    )
    assert 0.0 < p < 1.0
    assert 0.0 < adjusted_ba < 1.0
    assert expected_ab > 0


def test_hits_count_distribution_sums_to_one():
    dist = hits_count_distribution(adjusted_ba=0.28, expected_ab=4)
    assert abs(sum(dist) - 1.0) < 1e-9
    assert len(dist) == 5  # 0 through 4 hits


def test_over_0_5_is_close_to_hit_probability():
    # Over 0.5 hits ("at least 1") should be CLOSE to hit_probability()'s
    # own continuous-AB formula, but not bit-exact -- the binomial version
    # rounds expected_ab to an integer first, which collapses some of the
    # sub-1-at-bat differences between lineup spots. See hits_count_
    # distribution()'s docstring for the full explanation.
    p_hit, adjusted_ba, expected_ab = hit_probability(
        season=(80, 320), recent=(12, 45), vs_hand=(20, 75), lineup_spot=3,
        pitcher_hits_allowed=140, pitcher_ab_faced=580, park_factor=1.0,
    )
    p_over, p_under = over_under_probability(adjusted_ba, expected_ab, line=0.5)
    assert abs(p_over - p_hit) < 0.05
    assert abs(p_under - (1 - p_hit)) < 0.05


def test_over_under_probabilities_sum_to_one():
    p_over, p_under = over_under_probability(adjusted_ba=0.30, expected_ab=4, line=1.5)
    assert abs((p_over + p_under) - 1.0) < 1e-9


def test_higher_line_has_lower_over_probability():
    p_over_05, _ = over_under_probability(adjusted_ba=0.28, expected_ab=4, line=0.5)
    p_over_15, _ = over_under_probability(adjusted_ba=0.28, expected_ab=4, line=1.5)
    p_over_25, _ = over_under_probability(adjusted_ba=0.28, expected_ab=4, line=2.5)
    assert p_over_05 > p_over_15 > p_over_25


if __name__ == "__main__":
    # Allows running `python3 tests/test_hit_model.py` if pytest isn't installed.
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
