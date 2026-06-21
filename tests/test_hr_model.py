"""Sanity tests for hr_model.py -- no network required, pure math checks."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hr_model import (
    LEAGUE_AVG_HR_RATE,
    blended_hr_rate,
    hr_over_under_probability,
    hr_probability,
    pitcher_hr_adjustment,
    stabilized_hr_rate,
)


def test_stabilized_hr_rate_pulls_small_samples_hard_toward_league_average():
    # 2 homers in 15 AB looks like a 13% HR rate, but with prior_ab=400
    # that should barely move off league average.
    r = stabilized_hr_rate(homers=2, at_bats=15, prior_ab=400)
    assert abs(r - LEAGUE_AVG_HR_RATE) < 0.01


def test_stabilized_hr_rate_trusts_large_samples():
    # A real 40-homer, 600-AB season should land close to its own rate.
    r = stabilized_hr_rate(homers=40, at_bats=600, prior_ab=400)
    assert abs(r - 40 / 600) < 0.015


def test_stabilized_hr_rate_rejects_invalid_input():
    try:
        stabilized_hr_rate(homers=5, at_bats=2)
        assert False, "should have raised"
    except ValueError:
        pass


def test_pitcher_hr_adjustment_is_clipped():
    adj = pitcher_hr_adjustment(pitcher_hr_allowed=0, pitcher_ab_faced=3)
    assert 0.5 <= adj <= 1.8


def test_hr_probability_monotonic_in_lineup_spot():
    common_kwargs = dict(
        season=(25, 450), recent=(3, 50), vs_hand=(10, 180),
        pitcher_hr_allowed=18, pitcher_ab_faced=600, park_factor=1.0,
    )
    p_leadoff, _, _ = hr_probability(lineup_spot=1, **common_kwargs)
    p_ninth, _, _ = hr_probability(lineup_spot=9, **common_kwargs)
    assert p_leadoff > p_ninth


def test_hr_probability_responds_to_park_factor():
    common_kwargs = dict(
        season=(25, 450), recent=(3, 50), vs_hand=(10, 180), lineup_spot=3,
        pitcher_hr_allowed=18, pitcher_ab_faced=600,
    )
    p_coors, _, _ = hr_probability(park_factor=1.25, **common_kwargs)
    p_oracle, _, _ = hr_probability(park_factor=0.91, **common_kwargs)
    assert p_coors > p_oracle


def test_hr_probability_is_a_valid_probability():
    p, adjusted_rate, expected_ab = hr_probability(
        season=(25, 450), recent=(3, 50), vs_hand=(10, 180), lineup_spot=4,
        pitcher_hr_allowed=18, pitcher_ab_faced=600, park_factor=1.0,
    )
    assert 0.0 < p < 1.0
    assert 0.0 < adjusted_rate < 1.0
    assert expected_ab > 0


def test_over_0_5_matches_hr_probability_exactly():
    # Unlike the hits model's over/under, this should be an EXACT match --
    # both use the same continuous Poisson formula, no rounding step.
    p_hr, adjusted_rate, expected_ab = hr_probability(
        season=(25, 450), recent=(3, 50), vs_hand=(10, 180), lineup_spot=3,
        pitcher_hr_allowed=18, pitcher_ab_faced=600, park_factor=1.0,
    )
    p_over, p_under = hr_over_under_probability(adjusted_rate, expected_ab, line=0.5)
    assert abs(p_over - p_hr) < 1e-12
    assert abs(p_under - (1 - p_hr)) < 1e-12


def test_higher_hr_line_has_lower_over_probability():
    p_over_05, _ = hr_over_under_probability(adjusted_rate=0.04, expected_ab=4.1, line=0.5)
    p_over_15, _ = hr_over_under_probability(adjusted_rate=0.04, expected_ab=4.1, line=1.5)
    assert p_over_05 > p_over_15


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
