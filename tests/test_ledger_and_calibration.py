"""
Tests for ledger.py and calibration.py -- pure local I/O and math, no
network required.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.calibration import brier_score, calibration_table
from src.ledger import append_predictions, load_ledger, update_outcomes


def _sample_rows():
    return [
        {"date": "2026-06-19", "game_pk": 1, "player_id": 100, "player_name": "A",
         "team": "Yankees", "opponent": "Red Sox", "lineup_spot": 1, "venue": "Yankee Stadium",
         "park_factor": 1.04, "opponent_pitcher": "X", "p_hit": 0.72, "adjusted_ba": 0.30,
         "expected_ab": 4.3},
        {"date": "2026-06-19", "game_pk": 1, "player_id": 101, "player_name": "B",
         "team": "Red Sox", "opponent": "Yankees", "lineup_spot": 9, "venue": "Yankee Stadium",
         "park_factor": 1.04, "opponent_pitcher": "Y", "p_hit": 0.45, "adjusted_ba": 0.20,
         "expected_ab": 3.5},
    ]


def test_append_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ledger.csv"
        append_predictions(_sample_rows(), path)
        df = load_ledger(path)
        assert len(df) == 2
        assert set(df["player_name"]) == {"A", "B"}
        assert (df["graded"] == False).all()  # noqa: E712


def test_append_twice_accumulates_rows():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ledger.csv"
        append_predictions(_sample_rows(), path)
        append_predictions(_sample_rows(), path)
        df = load_ledger(path)
        assert len(df) == 4


def test_update_outcomes_grades_correct_rows():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ledger.csv"
        append_predictions(_sample_rows(), path)
        update_outcomes(path, {100: 1, 101: 0}, game_pk=1)
        df = load_ledger(path)
        row_a = df[df["player_id"] == 100].iloc[0]
        row_b = df[df["player_id"] == 101].iloc[0]
        assert row_a["actual_hit"] == 1 and row_a["graded"] == True  # noqa: E712
        assert row_b["actual_hit"] == 0 and row_b["graded"] == True  # noqa: E712


def test_brier_score_perfect_prediction_is_zero():
    df = pd.DataFrame({"p_hit": [1.0, 0.0], "actual_hit": [1, 0]})
    assert brier_score(df) == 0.0


def test_brier_score_ignores_ungraded_rows():
    df = pd.DataFrame({"p_hit": [0.6, 0.7], "actual_hit": [1, None]})
    score = brier_score(df)
    assert abs(score - 0.16) < 1e-9  # (0.6 - 1)^2 = 0.16, the ungraded row is dropped


def test_calibration_table_buckets_correctly():
    df = pd.DataFrame({
        "p_hit": [0.55, 0.58, 0.61, 0.15],
        "actual_hit": [1, 1, 0, 0],
    })
    table = calibration_table(df, n_bins=5)
    bucket = table[table["PredRange"] == "0.6-0.8"]
    # Only the 0.61 prediction falls in this bucket
    assert len(bucket) == 1
    assert bucket.iloc[0]["N"] == 1


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
