"""
Integration test for run_daily_hr.build_hr_predictions_for_game -- mocks
every mlb_api_client call, mirroring test_run_daily_integration.py for the
hits model.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from src import mlb_api_client as client
from src.park_factors import load_park_factors

import run_daily_hr  # noqa: E402

FAKE_GAME = {
    "game_pk": 999,
    "home_team": "New York Yankees",
    "away_team": "Boston Red Sox",
    "venue": "Yankee Stadium",
}

FAKE_LINEUPS = {
    "home": [{"player_id": 1, "name": "Home Slugger", "lineup_spot": 4, "hand": "L"}],
    "away": [{"player_id": 2, "name": "Away Slugger", "lineup_spot": 4, "hand": "R"}],
}

FAKE_PITCHERS = {
    "home": {"player_id": 10, "name": "Home Starter", "hand": "R"},
    "away": {"player_id": 20, "name": "Away Starter", "hand": "L"},
}


def test_build_hr_predictions_for_game(monkeypatch):
    monkeypatch.setattr(client, "get_confirmed_lineup", lambda game_pk: FAKE_LINEUPS)
    monkeypatch.setattr(client, "get_probable_pitchers", lambda game_pk: FAKE_PITCHERS)
    monkeypatch.setattr(client, "get_season_hr_stats", lambda pid: (22, 420))
    monkeypatch.setattr(client, "get_recent_hr_stats", lambda pid, days=30: (3, 45))
    monkeypatch.setattr(client, "get_hr_splits_vs_hand", lambda pid, hand: (12, 200))
    monkeypatch.setattr(client, "get_pitcher_hr_stats_against", lambda pid: (16, 580))

    # Yankee Stadium HR park factor (1.18) should load correctly from the dedicated HR table.
    hr_park_factors = load_park_factors(path=run_daily_hr.HR_PARK_FACTORS_PATH, value_col="hr_park_factor")
    rows = run_daily_hr.build_hr_predictions_for_game(FAKE_GAME, hr_park_factors)

    assert len(rows) == 2
    names = {r["player_name"] for r in rows}
    assert names == {"Home Slugger", "Away Slugger"}
    for r in rows:
        assert 0.0 < r["p_hr"] < 1.0
        assert r["game_pk"] == 999
    yankee_row = next(r for r in rows if r["team"] == "New York Yankees")
    assert yankee_row["park_factor"] == 1.18


if __name__ == "__main__":
    class _MonkeyPatch:
        def __init__(self):
            self._originals = []

        def setattr(self, obj, name, value):
            self._originals.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, val in self._originals:
                setattr(obj, name, val)

    mp = _MonkeyPatch()
    test_build_hr_predictions_for_game(mp)
    mp.undo()
    print("  ok  test_build_hr_predictions_for_game")
    print("\nAll 1 tests passed.")
