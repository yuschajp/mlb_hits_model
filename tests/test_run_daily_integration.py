"""
Integration test for run_daily.build_predictions_for_game -- mocks every
mlb_api_client call so this exercises the full orchestration logic (lineup
-> opposing pitcher -> stats lookups -> hit_model -> row construction)
without touching the network. Confirms the wiring is correct; does not
confirm the live API actually returns data in this shape.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from src import mlb_api_client as client
from src.park_factors import load_park_factors

import run_daily  # noqa: E402  (path appended above)

FAKE_GAME = {
    "game_pk": 999,
    "home_team": "New York Yankees",
    "away_team": "Boston Red Sox",
    "venue": "Yankee Stadium",
}

FAKE_LINEUPS = {
    "home": [{"player_id": 1, "name": "Home Leadoff", "lineup_spot": 1, "hand": "L"}],
    "away": [{"player_id": 2, "name": "Away Leadoff", "lineup_spot": 1, "hand": "R"}],
}

FAKE_PITCHERS = {
    "home": {"player_id": 10, "name": "Home Starter", "hand": "R"},
    "away": {"player_id": 20, "name": "Away Starter", "hand": "L"},
}


def test_build_predictions_for_game(monkeypatch):
    monkeypatch.setattr(client, "get_confirmed_lineup", lambda game_pk: FAKE_LINEUPS)
    monkeypatch.setattr(client, "get_probable_pitchers", lambda game_pk: FAKE_PITCHERS)
    monkeypatch.setattr(client, "get_season_hitting_stats", lambda pid: (80, 320))
    monkeypatch.setattr(client, "get_recent_hitting_stats", lambda pid, days=30: (10, 38))
    monkeypatch.setattr(client, "get_splits_vs_hand", lambda pid, hand: (15, 60))
    monkeypatch.setattr(client, "get_pitcher_stats_against", lambda pid: (140, 580))

    rows = run_daily.build_predictions_for_game(FAKE_GAME, load_park_factors())

    assert len(rows) == 2
    names = {r["player_name"] for r in rows}
    assert names == {"Home Leadoff", "Away Leadoff"}
    for r in rows:
        assert 0.0 < r["p_hit"] < 1.0
        assert r["game_pk"] == 999


def test_build_predictions_skips_game_with_no_lineups(monkeypatch):
    monkeypatch.setattr(client, "get_confirmed_lineup", lambda game_pk: {"home": [], "away": []})
    monkeypatch.setattr(client, "get_probable_pitchers", lambda game_pk: FAKE_PITCHERS)
    rows = run_daily.build_predictions_for_game(FAKE_GAME, load_park_factors())
    assert rows == []


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

    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        mp = _MonkeyPatch()
        t(mp)
        mp.undo()
        print(f"  ok  {name}")
    print(f"\nAll {len(tests)} tests passed.")
