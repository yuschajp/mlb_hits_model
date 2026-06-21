"""
Tests for mlb_api_client.py's parsing logic, using fixture JSON that
mirrors the documented Stats API response shape -- NOT live network calls
(this sandbox has no network access). These confirm the parsing code is
internally correct against the assumed schema; they can't confirm the
assumed schema itself matches the live API. Run scripts/run_daily.py with
--dump-raw against a real date to verify that part once you have network.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import mlb_api_client as client

SCHEDULE_FIXTURE = {
    "dates": [{
        "games": [{
            "gamePk": 745000,
            "status": {"detailedState": "Scheduled"},
            "venue": {"name": "Yankee Stadium"},
            "teams": {
                "home": {"team": {"id": 147, "name": "New York Yankees"}},
                "away": {"team": {"id": 111, "name": "Boston Red Sox"}},
            },
        }],
    }],
}

BOXSCORE_FIXTURE = {
    "teams": {
        "home": {
            "pitchers": [592789],
            "players": {
                "ID592789": {
                    "person": {"id": 592789, "fullName": "Starter Pitcher"},
                },
                "ID100001": {
                    "person": {"id": 100001, "fullName": "Leadoff Hitter"},
                    "battingOrder": "100",
                },
                "ID100002": {
                    "person": {"id": 100002, "fullName": "Cleanup Hitter"},
                    "battingOrder": "400",
                },
                "ID100003": {
                    # A pinch hitter who isn't a starter -- battingOrder
                    # doesn't end in "00", so should be excluded.
                    "person": {"id": 100003, "fullName": "Bench Player"},
                    "battingOrder": "401",
                },
            },
        },
        "away": {"pitchers": [], "players": {}},
    },
}

STATS_FIXTURE = {
    "stats": [{
        "splits": [{"stat": {"hits": 88, "atBats": 310, "avg": ".284"}}],
    }],
}

EMPTY_STATS_FIXTURE = {"stats": [{"splits": []}]}


def test_extract_hit_ab_parses_normal_response():
    assert client._extract_hit_ab(STATS_FIXTURE) == (88, 310)


def test_extract_hit_ab_handles_empty_splits():
    assert client._extract_hit_ab(EMPTY_STATS_FIXTURE) == (0, 0)


def test_extract_hit_ab_handles_malformed_response():
    assert client._extract_hit_ab({"unexpected": "shape"}) == (0, 0)


def test_get_schedule_parses_games(monkeypatch):
    monkeypatch.setattr(client, "_get", lambda path, params=None: SCHEDULE_FIXTURE)
    games = client.get_schedule()
    assert len(games) == 1
    assert games[0]["home_team"] == "New York Yankees"
    assert games[0]["away_team"] == "Boston Red Sox"
    assert games[0]["game_pk"] == 745000


PEOPLE_FIXTURE = {
    "people": [
        {"id": 100001, "fullName": "Leadoff Hitter", "batSide": {"code": "L"}, "pitchHand": {"code": "R"}},
        {"id": 100002, "fullName": "Cleanup Hitter", "batSide": {"code": "R"}, "pitchHand": {"code": "R"}},
        {"id": 592789, "fullName": "Starter Pitcher", "batSide": {"code": "R"}, "pitchHand": {"code": "R"}},
    ],
}


def _fake_get_dispatch(path, params=None):
    """Routes a fake _get call to the right fixture based on the endpoint path."""
    if "/people" in path:
        return PEOPLE_FIXTURE
    return BOXSCORE_FIXTURE


def test_get_confirmed_lineup_only_includes_starters(monkeypatch):
    monkeypatch.setattr(client, "_get", _fake_get_dispatch)
    lineups = client.get_confirmed_lineup(745000)
    home = lineups["home"]
    assert len(home) == 2  # the bench player (battingOrder "401") must be excluded
    assert home[0]["name"] == "Leadoff Hitter"
    assert home[0]["lineup_spot"] == 1
    assert home[0]["hand"] == "L"  # from the /people lookup, not the boxscore
    assert home[1]["name"] == "Cleanup Hitter"
    assert home[1]["lineup_spot"] == 4
    assert lineups["away"] == []


def test_get_probable_pitchers_parses_starter(monkeypatch):
    monkeypatch.setattr(client, "_get", _fake_get_dispatch)
    pitchers = client.get_probable_pitchers(745000)
    assert pitchers["home"]["name"] == "Starter Pitcher"
    assert pitchers["home"]["hand"] == "R"  # from the /people lookup, not the boxscore
    assert pitchers["away"] is None


def test_get_handedness_batch_parses_multiple_players(monkeypatch):
    monkeypatch.setattr(client, "_get", _fake_get_dispatch)
    handedness = client.get_handedness_batch([100001, 100002])
    assert handedness[100001] == {"bats": "L", "throws": "R"}
    assert handedness[100002] == {"bats": "R", "throws": "R"}


BOXSCORE_WITH_RESULTS_FIXTURE = {
    "teams": {
        "home": {
            "players": {
                "ID100001": {"person": {"id": 100001}, "stats": {"batting": {"hits": 2, "atBats": 4}}},
                "ID100002": {"person": {"id": 100002}, "stats": {"batting": {"hits": 0, "atBats": 3}}},
            },
        },
        "away": {
            "players": {
                "ID200001": {"person": {"id": 200001}, "stats": {"batting": {"hits": 1, "atBats": 4}}},
            },
        },
    },
}


def test_get_game_batting_results_parses_hits(monkeypatch):
    monkeypatch.setattr(client, "_get", lambda path, params=None: BOXSCORE_WITH_RESULTS_FIXTURE)
    results = client.get_game_batting_results(745000)
    assert results == {100001: 2, 100002: 0, 200001: 1}


HR_STATS_FIXTURE = {
    "stats": [{
        "splits": [{"stat": {"hits": 88, "homeRuns": 14, "atBats": 310, "avg": ".284"}}],
    }],
}


def test_extract_stat_ab_pulls_home_runs():
    assert client._extract_stat_ab(HR_STATS_FIXTURE, "homeRuns") == (14, 310)


def test_extract_stat_ab_still_pulls_hits():
    assert client._extract_stat_ab(HR_STATS_FIXTURE, "hits") == (88, 310)


def test_get_game_hr_results_parses_home_runs(monkeypatch):
    boxscore_with_hr = {
        "teams": {
            "home": {"players": {
                "ID1": {"person": {"id": 1}, "stats": {"batting": {"hits": 2, "homeRuns": 1}}},
                "ID2": {"person": {"id": 2}, "stats": {"batting": {"hits": 1, "homeRuns": 0}}},
            }},
            "away": {"players": {}},
        },
    }
    monkeypatch.setattr(client, "_get", lambda path, params=None: boxscore_with_hr)
    results = client.get_game_hr_results(745000)
    assert results == {1: 1, 2: 0}


if __name__ == "__main__":
    # Minimal monkeypatch shim so this can run without pytest installed.
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
        if "monkeypatch" in t.__code__.co_varnames:
            t(mp)
        else:
            t()
        mp.undo()
        print(f"  ok  {name}")
    print(f"\nAll {len(tests)} tests passed.")
