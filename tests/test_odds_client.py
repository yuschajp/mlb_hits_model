"""
Tests for odds_client.py's parsing logic, using fixture JSON that mirrors
The Odds API's documented response shape (from their own published
examples) -- not live network calls, since this sandbox has no internet
access and no API key. Confirms the parsing code is internally correct
against the documented schema; can't confirm The Odds API's live response
matches it exactly. Verify with a real key before trusting this in
production, same as the MLB Stats API client.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("ODDS_API_KEY", "test-key-not-real")

from src import odds_client as odds  # noqa: E402

EVENT_ODDS_FIXTURE = {
    "id": "abc123",
    "home_team": "Houston Astros",
    "away_team": "Texas Rangers",
    "bookmakers": [
        {
            "key": "betmgm",
            "title": "BetMGM",
            "markets": [
                {
                    "key": "batter_hits",
                    "outcomes": [
                        {"name": "Over", "description": "Yordan Alvarez", "price": -135, "point": 0.5},
                        {"name": "Under", "description": "Yordan Alvarez", "price": 110, "point": 0.5},
                        {"name": "Over", "description": "Josh Jung", "price": -110, "point": 0.5},
                        {"name": "Under", "description": "Josh Jung", "price": -120, "point": 0.5},
                    ],
                },
            ],
        },
        {
            "key": "fanduel",
            "title": "FanDuel",
            "markets": [
                {
                    "key": "batter_hits",
                    "outcomes": [
                        # FanDuel offers a better (less negative) price on Alvarez than BetMGM
                        {"name": "Over", "description": "Yordan Alvarez", "price": -120, "point": 0.5},
                        {"name": "Under", "description": "Yordan Alvarez", "price": 100, "point": 0.5},
                    ],
                },
            ],
        },
    ],
}


def test_extract_best_over_05_picks_highest_price_across_books():
    best = odds._extract_best_over_05(EVENT_ODDS_FIXTURE)
    # -120 is a better (higher) price than -135 for the bettor
    assert best["Yordan Alvarez"] == {"price": -120, "bookmaker": "fanduel"}
    assert best["Josh Jung"] == {"price": -110, "bookmaker": "betmgm"}


def test_extract_best_over_05_ignores_under_outcomes():
    best = odds._extract_best_over_05(EVENT_ODDS_FIXTURE)
    assert len(best) == 2  # only the two batters with an Over 0.5 outcome


def test_american_to_implied_prob_negative_odds():
    # -120 means bet 120 to win 100 -> implied prob = 120/220
    prob = odds.american_to_implied_prob(-120)
    assert abs(prob - (120 / 220)) < 1e-9


def test_american_to_implied_prob_positive_odds():
    # +150 means bet 100 to win 150 -> implied prob = 100/250
    prob = odds.american_to_implied_prob(150)
    assert abs(prob - (100 / 250)) < 1e-9


def test_get_todays_mlb_events_requires_api_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    try:
        odds.get_todays_mlb_events()
        assert False, "should have raised without an API key set"
    except RuntimeError as e:
        assert "ODDS_API_KEY" in str(e)


if __name__ == "__main__":
    class _MonkeyPatch:
        def __init__(self):
            self._originals = []

        def setattr(self, obj, name, value):
            self._originals.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def delenv(self, name, raising=True):
            self._originals.append(("env", name, os.environ.get(name)))
            os.environ.pop(name, None)

        def undo(self):
            for entry in self._originals:
                if entry[0] == "env":
                    _, name, val = entry
                    if val is not None:
                        os.environ[name] = val
                else:
                    obj, name, val = entry
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
