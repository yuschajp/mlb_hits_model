"""
odds_client.py

Wrapper around The Odds API (the-odds-api.com) for MLB player-hits prop
odds. Requires a free API key -- sign up at https://the-odds-api.com/.

Why this provider: it has a documented `batter_hits` market for MLB, and
a sportsbook's "Over 0.5 hits" line is mathematically identical to our
"record a hit" prop (over_under_probability(..., line=0.5) in
hit_model.py). Current (non-historical) odds, including player props, are
available on the free usage plan as of when this was written -- but
player-prop calls cost usage credits separately from simpler endpoints,
and exact free-tier quotas change over time. Check
https://the-odds-api.com/#get-access for the current plan limits before
wiring this into a full daily slate -- this client has NOT been tested
against a live response (no API key or network access in the environment
where it was written). Verify field paths the same way the MLB Stats API
client was verified: print a raw response and compare before trusting it.

Set the ODDS_API_KEY environment variable before using this module:
    export ODDS_API_KEY=your_key_here
"""

import os

import requests

BASE_URL = "https://api.the-odds-api.com/v4"
TIMEOUT = 10


def _api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            "Set the ODDS_API_KEY environment variable (free signup at "
            "https://the-odds-api.com/) before calling odds_client functions."
        )
    return key


def get_todays_mlb_events():
    """Returns [{event_id, home_team, away_team, commence_time}, ...]."""
    resp = requests.get(
        f"{BASE_URL}/sports/baseball_mlb/events",
        params={"apiKey": _api_key()},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return [
        {
            "event_id": e.get("id"),
            "home_team": e.get("home_team"),
            "away_team": e.get("away_team"),
            "commence_time": e.get("commence_time"),
        }
        for e in resp.json()
    ]


def get_best_over_0_5_hits_odds(event_id, regions="us"):
    """
    Returns {player_name: {"price": american_odds, "bookmaker": key}},
    keeping only the best (highest/most favorable) price across all
    available bookmakers for the "Over 0.5 hits" line -- i.e. the best
    available price for "this player records a hit."
    """
    resp = requests.get(
        f"{BASE_URL}/sports/baseball_mlb/events/{event_id}/odds",
        params={"apiKey": _api_key(), "regions": regions, "markets": "batter_hits", "oddsFormat": "american"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return _extract_best_over_05(resp.json())


def _extract_best_over_05(raw_event_odds_response):
    best = {}
    for bookmaker in raw_event_odds_response.get("bookmakers", []):
        bk_key = bookmaker.get("key")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "batter_hits":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") != "Over" or outcome.get("point") != 0.5:
                    continue
                player = outcome.get("description")
                price = outcome.get("price")
                if player is None or price is None:
                    continue
                if player not in best or price > best[player]["price"]:
                    best[player] = {"price": price, "bookmaker": bk_key}
    return best


def american_to_implied_prob(american_odds):
    """Converts American odds to implied probability (includes the bookmaker's margin -- not de-vigged)."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return -american_odds / (-american_odds + 100)
