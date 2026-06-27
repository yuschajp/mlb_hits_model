"""
mlb_api_client.py

Thin wrapper around MLB's free, public Stats API (statsapi.mlb.com). No API
key required. This is not an officially documented API -- it's the same
backend MLB's own apps use, reverse-engineered and documented by the
open-source community (see the long-running toddrob99/MLB-StatsAPI project
on GitHub for the most complete field reference). The endpoint paths and
field names below reflect that community-established structure as of my
training, but I have NO live network access in the environment where I
wrote this, so I could not hit the real API and confirm the exact JSON
shape myself.

BEFORE RELYING ON THIS: run get_schedule() for today's date and print the
raw JSON (there's a --dump-raw flag wired through run_daily.py for exactly
this), and sanity-check the field paths used in the parsing helpers below
against what actually comes back. The defensive .get() chains throughout
are there so a missing/renamed field degrades to a sensible default
instead of crashing the whole daily run -- but you should still go verify.

Endpoints used:
    GET /v1/schedule                      -- today's games
    GET /v1/game/{gamePk}/boxscore        -- confirmed lineups + final box score
    GET /v1/people/{personId}/stats       -- season / date-range / split stats
"""

from datetime import date, timedelta

import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 10


def _get(path, params=None):
    resp = requests.get(f"{BASE_URL}{path}", params=params or {}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_schedule(game_date=None):
    """
    Returns a list of dicts: [{game_pk, home_team, away_team, home_team_id,
    away_team_id, venue, status}, ...] for the given date (default: today).
    """
    game_date = game_date or date.today()
    raw = _get("/schedule", {"sportId": 1, "date": game_date.strftime("%Y-%m-%d"), "hydrate": "venue"})

    games = []
    for day in raw.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            games.append({
                "game_pk": g.get("gamePk"),
                "status": g.get("status", {}).get("detailedState"),
                "venue": g.get("venue", {}).get("name"),
                "home_team": teams.get("home", {}).get("team", {}).get("name"),
                "home_team_id": teams.get("home", {}).get("team", {}).get("id"),
                "away_team": teams.get("away", {}).get("team", {}).get("name"),
                "away_team_id": teams.get("away", {}).get("team", {}).get("id"),
            })
    return games


def get_handedness_batch(player_ids):
    """
    Returns {player_id: {"bats": code, "throws": code}} for multiple
    players in one request, using the /people endpoint's personIds batch
    param. batSide/pitchHand are biographical (a player's handedness
    doesn't change game to game) and do NOT appear in the boxscore
    response at all -- confirmed by hitting the live API directly; an
    earlier version of this client incorrectly assumed they were on the
    boxscore player object.
    """
    if not player_ids:
        return {}
    ids_str = ",".join(str(p) for p in player_ids)
    raw = _get("/people", {"personIds": ids_str})
    result = {}
    for person in raw.get("people", []):
        pid = person.get("id")
        result[pid] = {
            "bats": person.get("batSide", {}).get("code"),
            "throws": person.get("pitchHand", {}).get("code"),
        }
    return result


def get_confirmed_lineup(game_pk):
    """
    Returns {"home": [...], "away": [...]} where each list has dicts:
    {player_id, name, lineup_spot, hand} sorted by batting order, only for
    starters (battingOrder ending in "00"). Returns empty lists for a team
    if lineups haven't been posted yet (usually ~1-2 hours before first pitch).

    Handedness is fetched separately via get_handedness_batch(), since the
    boxscore response itself doesn't carry batSide/pitchHand.
    """
    raw = _get(f"/game/{game_pk}/boxscore")
    lineups = {"home": [], "away": []}

    for side in ("home", "away"):
        players = raw.get("teams", {}).get(side, {}).get("players", {})
        starters = []
        for _, p in players.items():
            batting_order = p.get("battingOrder")
            if not batting_order or not batting_order.endswith("00"):
                continue
            person = p.get("person", {})
            starters.append({
                "player_id": person.get("id"),
                "name": person.get("fullName"),
                "lineup_spot": int(batting_order[0]),
                "hand": None,  # filled in below
            })
        lineups[side] = sorted(starters, key=lambda x: x["lineup_spot"])

    all_ids = [p["player_id"] for side in lineups.values() for p in side]
    handedness = get_handedness_batch(all_ids)
    for side in lineups.values():
        for p in side:
            p["hand"] = handedness.get(p["player_id"], {}).get("bats")

    return lineups


def get_probable_pitchers(game_pk):
    """
    Returns {"home": {player_id, name, hand}, "away": {...}} from the
    boxscore. Handedness comes from get_handedness_batch() -- see note on
    get_confirmed_lineup() above.
    """
    raw = _get(f"/game/{game_pk}/boxscore")
    result = {}
    pending_ids = []
    for side in ("home", "away"):
        team = raw.get("teams", {}).get(side, {})
        pitchers = team.get("pitchers", [])
        if not pitchers:
            result[side] = None
            continue
        starter_id = pitchers[0]  # first pitcher listed is conventionally the starter
        player = team.get("players", {}).get(f"ID{starter_id}", {})
        person = player.get("person", {})
        result[side] = {"player_id": person.get("id"), "name": person.get("fullName"), "hand": None}
        pending_ids.append(person.get("id"))

    handedness = get_handedness_batch(pending_ids)
    for side, info in result.items():
        if info is not None:
            info["hand"] = handedness.get(info["player_id"], {}).get("throws")
    return result


def get_game_batting_results(game_pk):
    """
    Returns {player_id: hits} for every batter who appeared in a completed
    game (both teams), pulled from the boxscore's per-player batting stat
    line. Used by the grading script to compare predictions to what
    actually happened.
    """
    raw = _get(f"/game/{game_pk}/boxscore")
    results = {}
    for side in ("home", "away"):
        players = raw.get("teams", {}).get(side, {}).get("players", {})
        for _, p in players.items():
            person = p.get("person", {})
            player_id = person.get("id")
            batting = p.get("stats", {}).get("batting", {})
            if player_id is not None and "hits" in batting:
                results[player_id] = int(batting["hits"])
    return results


def get_season_hr_stats(player_id, season=None):
    """Returns (homers, at_bats) for the player's season-to-date hitting line."""
    season = season or date.today().year
    raw = _get(f"/people/{player_id}/stats", {"stats": "season", "group": "hitting", "season": season})
    return _extract_stat_ab(raw, "homeRuns")


def get_recent_hr_stats(player_id, days=30):
    """Returns (homers, at_bats) over the trailing N days."""
    end = date.today()
    start = end - timedelta(days=days)
    raw = _get(f"/people/{player_id}/stats", {
        "stats": "byDateRange",
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "group": "hitting",
    })
    return _extract_stat_ab(raw, "homeRuns")


def get_hr_splits_vs_hand(player_id, hand, season=None):
    """Returns (homers, at_bats) for the player's career-to-date HR split against LHP or RHP."""
    season = season or date.today().year
    sit_code = "vl" if hand == "L" else "vr"
    raw = _get(f"/people/{player_id}/stats", {
        "stats": "statSplits", "sitCodes": sit_code, "group": "hitting", "season": season,
    })
    return _extract_stat_ab(raw, "homeRuns")


def get_pitcher_hr_stats_against(player_id, season=None):
    """Returns (homers_allowed, at_bats_faced) for the pitcher's season-to-date pitching line."""
    season = season or date.today().year
    raw = _get(f"/people/{player_id}/stats", {"stats": "season", "group": "pitching", "season": season})
    return _extract_stat_ab(raw, "homeRuns")


def get_game_hr_results(game_pk):
    """Returns {player_id: homers} for every batter in a completed game's boxscore."""
    raw = _get(f"/game/{game_pk}/boxscore")
    results = {}
    for side in ("home", "away"):
        players = raw.get("teams", {}).get(side, {}).get("players", {})
        for _, p in players.items():
            person = p.get("person", {})
            player_id = person.get("id")
            batting = p.get("stats", {}).get("batting", {})
            if player_id is not None and "homeRuns" in batting:
                results[player_id] = int(batting["homeRuns"])
    return results


def get_season_hitting_stats(player_id, season=None):
    """Returns (hits, at_bats) for the player's season-to-date hitting line."""
    season = season or date.today().year
    raw = _get(f"/people/{player_id}/stats", {"stats": "season", "group": "hitting", "season": season})
    return _extract_hit_ab(raw)


def get_recent_hitting_stats(player_id, days=30):
    """Returns (hits, at_bats) over the trailing N days."""
    end = date.today()
    start = end - timedelta(days=days)
    raw = _get(f"/people/{player_id}/stats", {
        "stats": "byDateRange",
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "group": "hitting",
    })
    return _extract_hit_ab(raw)


def get_splits_vs_hand(player_id, hand, season=None):
    """
    Returns (hits, at_bats) for the player's career-to-date split against
    LHP or RHP. hand should be 'L' or 'R' (the opposing starter's hand).

    NOTE: the exact sitCodes param value for platoon splits is the part of
    this client I'm least certain about without live access -- verify the
    'vl'/'vr' codes below against a real response and adjust if needed.
    """
    season = season or date.today().year
    sit_code = "vl" if hand == "L" else "vr"
    raw = _get(f"/people/{player_id}/stats", {
        "stats": "statSplits", "sitCodes": sit_code, "group": "hitting", "season": season,
    })
    return _extract_hit_ab(raw)


def get_pitcher_stats_against(player_id, season=None):
    """Returns (hits_allowed, at_bats_faced) for the pitcher's season-to-date pitching line."""
    season = season or date.today().year
    raw = _get(f"/people/{player_id}/stats", {"stats": "season", "group": "pitching", "season": season})
    return _extract_hit_ab(raw)


def _extract_hit_ab(raw_stats_response):
    """Shared parsing for the people/stats response shape: stats[].splits[].stat. Hits-specific wrapper."""
    return _extract_stat_ab(raw_stats_response, "hits")


def _extract_stat_ab(raw_stats_response, stat_key):
    """
    Generalized version of _extract_hit_ab: pulls (stat_key value, atBats)
    from the stats[].splits[].stat shape, defaulting to (0, 0) if the
    structure doesn't match. stat_key is "hits" for the hits model,
    "homeRuns" for the home run model -- same response shape either way.
    """
    try:
        splits = raw_stats_response["stats"][0]["splits"]
        if not splits:
            return (0, 0)
        stat = splits[0]["stat"]
        value = int(stat.get(stat_key, 0))
        at_bats = int(stat.get("atBats", 0))
        return (value, at_bats)
    except (KeyError, IndexError, TypeError, ValueError):
        return (0, 0)


# ── Strikeout-specific functions ──────────────────────────────────────────────

def _extract_k_innings(raw_stats_response):
    """
    Pulls (strikeouts, innings_pitched_as_float) from a pitching stats
    response. innings_pitched in the API is a string like "34.1" meaning
    34 full innings + 1 out, which equals 34 + 1/3 = 34.333 innings.
    """
    try:
        splits = raw_stats_response["stats"][0]["splits"]
        if not splits:
            return (0, 0.0)
        stat = splits[0]["stat"]
        ks = int(stat.get("strikeOuts", 0))
        ip_str = stat.get("inningsPitched", "0.0")
        # Convert "34.1" to 34.333 etc.
        parts = str(ip_str).split(".")
        full_innings = int(parts[0])
        outs = int(parts[1]) if len(parts) > 1 else 0
        ip_float = full_innings + outs / 3.0
        return (ks, round(ip_float, 3))
    except (KeyError, IndexError, TypeError, ValueError):
        return (0, 0.0)


def get_pitcher_season_k_stats(player_id, season=None):
    """
    Returns (strikeouts, innings_pitched) for a pitcher's season-to-date.
    innings_pitched is a float (e.g. 34.333 for 34.1 innings).
    """
    season = season or date.today().year
    raw = _get(f"/people/{player_id}/stats", {"stats": "season", "group": "pitching", "season": season})
    return _extract_k_innings(raw)


def get_pitcher_recent_k_stats(player_id, days=35):
    """
    Returns (strikeouts, innings_pitched) over the trailing N days.
    35 days ≈ last 5-6 starts for a regular rotation starter.
    """
    end = date.today()
    start = end - timedelta(days=days)
    raw = _get(f"/people/{player_id}/stats", {
        "stats": "byDateRange",
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "group": "pitching",
    })
    return _extract_k_innings(raw)


def get_pitcher_start_log(player_id, season=None):
    """
    Returns a list of individual game logs for the pitcher this season,
    used to compute average innings per start and number of starts made.
    Returns list of dicts with 'inningsPitched', 'strikeOuts', 'date'.
    """
    season = season or date.today().year
    raw = _get(f"/people/{player_id}/stats", {"stats": "gameLog", "group": "pitching", "season": season})
    starts = []
    try:
        splits = raw["stats"][0]["splits"]
        for s in splits:
            stat = s.get("stat", {})
            # Only count starts, not relief appearances
            if stat.get("gamesStarted", 0) > 0 or int(stat.get("inningsPitched", "0").split(".")[0]) >= 3:
                ip_str = stat.get("inningsPitched", "0.0")
                parts = str(ip_str).split(".")
                ip_float = int(parts[0]) + (int(parts[1]) if len(parts) > 1 else 0) / 3.0
                starts.append({
                    "date":   s.get("date", ""),
                    "ks":     int(stat.get("strikeOuts", 0)),
                    "ip":     round(ip_float, 3),
                })
    except (KeyError, IndexError, TypeError):
        pass
    return starts


def get_team_k_rate(team_id, season=None):
    """
    Returns a team's strikeout rate (Ks per plate appearance) from their
    season batting stats. Used as the opponent K rate input to k_model.py.
    Returns a float like 0.227 (league average).
    """
    season = season or date.today().year
    raw = _get(f"/teams/{team_id}/stats", {"stats": "season", "group": "hitting", "season": season})
    try:
        splits = raw["stats"][0]["splits"]
        if not splits:
            return 0.227
        stat = splits[0]["stat"]
        ks  = int(stat.get("strikeOuts", 0))
        pas = int(stat.get("plateAppearances", 0))
        return round(ks / pas, 4) if pas > 0 else 0.227
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.227


def get_game_pitcher_k_results(game_pk):
    """
    Returns {pitcher_id: strikeouts} for both starting pitchers in a
    completed game's boxscore. Only returns starters (excludes relievers).
    """
    raw = _get(f"/game/{game_pk}/boxscore")
    results = {}
    for side in ("home", "away"):
        pitchers = raw.get("teams", {}).get(side, {}).get("pitchers", [])
        players  = raw.get("teams", {}).get(side, {}).get("players", {})
        if not pitchers:
            continue
        # The first pitcher listed is the starter
        starter_id = pitchers[0] if pitchers else None
        if starter_id is None:
            continue
        pitcher_key = f"ID{starter_id}"
        pitcher_data = players.get(pitcher_key, {})
        person_id = pitcher_data.get("person", {}).get("id")
        pitching  = pitcher_data.get("stats", {}).get("pitching", {})
        if person_id and "strikeOuts" in pitching:
            results[person_id] = int(pitching["strikeOuts"])
    return results
