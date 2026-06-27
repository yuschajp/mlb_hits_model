"""
wc_data_client.py

Pulls World Cup 2026 data from openfootball/worldcup.json on GitHub.
No API key required.

Actual 2026 schema:
  {
    "name": "World Cup 2026",
    "matches": [
      {
        "round": "Matchday 1",
        "date": "2026-06-11",
        "time": "13:00 UTC-6",
        "team1": "Mexico",          <- plain string
        "team2": "South Africa",
        "score": {"ft": [2, 0], "ht": [1, 0]},  <- null if not played
        "goals1": [...],
        "goals2": [...],
        "group": "Group A",
        "ground": "Mexico City"
      }, ...
    ]
  }
"""

import requests
from datetime import date as date_type, timedelta

WC_JSON_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
TIMEOUT = 10
_cache = {}


def _fetch():
    if "data" in _cache:
        return _cache["data"]
    try:
        resp = requests.get(WC_JSON_URL, timeout=TIMEOUT)
        resp.raise_for_status()
        _cache["data"] = resp.json()
        return _cache["data"]
    except Exception as e:
        print(f"  [warn] Could not fetch worldcup.json: {e}")
        return {}


def _parse_match(m):
    score = m.get("score")
    ft    = score.get("ft") if isinstance(score, dict) else None
    home_goals = int(ft[0]) if ft else None
    away_goals = int(ft[1]) if ft else None
    team1 = m.get("team1", "")
    team2 = m.get("team2", "")
    # Teams are plain strings in 2026 format
    if isinstance(team1, dict):
        team1 = team1.get("name", "")
    if isinstance(team2, dict):
        team2 = team2.get("name", "")
    return {
        "match_id":   m.get("num", hash(f"{m.get('date','')}{team1}{team2}") % 100000),
        "home_team":  team1,
        "away_team":  team2,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "date":       m.get("date", ""),
        "time":       m.get("time", ""),
        "stage":      m.get("round", ""),
        "group":      m.get("group", ""),
        "venue":      m.get("ground", ""),
        "status":     "FINISHED" if ft is not None else "SCHEDULED",
    }


def get_all_matches():
    data = _fetch()
    return [_parse_match(m) for m in data.get("matches", [])]


def get_all_completed_matches():
    return [m for m in get_all_matches() if m["status"] == "FINISHED"]


def get_todays_matches():
    today = date_type.today().isoformat()
    return [m for m in get_all_matches() if m["date"] == today]


def get_upcoming_matches(days_ahead=3):
    today = date_type.today()
    dates = {(today + timedelta(days=i)).isoformat() for i in range(days_ahead + 1)}
    return [m for m in get_all_matches()
            if m["date"] in dates and m["status"] == "SCHEDULED"]


def get_match_result(match_id):
    for m in get_all_matches():
        if m["match_id"] == match_id and m["status"] == "FINISHED":
            return m
    return None


def get_standings():
    from collections import defaultdict
    completed  = get_all_completed_matches()
    all_matches = get_all_matches()
    team_group  = {m["home_team"]: m["group"] for m in all_matches if m["group"]}
    team_group.update({m["away_team"]: m["group"] for m in all_matches if m["group"]})

    table = defaultdict(lambda: {"played":0,"points":0,"gf":0,"ga":0,"group":""})
    for m in completed:
        h, a, hg, ag = m["home_team"], m["away_team"], m["home_goals"], m["away_goals"]
        table[h]["played"] += 1; table[a]["played"] += 1
        table[h]["gf"] += hg; table[h]["ga"] += ag
        table[a]["gf"] += ag; table[a]["ga"] += hg
        if hg > ag:   table[h]["points"] += 3
        elif hg == ag: table[h]["points"] += 1; table[a]["points"] += 1
        else:          table[a]["points"] += 3
        table[h]["group"] = team_group.get(h, "")
        table[a]["group"] = team_group.get(a, "")

    return [{"team": t, "gd": v["gf"]-v["ga"], **v}
            for t, v in sorted(table.items(), key=lambda x: -x[1]["points"])]
