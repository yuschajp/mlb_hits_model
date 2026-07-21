"""
tennis_data_client.py

Two data sources:

1. HISTORICAL MATCHES (for Elo computation) -- pulled from local TML-Database clone.

2. LIVE ATP MATCHES (today's draw and results) -- pulled from ESPN's public API.
   Pulls ALL ATP events currently in season (majors, Masters, 500s, 250s, etc).
"""

import csv
import requests
from datetime import date
from pathlib import Path

TENNIS_DATA_DIR = Path.home() / "Desktop" / "tennis_data"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"
TIMEOUT = 15
_cache = {}

TOUR_TYPE_TEXT = {
    "atp": "men's singles",
    "wta": "women's singles",
}


# ── Historical matches (local TML-Database clone) ─────────────────────────────

def _read_local_csv(year):
    """Reads a year's CSV from the local TML-Database clone."""
    path = TENNIS_DATA_DIR / f"{year}.csv"
    if path in _cache:
        return _cache[path]
    if not path.exists():
        print(f"  [warn] {path} not found. Clone with:")
        print(f"    git clone https://github.com/Tennismylife/TML-Database.git {TENNIS_DATA_DIR}")
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    _cache[path] = rows
    return rows


def _parse_matches(rows):
    matches = []
    for r in rows:
        try:
            matches.append({
                "date":       r.get("tourney_date", ""),
                "tournament": r.get("tourney_name", ""),
                "surface":    r.get("surface", ""),
                "round":      r.get("round", ""),
                "winner":     r.get("winner_name", ""),
                "loser":      r.get("loser_name", ""),
                "score":      r.get("score", ""),
            })
        except Exception:
            continue
    return matches


def get_recent_matches(tour="atp", years=None):
    """
    Returns matches from recent years from the local TML-Database clone.

    tour="wta" currently returns an empty list.
    """
    if tour != "atp":
        print(f"  [warn] TML-Database is ATP-only. No WTA data source configured yet.")
        return []

    current_year = date.today().year
    if years is None:
        years = [current_year, current_year - 1, current_year - 2]

    all_matches = []
    for year in years:
        rows = _read_local_csv(year)
        if rows:
            all_matches.extend(_parse_matches(rows))

    return all_matches


def get_grass_matches(tour="atp", years=None):
    matches = get_recent_matches(tour=tour, years=years)
    return [m for m in matches if m["surface"].lower() == "grass"]


# ── Live ATP matches (ESPN) ────────────────────────────────────────────────────

def _espn_get(league, resource, params=None):
    """league: 'atp' or 'wta'"""
    url = f"{ESPN_BASE}/{league}/{resource}"
    try:
        resp = requests.get(url, params=params or {}, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [warn] ESPN API error ({league}/{resource}): {e}")
        return {}


def get_atp_matches(tour="atp", date_range=None):
    """
    Returns ALL ATP SINGLES matches from ESPN for today's events.
    Pulls from any tournament currently in season (majors, Masters, 500s, 250s).
    
    No auto-detection -- just grabs whatever matches are live on ESPN today.

    Returns list of dicts: {date, round, completed, player1, player2,
                             winner, loser (if completed), tournament}
    """
    if tour not in TOUR_TYPE_TEXT:
        raise ValueError(f"tour must be one of {list(TOUR_TYPE_TEXT)}, got {tour!r}")
    wanted_type_text = TOUR_TYPE_TEXT[tour]

    # Fetch all events from ESPN
    data = _espn_get("atp", "scoreboard", {})
    events = data.get("events", [])

    if not events:
        print(f"  [info] No ATP events found on ESPN scoreboard.")
        return []

    matches = []
    
    for event in events:
        event_name = event.get("name", "")
        tournament_name = event_name
        
        for grouping in event.get("groupings", []):
            grouping_name = grouping.get("grouping", {}).get("displayName", "")
            # Only singles draws
            if "singles" not in grouping_name.lower():
                continue

            for comp in grouping.get("competitions", []):
                # Filter by competition type (Men's Singles)
                comp_type_text = comp.get("type", {}).get("text", "").lower()
                if comp_type_text != wanted_type_text:
                    continue

                competitors = comp.get("competitors", [])
                if len(competitors) != 2:
                    continue

                status = comp.get("status", {}).get("type", {})
                completed = status.get("completed", False)

                p1, p2 = competitors[0], competitors[1]
                p1_name = p1.get("athlete", {}).get("displayName", "")
                p2_name = p2.get("athlete", {}).get("displayName", "")

                round_label = comp.get("round", {}).get("displayName", "")

                match = {
                    "date":       comp.get("date", "")[:10],
                    "round":      round_label,
                    "grouping":   grouping_name,
                    "completed":  completed,
                    "player1":    p1_name,
                    "player2":    p2_name,
                    "tournament": tournament_name,
                }

                if completed:
                    if p1.get("winner"):
                        match["winner"], match["loser"] = p1_name, p2_name
                    elif p2.get("winner"):
                        match["winner"], match["loser"] = p2_name, p1_name
                    else:
                        continue

                matches.append(match)

    return matches


# Backward compatibility aliases
def get_wimbledon_draw(tour="atp", date_range=None):
    """Deprecated: use get_atp_matches() instead."""
    return get_atp_matches(tour=tour, date_range=date_range)


def get_summer_major_draw(tour="atp", date_range=None):
    """Deprecated: use get_atp_matches() instead."""
    return get_atp_matches(tour=tour, date_range=date_range)


def get_player_recent_form(player_name, matches, n=15):
    player_matches = [
        m for m in matches
        if m["winner"] == player_name or m["loser"] == player_name
    ]
    player_matches.sort(key=lambda m: m["date"], reverse=True)
    return player_matches[:n]
