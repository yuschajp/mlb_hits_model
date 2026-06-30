"""
tennis_data_client.py

Two data sources:

1. HISTORICAL MATCHES (for Elo computation) -- pulled from a local clone
   of the TML-Database repo (Tennismylife/TML-Database), an actively
   maintained fork of Jeff Sackmann's tennis data with the same column
   schema. Sackmann's original raw.githubusercontent.com URLs were
   returning 404s as of this writing, so we use TML-Database instead,
   cloned locally via:

       git clone https://github.com/Tennismylife/TML-Database.git ~/Desktop/tennis_data

2. LIVE WIMBLEDON MATCHES (today's draw and results) -- pulled from
   ESPN's public (unofficial) tennis API, which has no auth requirement:

       https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard
       https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard
"""

import csv
import requests
from datetime import date
from pathlib import Path

TENNIS_DATA_DIR = Path.home() / "Desktop" / "tennis_data"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"
TIMEOUT = 15
_cache = {}


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
    tour is accepted for API compatibility but TML-Database is ATP-only --
    WTA calls will return empty until a WTA equivalent is wired up.
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


# ── Live Wimbledon matches (ESPN) ──────────────────────────────────────────────

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


def get_wimbledon_draw(tour="atp", date_range=None):
    """
    Returns Wimbledon matches from ESPN.

    ESPN's tennis scoreboard returns ONE event per Slam (e.g. "Wimbledon"),
    with individual matches nested under event['groupings'][i]['competitions'],
    where each grouping corresponds to a draw (Men's Singles, Women's Singles,
    Men's Doubles, etc). We flatten all singles competitions across all
    groupings into a single match list.

    Returns list of dicts: {date, round, completed, player1, player2,
                             winner, loser (if completed)}
    """
    league = "atp" if tour == "atp" else "wta"
    data = _espn_get(league, "scoreboard", {})
    events = data.get("events", [])

    wimbledon_event = next(
        (e for e in events if "wimbledon" in e.get("name", "").lower()), None
    )
    if not wimbledon_event:
        return []

    matches = []
    for grouping in wimbledon_event.get("groupings", []):
        grouping_name = grouping.get("grouping", {}).get("displayName", "")
        # Only singles draws -- skip doubles/mixed for the Elo model
        if "singles" not in grouping_name.lower():
            continue

        for comp in grouping.get("competitions", []):
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            status = comp.get("status", {}).get("type", {})
            completed = status.get("completed", False)

            p1, p2 = competitors[0], competitors[1]
            p1_name = p1.get("athlete", {}).get("displayName", "")
            p2_name = p2.get("athlete", {}).get("displayName", "")

            notes = comp.get("notes", [])
            round_label = notes[0].get("text", "") if notes else ""

            match = {
                "date":      comp.get("date", "")[:10],
                "round":     round_label,
                "grouping":  grouping_name,
                "completed": completed,
                "player1":   p1_name,
                "player2":   p2_name,
            }

            if completed:
                if p1.get("winner"):
                    match["winner"], match["loser"] = p1_name, p2_name
                elif p2.get("winner"):
                    match["winner"], match["loser"] = p2_name, p1_name
                else:
                    continue  # walkover/no winner flagged, skip

            matches.append(match)

    return matches


def get_player_recent_form(player_name, matches, n=15):
    player_matches = [
        m for m in matches
        if m["winner"] == player_name or m["loser"] == player_name
    ]
    player_matches.sort(key=lambda m: m["date"], reverse=True)
    return player_matches[:n]
