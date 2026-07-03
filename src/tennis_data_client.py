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

   NOTE: TML-Database is ATP-only. get_recent_matches(tour="wta") returns
   an empty list -- there is currently no WTA historical data source
   wired up. This means WTA Elo ratings default to INITIAL_ELO (1500)
   for every player, producing meaningless 50/50 predictions rather than
   real ones. This was caught via a cluster of exact-1500/1500/50.0%
   predictions on the dashboard for WTA players. Fixing this properly
   needs a real WTA-equivalent dataset -- flagging clearly rather than
   silently guessing, since a fake "no opinion" prediction is worse than
   an honest gap.

2. LIVE WIMBLEDON MATCHES (today's draw and results) -- pulled from
   ESPN's public (unofficial) tennis API, which has no auth requirement:

       https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard
       https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard

--- Bug fixes (found via live API inspection) ---

1. ROUND FIELD: previously read from comp["notes"][0]["text"], which is
   actually a plain-English match recap string (e.g. "Zsombor Piros
   (HUN) bt Ivan Ivanov (BUL) 6-2 6-2"), not a round label at all. The
   real round name is at comp["round"]["displayName"] (e.g. "Qualifying
   1st Round"), confirmed by dumping the raw ESPN response.

2. TOUR/GENDER MISLABELING: both the "atp" and "wta" league URLs return
   the SAME combined Wimbledon event, with ALL groupings (Men's Singles,
   Women's Singles, Men's Doubles, Women's Doubles, Mixed Doubles)
   bundled together regardless of which league endpoint you call. The
   previous code only filtered "singles" vs "doubles" by grouping name,
   then labeled every match with whatever `tour` string the CALLER
   passed in -- meaning a call with tour="atp" pulled in real WTA
   players' matches too, mislabeled as "atp". Fixed by filtering on each
   competition's own comp["type"]["text"] field (e.g. "Men's Singles" /
   "Women's Singles"), confirmed present per-competition in the raw
   response, instead of trusting the caller's label.
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

    tour="wta" currently returns an empty list -- see module docstring.
    This means WTA predictions get NO real Elo signal right now. Do not
    remove this warning without wiring up a real WTA data source; the
    resulting empty-list-fallback silently producing "confident-looking"
    50/50 predictions caused a real, hard-to-spot dashboard bug.
    """
    if tour != "atp":
        print(f"  [warn] TML-Database is ATP-only. No WTA data source configured yet.")
        print(f"  [warn] WTA Elo ratings will default to 1500 for every player -- "
              f"predictions will be uninformative 50/50 splits, not real forecasts.")
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
    Returns Wimbledon SINGLES matches from ESPN for the requested tour.

    IMPORTANT: ESPN's "atp" and "wta" scoreboard endpoints return the
    SAME combined Wimbledon event -- all groupings (Men's/Women's
    Singles, Men's/Women's/Mixed Doubles) come back regardless of which
    league URL is called. This function filters to singles matches for
    the SPECIFIC requested tour using each competition's own
    comp["type"]["text"] field (e.g. "Men's Singles"), not just the
    grouping name -- confirmed via direct inspection of the live API
    response that this field is reliably present per-competition.

    Returns list of dicts: {date, round, completed, player1, player2,
                             winner, loser (if completed)}
    """
    if tour not in TOUR_TYPE_TEXT:
        raise ValueError(f"tour must be one of {list(TOUR_TYPE_TEXT)}, got {tour!r}")
    wanted_type_text = TOUR_TYPE_TEXT[tour]

    # Either league URL returns the same combined event, so "atp" works
    # as the fetch target regardless of which tour was requested.
    data = _espn_get("atp", "scoreboard", {})
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
            # Filter by the competition's OWN type field, not just the
            # grouping name -- this is what actually distinguishes
            # Men's from Women's Singles reliably, and is what was
            # missing before (grouping-level filtering only excluded
            # doubles, never separated the two singles draws).
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

            # Real round label, e.g. "Qualifying 1st Round", "Quarterfinal".
            # comp["notes"][0]["text"] (the old source) is a plain-English
            # match recap, not a round name -- see module docstring.
            round_label = comp.get("round", {}).get("displayName", "")

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
