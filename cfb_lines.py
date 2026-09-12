"""Pull CFB lines from CFBD and write the board's lines file. No pasting.

    export CFBD_KEY=...
    python3 cfb_lines.py 2026-09-12

Writes data/cfb/lines_<date>.txt in the format cfb_board2.py expects:
    Away, Home, home_spread[, N]        negative = home favoured, N = neutral

Uses the MEDIAN spread across books rather than one book's number -- a single
outlier line is the most common way a board ends up with a fake 30-point edge.
Exits 0 with no file when there are no games, so the daily runner just skips.
"""
import json, os, sys, urllib.request
from datetime import date as _date, datetime
from zoneinfo import ZoneInfo

# CFBD returns startDate in UTC. Comparing the raw string puts every kickoff
# after 8pm ET on the NEXT day's board -- which silently split one slate across
# two files and left night games ungraded. College football is scheduled in
# Eastern, so convert before comparing.
ET = ZoneInfo("America/New_York")


def kick_et(g):
    """Eastern kickoff as HH:MM, or '' if unknown."""
    raw = str(pick(g, "startDate", "start_date", default=""))
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(ET).strftime("%H:%M")
    except ValueError:
        return ""


def kick_date(g):
    raw = str(g.get("startDate") or g.get("start_date") or "")
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(ET).date().isoformat()
    except ValueError:
        return raw[:10]
from pathlib import Path
from statistics import median

KEY = os.environ.get("CFBD_KEY")
BASE = "https://api.collegefootballdata.com"
DATE = sys.argv[1] if len(sys.argv) > 1 else _date.today().isoformat()
YEAR = int(DATE[:4])
OUT = Path(f"data/cfb/lines_{DATE}.txt")


def get(path, **pm):
    if not KEY:
        sys.exit("Set CFBD_KEY first")
    q = "&".join(f"{k}={v}" for k, v in pm.items() if v is not None)
    req = urllib.request.Request(f"{BASE}{path}?{q}", headers={
        "Authorization": f"Bearer {KEY}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def pick(d, *names, default=None):
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return default


# Which week is this date in? Ask CFBD rather than computing it.
weeks, games = set(), []
for wk in range(1, 17):
    try:
        gs = get("/games", year=YEAR, week=wk, seasonType="regular")
    except Exception:
        continue
    hit = [g for g in gs if kick_date(g) == DATE]
    if hit:
        weeks.add(wk); games += hit
    if weeks and wk > max(weeks) + 1:
        break

if not games:
    print(f"no CFB games on {DATE} -- no lines file written")
    sys.exit(0)

spreads = {}
for wk in sorted(weeks):
    try:
        for gm in get("/lines", year=YEAR, week=wk, seasonType="regular"):
            vals = [float(l["spread"]) for l in (gm.get("lines") or [])
                    if l.get("spread") is not None]
            if gm.get("id") and vals:
                spreads[gm["id"]] = median(vals)
    except Exception as e:
        print(f"  week {wk} lines failed: {e}")

rows, unpriced = [], 0
for g in games:
    gid = pick(g, "id", "gameId")
    h, a = pick(g, "homeTeam", "home_team"), pick(g, "awayTeam", "away_team")
    sp = spreads.get(gid)
    if not (h and a):
        continue
    if sp is None:
        unpriced += 1; continue
    neutral = bool(pick(g, "neutralSite", "neutral_site", default=False))
    t = kick_et(g)
    extra = ("" if not neutral else ", N") + (f", {t}" if t else "")
    rows.append(f"{a}, {h}, {sp}{extra}")

if not rows:
    print(f"{len(games)} games on {DATE} but none priced yet -- no file written")
    sys.exit(0)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(f"# CFB {DATE} -- auto-pulled from CFBD, median spread across books\n"
               "# Away, Home, home_spread[, N][, HH:MM ET]  negative = home favoured\n"
               + "\n".join(rows) + "\n")
print(f"{len(rows)} priced games -> {OUT}"
      + (f"   ({unpriced} not yet priced, skipped)" if unpriced else ""))
