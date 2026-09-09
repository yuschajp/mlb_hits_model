"""Pull NFL lines and write the TD board's lines file.

    python3 nfl_lines.py

Writes data/nfl/lines_current.txt:
    away, home, home_spread, total[, N]     POSITIVE = home favoured

SOURCE: the-odds-api, via ODDS_API_KEY in .env. Chosen over the two alternatives
after both failed in practice:
  * nflverse games.csv backfills -- 0 of 16 week-1 lines at time of writing.
    It stays the source for HISTORY and validation, where you want a CLOSING
    line, but it cannot tell you today's number.
  * ESPN's public scoreboard returns 403 from a plain client.
The Odds API is a licensed feed with a documented quota, which is the right way
to consume odds anyway.

MEDIAN ACROSS BOOKS, not one book. A single outlier line is the most common way
a board grows a fake edge, and the model is not good enough to distinguish an
outlier from an opportunity.
"""
import json, os, sys, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

# The /odds endpoint returns EVERY upcoming game, not this week's. Left
# unfiltered that is ~270 rows, and since the board joins players to games by
# team, every player would be duplicated across ~17 games. Window it.
DAYS = int(os.environ.get("NFL_LINE_DAYS", "7"))

KEY = os.environ.get("ODDS_API_KEY")
if not KEY:
    sys.exit("ODDS_API_KEY not set. Run from run_all_daily.sh, or: set -a; . ./.env; set +a")

URL = ("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
       f"?apiKey={KEY}&regions=us&markets=spreads,totals&oddsFormat=american")
OUT = Path("data/nfl/lines_current.txt")

ABBR = {
 "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL",
 "Buffalo Bills":"BUF","Carolina Panthers":"CAR","Chicago Bears":"CHI",
 "Cincinnati Bengals":"CIN","Cleveland Browns":"CLE","Dallas Cowboys":"DAL",
 "Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
 "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX",
 "Kansas City Chiefs":"KC","Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC",
 "Los Angeles Rams":"LA","Miami Dolphins":"MIA","Minnesota Vikings":"MIN",
 "New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
 "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT",
 "San Francisco 49ers":"SF","Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB",
 "Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

req = urllib.request.Request(URL, headers={"Accept": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        events = json.load(r)
        left = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
except urllib.error.HTTPError as e:
    body = e.read()[:300].decode("utf-8", "replace")
    sys.exit(f"Odds API HTTP {e.code}: {body}")

cutoff = datetime.now(timezone.utc) + timedelta(days=DAYS)
rows, unmapped, unpriced, seen, dropped = [], set(), [], set(), 0
for ev in events:
    ct = ev.get("commence_time")
    if ct:
        try:
            if datetime.fromisoformat(ct.replace("Z", "+00:00")) > cutoff:
                dropped += 1; continue
        except ValueError:
            pass
    h, a = ev.get("home_team"), ev.get("away_team")
    hk, ak = ABBR.get(h), ABBR.get(a)
    if not hk or not ak:
        unmapped.add(h if not hk else a); continue
    sp, tt = [], []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") == "spreads":
                for o in mk.get("outcomes", []):
                    if o.get("name") == h and o.get("point") is not None:
                        sp.append(-float(o["point"]))     # book: home -3 -> we want +3
            elif mk.get("key") == "totals":
                for o in mk.get("outcomes", []):
                    if o.get("name") == "Over" and o.get("point") is not None:
                        tt.append(float(o["point"]))
    if not sp or not tt:
        unpriced.append(f"{ak}@{hk}"); continue
    if hk in seen or ak in seen:          # a team can only play once per slate
        continue
    seen.add(hk); seen.add(ak)
    rows.append(f"{ak}, {hk}, {median(sp)}, {median(tt)}")

if not rows:
    print("no priced NFL games returned -- lines file not written")
    sys.exit(0)

OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.is_symlink():
    OUT.unlink()
OUT.write_text("# NFL -- auto-pulled from the-odds-api, median across books\n"
               "# away, home, home_spread, total   POSITIVE = home favoured\n"
               + "\n".join(rows) + "\n")
print(f"{len(rows)} priced games in the next {DAYS} days -> {OUT}")
if dropped: print(f"  {dropped} games beyond the window, skipped")
if left:      print(f"  quota: {used} used, {left} remaining this period")
if unpriced:  print(f"  {len(unpriced)} unpriced: {', '.join(unpriced[:8])}")
if unmapped:  print(f"  !! UNMAPPED TEAM NAMES {sorted(unmapped)} -- add to ABBR")
