"""Turn the anytime-TD board from BEST CHANCE into VALUE.

    python3 nfl_td_value.py

The board ranks by P(TD). That answers "who is most likely to score", which is
not the same question as "which price is wrong". This pulls real anytime-TD
prices from the-odds-api, converts them to no-vig implied probabilities, and
ranks by the gap.

DEVIGGING IS NOT OPTIONAL
    A book's quoted implied probabilities sum to well over 100%. Comparing the
    model against the RAW implied number counts the book's margin as edge and
    manufactures value on every single player. Each game's anytime-TD market is
    normalised so the quoted probabilities sum to the expected number of TD
    scorers implied by the market itself -- then the comparison is like-for-like.

COST: one credit per game per market. ~14 a week against a 500 balance.
"""
import json, os, re, sys, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

KEY = os.environ.get("ODDS_API_KEY")
if not KEY:
    sys.exit("ODDS_API_KEY not set:  set -a; . ./.env; set +a")
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MIN_EDGE = float(os.environ.get("TD_MIN_EDGE", "0.04"))   # 4 percentage points


def api(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r), r.headers.get("x-requests-remaining")


def norm(n):
    n = str(n).lower().replace(".", "").replace("'", "")
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"[^a-z]", "", n)


TD_A, TD_B = -0.651, 0.1361        # team offensive TDs = a + b * implied points

# game totals, for the devig target -- read from the lines file we already pull
TOTALS = {}
try:
    NAME = {}
    for ln in Path("data/nfl/lines_current.txt").read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = [x.strip() for x in ln.split(",")]
        NAME[(f[0], f[1])] = float(f[3])
except Exception:
    NAME = {}
ABBR = {"Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL",
 "Buffalo Bills":"BUF","Carolina Panthers":"CAR","Chicago Bears":"CHI",
 "Cincinnati Bengals":"CIN","Cleveland Browns":"CLE","Dallas Cowboys":"DAL",
 "Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
 "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX",
 "Kansas City Chiefs":"KC","Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC",
 "Los Angeles Rams":"LA","Miami Dolphins":"MIA","Minnesota Vikings":"MIN",
 "New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
 "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT",
 "San Francisco 49ers":"SF","Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB",
 "Tennessee Titans":"TEN","Washington Commanders":"WAS"}


def implied(american):
    a = float(american)
    return (-a) / ((-a) + 100) if a < 0 else 100 / (a + 100)


DAYS = int(os.environ.get("TD_DAYS", "3"))
events, left = api(f"{BASE}/events?apiKey={KEY}")
cutoff = datetime.now(timezone.utc) + timedelta(days=DAYS)
kept = []
for ev in events:
    ct = ev.get("commence_time")
    try:
        if ct and datetime.fromisoformat(ct.replace("Z", "+00:00")) > cutoff:
            continue
    except ValueError:
        pass
    kept.append(ev)
print(f"{len(events)} upcoming events, {len(kept)} within {DAYS} days "
      f"(quota left {left})")
print(f"  -> this will spend ~{len(kept)} credits, one per game")
events = kept

rows, used = [], 0
for ev in events:
    try:
        o, left = api(f"{BASE}/events/{ev['id']}/odds?apiKey={KEY}&regions=us"
                      f"&markets=player_anytime_td&oddsFormat=american")
        used += 1
    except urllib.error.HTTPError as e:
        print(f"  {ev.get('away_team')} @ {ev.get('home_team')}: HTTP {e.code}")
        continue
    per_player = {}
    for bk in o.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != "player_anytime_td":
                continue
            for out in mk.get("outcomes", []):
                nm = out.get("description") or out.get("name")
                if out.get("price") is None or not nm:
                    continue
                per_player.setdefault(norm(nm), {"name": nm, "p": []})["p"].append(
                    implied(out["price"]))
    if not per_player:
        continue
    med = {k: (v["name"], float(np.median(v["p"]))) for k, v in per_player.items()}
    tot = sum(p for _, p in med.values())
    if tot <= 0:
        continue
    # DEVIG. Anytime-TD prices across a game sum to far more than the true
    # expected number of distinct scorers -- that excess is the hold, and
    # treating it as edge manufactures value on every player. Normalising to
    # 1.0 (the usual proportional devig) is WRONG here: this is not a
    # mutually-exclusive market, several players score in a game.
    # Target instead the expected scorers implied by the game's own TOTAL:
    #     team TDs   = TD_A + TD_B * implied team points   (fitted on 4,254 games)
    #     scorers    = 0.857 x TDs  (multi-TD games mean scorers < touchdowns;
    #                  ratio measured on this model's own allocation check)
    tl = NAME.get((ABBR.get(ev["away_team"], ""), ABBR.get(ev["home_team"], "")))
    if tl:
        exp_tds = 2 * (TD_A + TD_B * (tl / 2.0))
        target = 0.857 * exp_tds
        scale = target / tot
    else:
        scale = 1.0            # no total available -> leave raw, and say so
    for k, (nm, p) in med.items():
        rows.append(dict(key=k, book_name=nm,
                         matchup=f"{ev['away_team']} @ {ev['home_team']}",
                         p_book_raw=p, p_book=min(p * scale, 0.98),
                         devigged=bool(tl)))
print(f"  used {used} credits, {left} remaining")

if not rows:
    sys.exit("no anytime-TD prices returned")
mk = pd.DataFrame(rows)
bd = pd.read_csv("data/nfl/td_board_w1.csv")
bd["key"] = bd.player_display_name.map(norm)
m = bd.merge(mk, on="key", how="inner", suffixes=("", "_mkt"))
# LEVEL vs ALLOCATION.
# The model splits each team's touchdowns across ITS player list; the book uses
# a longer one. On the matched subset the model's mass ran 1.15x the book's,
# giving every player a fake +2.3pp and flagging 37% of the market as value.
# The model's claim is about WHO scores, not how many -- so normalise both
# sides to the same per-game total and compare only the allocation.
_mt = m.groupby("matchup").p_td.transform("sum")
_bt = m.groupby("matchup").p_book.transform("sum")
m["p_model_adj"] = m.p_td * (_bt / _mt)
m["edge"] = m.p_model_adj - m.p_book
m["level_ratio"] = _mt / _bt
m["ev"] = m.p_model_adj / m.p_book - 1
m = m.sort_values("edge", ascending=False)

unmatched = len(bd) - len(m)
print(f"\nmatched {len(m)} of {len(mk)} priced players "
      f"({unmatched} board rows had no price)\n")
print(f"{'player':<24}{'tm':<5}{'game':<26}{'model':>8}{'book':>8}{'edge':>8}{'EV':>8}")
print("-" * 94)
for _, r in m.head(20).iterrows():
    star = "  <<" if r.edge >= MIN_EDGE else ""
    print(f"{r.player_display_name[:23]:<24}{r.team:<5}{r.matchup[:25]:<26}"
          f"{r.p_model_adj*100:>7.1f}%{r.p_book*100:>7.1f}%{r.edge*100:>+7.1f}{r.ev*100:>+7.1f}%{star}")
print("\nBOTTOM (model says the book is too high):")
for _, r in m.tail(5).iterrows():
    print(f"{r.player_display_name[:23]:<24}{r.team:<5}{r.matchup[:25]:<26}"
          f"{r.p_model_adj*100:>7.1f}%{r.p_book*100:>7.1f}%{r.edge*100:>+7.1f}{r.ev*100:>+7.1f}%")
m.to_csv("data/nfl/td_value.csv", index=False)
n_play = int((m.edge >= MIN_EDGE).sum())
print(f"\n{n_play} players clear the {MIN_EDGE*100:.0f}pp edge floor  "
      f"-> data/nfl/td_value.csv")
print("\nThis is the FIRST time any model in this stack has been compared to a")
print("real price. Log it and grade it -- the edge column is a claim, not a fact.")
