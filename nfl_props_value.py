"""Over/under props WITH book lines and a BET / PASS verdict.

    python3 nfl_props_value.py

The fair-lines board tells you the model's number and leaves you to compare it
to your book. This pulls the actual posted lines and prices, converts both
sides to probabilities, and says what to do.

DEVIG HERE IS DIFFERENT FROM ANYTIME TD
    Over/under is a genuine TWO-WAY market: exactly one side wins. So the
    standard proportional devig is correct -- p_over / (p_over + p_under).
    (Anytime TD is not two-way, which is why that board needed a different
    normalisation.) Comparing against the raw over price instead would count
    the book's hold as edge and turn every line into a bet.

VERDICT
    BET OVER / BET UNDER only when the model's probability beats the devigged
    book probability by MIN_EDGE. Everything else is PASS -- which will be most
    of the board, because a 70-yard-wide outcome distribution swallows small
    disagreements.

COST: one credit per game per market. 3 markets x ~14 games = ~42.
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
MIN_EDGE = float(os.environ.get("PROP_MIN_EDGE", "0.05"))
DAYS = int(os.environ.get("PROP_DAYS", "3"))

MARKETS = {                      # odds-api key -> (stat, opportunity, min opp, shrink k)
    "player_receptions":    ("receptions",      "targets", 2.5, 25.0),
    "player_reception_yds": ("receiving_yards", "targets", 2.5, 40.0),
    "player_rush_yds":      ("rushing_yards",   "carries", 5.0, 40.0),
}
SEASONS = list(range(2018, 2027))
OPP_WINDOW, K_OPP, BINS = 4, 0.25, [0, 2, 4, 6, 9, 99]


def api(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r), r.headers.get("x-requests-remaining")


def norm(n):
    n = str(n).lower().replace(".", "").replace("'", "")
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"[^a-z]", "", n)


def implied(a):
    a = float(a)
    return (-a) / ((-a) + 100) if a < 0 else 100 / (a + 100)


# ---------- model side: empirical ratio distributions ----------------------
frames = []
for y in SEASONS:
    try:
        frames.append(pd.read_csv(f"data/nfl/player_week_{y}.csv", low_memory=False))
    except FileNotFoundError:
        pass
h = pd.concat(frames, ignore_index=True)
h = h[h.season_type.eq("REG") & h.position_group.isin(["RB", "WR", "TE"])].copy()
for c in ("carries", "targets", "receptions", "receiving_yards", "rushing_yards"):
    h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0.0)
h = h.sort_values(["player_id", "season", "week"])
gb = h.groupby("player_id", sort=False)
for c in ("carries", "targets", "receptions", "receiving_yards", "rushing_yards"):
    h[f"pr_{c}"] = gb[c].cumsum() - h[c]
for c in ("carries", "targets"):
    h[f"rec_{c}"] = (gb[c].shift(1).groupby(h.player_id)
                     .rolling(OPP_WINDOW, min_periods=1).mean()
                     .reset_index(level=0, drop=True))
h["games_prior"] = gb.cumcount()
pm = h.groupby("position_group")[["carries", "targets"]].transform("mean")
W = np.minimum(h.games_prior, OPP_WINDOW)
h["e_carries"] = (h.rec_carries.fillna(pm.carries) * W + K_OPP * pm.carries) / (W + K_OPP)
h["e_targets"] = (h.rec_targets.fillna(pm.targets) * W + K_OPP * pm.targets) / (W + K_OPP)
hist = h[h.games_prior >= 4]

RAT, RATE = {}, {}
for mk, (stat, opp, minopp, k) in MARKETS.items():
    mu_lg = hist[f"pr_{stat}"].sum() / max(hist[f"pr_{opp}"].sum(), 1)
    rate = (hist[f"pr_{stat}"] + k * mu_lg) / (hist[f"pr_{opp}"] + k)
    mu = hist[f"e_{opp}"] * rate
    sub = hist[(hist[f"e_{opp}"] >= minopp) & (mu > 0.05)].copy()
    sub["ratio"] = sub[stat] / mu[sub.index]
    sub["b"] = pd.cut(sub[f"e_{opp}"], BINS, labels=False, include_lowest=True)
    RAT[mk] = {b: np.sort(g.ratio.values) for b, g in sub.groupby("b") if len(g) > 200}
    RATE[mk] = (mu_lg, k)

cur = pd.read_csv("data/nfl/props_board.csv")
cur["key"] = cur.player.map(norm)
IDX = {(r.key, r.market): r for _, r in cur.iterrows()}

# ---------- book side ------------------------------------------------------
events, left = api(f"{BASE}/events?apiKey={KEY}")
cutoff = datetime.now(timezone.utc) + timedelta(days=DAYS)
ev = [e for e in events if e.get("commence_time")
      and datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= cutoff]
print(f"{len(ev)} games within {DAYS} days  (quota {left})  "
      f"-> ~{len(ev)*len(MARKETS)} credits")

rows, used = [], 0
for e in ev:
    try:
        o, left = api(f"{BASE}/events/{e['id']}/odds?apiKey={KEY}&regions=us"
                      f"&markets={','.join(MARKETS)}&oddsFormat=american")
        used += len(MARKETS)
    except urllib.error.HTTPError as err:
        print(f"  {e['away_team']} @ {e['home_team']}: HTTP {err.code}"); continue
    acc = {}
    for bk in o.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] not in MARKETS: continue
            for out in m.get("outcomes", []):
                nm, pt, pr = out.get("description"), out.get("point"), out.get("price")
                side = str(out.get("name", "")).lower()
                if not nm or pt is None or pr is None or side not in ("over", "under"):
                    continue
                d = acc.setdefault((norm(nm), m["key"], float(pt)),
                                   {"name": nm, "over": [], "under": []})
                d[side].append(implied(pr))
    for (k, mkkey, line), d in acc.items():
        if not d["over"] or not d["under"]: continue
        po, pu = float(np.median(d["over"])), float(np.median(d["under"]))
        p_book = po / (po + pu)                     # two-way devig -- correct here
        stat, opp, minopp, _ = MARKETS[mkkey]
        r = IDX.get((k, stat))
        if r is None: continue
        R = RAT[mkkey].get(int(r.bucket))
        if R is None: continue
        p_model = float((R * r.mu > line).mean())
        rows.append(dict(player=r.player, team=r.team, market=stat, line=line,
                         p_model=p_model, p_book=p_book, edge=p_model - p_book,
                         fair=r.fair, p25=r.p25, p75=r.p75,
                         game=f"{e['away_team'][:3]}@{e['home_team'][:3]}"))
print(f"  used ~{used} credits, {left} remaining")

if not rows:
    sys.exit("no prop lines matched")
d = pd.DataFrame(rows)
d["verdict"] = np.where(d.edge >= MIN_EDGE, "BET OVER",
                np.where(d.edge <= -MIN_EDGE, "BET UNDER", "PASS"))
d["abs_edge"] = d.edge.abs()
d = d.sort_values("abs_edge", ascending=False)
d.to_csv("data/nfl/props_value.csv", index=False)

# SANITY GATE. The book sets lines to be coin flips, so the model's mean
# P(over) across a full board must sit near 50%. If it does not, the model's
# LEVEL is wrong and every verdict is an artefact of that, not an edge.
# The PIT test validated the distribution's SHAPE given the model's mean --
# it never validated the mean itself, which is where this model's error lives.
_mp = d.p_model.mean()
if abs(_mp - 0.50) > 0.03:
    d["verdict"] = "PASS (model level unreliable)"
    print(f"\n*** SANITY GATE TRIPPED ***")
    print(f"    mean model P(over) = {_mp*100:.1f}%, should be ~50% vs book lines.")
    print(f"    mean fair-minus-line = {(d.fair-d.line).mean():+.1f}")
    print(f"    The per-player LEVEL is off, so no verdict is trustworthy.")
    print(f"    All rows forced to PASS. Fix usage estimation before betting this.\n")

bets = d[~d.verdict.str.startswith("PASS")]
print(f"\n{len(d)} priced player-markets   "
      f"{len(bets)} BET / {len(d)-len(bets)} PASS   (floor {MIN_EDGE*100:.0f}pp)\n")
print(f"{'player':<22}{'market':<17}{'line':>7}{'model':>8}{'book':>8}{'edge':>8}  verdict")
print("-" * 84)
for _, r in d.head(22).iterrows():
    print(f"{r.player[:21]:<22}{r.market:<17}{r.line:>7.1f}{r.p_model*100:>7.1f}%"
          f"{r.p_book*100:>7.1f}%{r.edge*100:>+7.1f}  {r.verdict}")
print(f"\nmean edge {d.edge.mean()*100:+.2f}pp   {(d.edge>0).mean()*100:.0f}% positive")
print("A mean far from zero means a normalisation problem, not an edge.")
