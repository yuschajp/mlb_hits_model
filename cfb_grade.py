"""Grade every logged board against CFBD finals.

    python3 cfb_grade.py              grade every ungraded picks_*.csv
    python3 cfb_grade.py 2026-09-10   one date
    python3 cfb_grade.py --regrade    redo everything

WHY CFBD AND NOT ESPN
    ESPN's public scoreboard returns 403 from a plain client, so the previous
    version silently graded nothing. CFBD needs the key you already have, and
    it is the same source the boards are built from -- which means team names
    match EXACTLY. The ESPN version had to fuzzy-match "Texas A&M" against
    "Texas A&M Aggies", and a fuzzy match that fails is a game that quietly
    never grades.

    Kickoffs are bucketed by EASTERN date, not UTC. CFBD returns UTC, so an
    8pm ET game reads as the next day and would be looked up under the wrong
    date -- the same bug that put Thursday's Miami game on Friday's board.
"""
import glob, json, os, sys, urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

KEY = os.environ.get("CFBD_KEY")
BASE = "https://api.collegefootballdata.com"
ET = ZoneInfo("America/New_York")
OUT = Path("data/cfb")


def get(path, **pm):
    if not KEY:
        sys.exit("Set CFBD_KEY first:  set -a; . ./.env; set +a")
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


def kick_date(g):
    raw = str(pick(g, "startDate", "start_date", default=""))
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(ET).date().isoformat()
    except ValueError:
        return raw[:10]


_cache = {}
def finals_for(date):
    """{(away, home): (away_pts, home_pts)} for every completed game that day."""
    if date in _cache:
        return _cache[date]
    year = int(date[:4])
    out = {}
    for wk in range(1, 17):
        try:
            gs = get("/games", year=year, week=wk, seasonType="regular")
        except Exception as e:
            print(f"    week {wk}: {e}")
            continue
        hit = [g for g in gs if kick_date(g) == date]
        if not hit:
            if out:
                break
            continue
        for g in hit:
            hp = pick(g, "homePoints", "home_points")
            ap = pick(g, "awayPoints", "away_points")
            h = pick(g, "homeTeam", "home_team")
            a = pick(g, "awayTeam", "away_team")
            if None in (hp, ap, h, a):
                continue
            out[(a, h)] = (float(ap), float(hp))
        if out:
            break
    _cache[date] = out
    return out


args = [x for x in sys.argv[1:] if not x.startswith("-")]
regrade = "--regrade" in sys.argv
files = ([f"data/cfb/picks_{d}.csv" for d in args] if args
         else sorted(glob.glob("data/cfb/picks_*.csv")))

frames, missed = [], []
for fp in files:
    date = Path(fp).stem.replace("picks_", "")
    gout = OUT / f"graded_{date}.csv"
    if gout.exists() and not regrade and not args:
        frames.append(pd.read_csv(gout)); continue
    try:
        p = pd.read_csv(fp)
    except Exception as e:
        print(f"  {fp}: {e}"); continue

    fin = finals_for(date)
    if not fin:
        print(f"  {date}: no completed games returned -- skipped"); continue

    ah, aa = [], []
    for _, r in p.iterrows():
        got = fin.get((r.away, r.home))
        if got is None:
            ah.append(np.nan); aa.append(np.nan)
            missed.append(f"{date}  {r.away} @ {r.home}")
        else:
            aa.append(got[0]); ah.append(got[1])
    p["date"] = date
    p["actual_home"], p["actual_away"] = ah, aa
    # SIGN CONVENTION -- the one thing to get right here.
    # model/market are SPREADS: negative means the home team is favoured.
    # So the actual result must use the SAME sign: negative when home wins.
    # Writing home-minus-away instead flips every game, turns a 70-point cover
    # into a 137-point "error", and grades wins as losses.
    p["actual_margin"] = p.actual_away - p.actual_home
    covered_home = p.actual_margin < p.market
    p["result"] = np.where(p.actual_margin.isna(), "",
                   np.where(p.actual_margin == p.market, "P",
                    np.where(((p.side == p.home) & covered_home) |
                             ((p.side == p.away) & ~covered_home), "W", "L")))
    p.to_csv(gout, index=False)
    w, l = int((p.result == "W").sum()), int((p.result == "L").sum())
    print(f"  graded {date}: {w}-{l}"
          + (f"-{int((p.result=='P').sum())}" if (p.result == 'P').any() else "")
          + f"  -> {gout.name}")
    frames.append(p)

if not frames:
    sys.exit("nothing graded")
if args:
    # a date was requested, but the SEASON line should still be the season --
    # otherwise grading one night prints "SEASON 0-1" and looks like a collapse
    for g in sorted(glob.glob("data/cfb/graded_*.csv")):
        if Path(g).stem.replace("graded_", "") not in args:
            try: frames.append(pd.read_csv(g))
            except Exception: pass
d = pd.concat(frames, ignore_index=True)
d = d[d.result.isin(["W", "L", "P"])]
if missed:
    print(f"\n  {len(missed)} games had no CFBD final (not played yet, or name drift):")
    for m in missed[:10]:
        print(f"    {m}")

def wilson(k, m, z=1.96):
    if not m: return (0, 0)
    ph = k/m; dd = 1+z*z/m; c = (ph+z*z/(2*m))/dd
    h = z*((ph*(1-ph)/m + z*z/(4*m*m))**0.5)/dd
    return c-h, c+h

W, L = int((d.result == "W").sum()), int((d.result == "L").sum())
N = W + L
print("\n" + "=" * 58)
print(f"SEASON  {W}-{L}  =  {W/N*100:.1f}%" if N else "SEASON  nothing decided")
if N:
    lo, hi = wilson(W, N); se = (0.25/N)**0.5
    print(f"        95% CI [{lo*100:.1f}%, {hi*100:.1f}%]   z vs 52.38% = {(W/N-0.5238)/se:+.2f}")
print("=" * 58)
def line(lab, s):
    w, l = int((s.result=="W").sum()), int((s.result=="L").sum())
    if w+l: print(f"   {lab:<26}{f'{w}-{l}':>10}{w/(w+l)*100:>8.1f}%{w+l:>6}")
print(f"   {'split':<26}{'record':>10}{'win%':>9}{'n':>6}")
print("   " + "-"*51)
if "cls" in d:
    for c in sorted(d.cls.dropna().unique()): line(f"class {c}", d[d.cls == c])
for k in ("FAV", "DOG"):
    if "dir" in d: line(f"{k} side", d[d['dir'] == k])
if "edge" in d:
    e = d.edge.abs()
    for a, b, lab in [(0,4,"edge under 4"), (4,9,"edge 4-9"), (9,99,"edge 9+")]:
        line(lab, d[(e >= a) & (e < b)])
v = d.dropna(subset=["model", "market", "actual_margin"])
if len(v):
    me = (v.model - v.actual_margin).abs().mean()
    ke = (v.market - v.actual_margin).abs().mean()
    print(f"\n   model MAE {me:.2f}   market MAE {ke:.2f}   "
          f"model is {me-ke:+.2f} vs market   (n={len(v)})")
print(f"\n   {int(N)} decided. ~1,400 needed to resolve a real edge.")
