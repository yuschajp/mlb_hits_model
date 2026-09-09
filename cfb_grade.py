"""Grade every logged board against real finals. No API key -- ESPN's public
scoreboard only.

    python3 cfb_grade.py              grade every ungraded picks_*.csv
    python3 cfb_grade.py 2026-09-12   grade one date
    python3 cfb_grade.py --regrade    redo everything

For each data/cfb/picks_<DATE>.csv it writes data/cfb/graded_<DATE>.csv adding
actual_home, actual_away, actual_margin (home perspective) and result (W/L/P),
then prints a season summary broken out the way week 1 said matters: FBS vs FCS,
favourite vs underdog, and edge bucket.

WHY THIS EXISTS
  Resolving a true 55% edge at 95% confidence takes roughly 1,400 games. Every
  result graded by hand is a game you will not get to. This turns grading into
  something that happens automatically after every slate.
"""
import glob, json, re, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

ESPN = ("https://site.api.espn.com/apis/site/v2/sports/football/"
        "college-football/scoreboard?dates={d}&groups=80&limit=900")


def norm(s):
    """Loose key so 'Texas A&M', 'Texas A&M Aggies' and 'texas am' all agree."""
    s = str(s).lower()
    s = s.replace("&", "and").replace("'", "").replace(".", "")
    s = re.sub(r"\b(state)\b", "st", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def fetch(date):
    """Return {normalised team key: (home_pts, away_pts, home_key, away_key)}."""
    url = ESPN.format(d=date.replace("-", ""))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as f:
        js = json.load(f)
    out = {}
    for ev in js.get("events", []):
        for comp in ev.get("competitions", []):
            st = (comp.get("status") or {}).get("type") or {}
            if not st.get("completed"):
                continue
            sides = {}
            for c in comp.get("competitors", []):
                t = c.get("team") or {}
                names = {t.get("displayName"), t.get("shortDisplayName"),
                         t.get("location"), t.get("name"), t.get("nickname")}
                sides[c.get("homeAway")] = (
                    [norm(n) for n in names if n], int(c.get("score") or 0))
            if "home" not in sides or "away" not in sides:
                continue
            (hk, hp), (ak, ap) = sides["home"], sides["away"]
            for a in hk:
                for b in ak:
                    out[(a, b)] = (hp, ap)
    return out


def look(idx, home, away):
    h, a = norm(home), norm(away)
    if (h, a) in idx:
        return idx[(h, a)]
    for (kh, ka), v in idx.items():          # substring fallback
        if (kh.startswith(h) or h.startswith(kh)) and \
           (ka.startswith(a) or a.startswith(ka)):
            return v
    return None


args = [x for x in sys.argv[1:] if not x.startswith("-")]
regrade = "--regrade" in sys.argv
files = ([f"data/cfb/picks_{d}.csv" for d in args] if args
         else sorted(glob.glob("data/cfb/picks_*.csv")))

frames, missed = [], []
for fp in files:
    date = Path(fp).stem.replace("picks_", "")
    gout = Path(f"data/cfb/graded_{date}.csv")
    if gout.exists() and not regrade and not args:
        frames.append(pd.read_csv(gout));  continue
    try:
        p = pd.read_csv(fp)
    except Exception as e:
        print(f"  {fp}: {e}");  continue
    try:
        idx = fetch(date)
    except Exception as e:
        print(f"  {date}: ESPN fetch failed ({e}) -- skipped");  continue

    hp, ap = [], []
    for _, r in p.iterrows():
        got = look(idx, r.home, r.away)
        if got is None:
            hp.append(np.nan); ap.append(np.nan)
            missed.append(f"{date}  {r.away} @ {r.home}")
        else:
            hp.append(got[0]); ap.append(got[1])
    p["date"] = date
    p["actual_home"], p["actual_away"] = hp, ap
    p["actual_margin"] = p.actual_home - p.actual_away      # home perspective
    covered_home = p.actual_margin < p.market               # market is home-persp
    p["result"] = np.where(p.actual_margin.isna(), "",
                   np.where(p.actual_margin == p.market, "P",
                    np.where(((p.side == p.home) & covered_home) |
                             ((p.side == p.away) & ~covered_home), "W", "L")))
    p.to_csv(gout, index=False)
    print(f"  graded {date}: {(p.result=='W').sum()}-{(p.result=='L').sum()}"
          + (f"-{(p.result=='P').sum()}" if (p.result=='P').any() else "")
          + f"   -> {gout}")
    frames.append(p)
    time.sleep(0.4)

if not frames:
    sys.exit("nothing graded")
d = pd.concat(frames, ignore_index=True)
d = d[d.result.isin(["W", "L", "P"])]
if missed:
    print(f"\n  {len(missed)} games had no ESPN match (check team spelling):")
    for m in missed[:12]:
        print(f"    {m}")

def line(lab, s):
    w, l, p = (s.result == "W").sum(), (s.result == "L").sum(), (s.result == "P").sum()
    n = w + l
    rec = f"{w}-{l}" + (f"-{p}" if p else "")
    pct = f"{w/n*100:.1f}%" if n else "  -  "
    print(f"   {lab:<26}{rec:>10}{pct:>9}{n:>6}")

def wilson(k, m, z=1.96):
    if not m: return (0, 0)
    ph = k / m; dd = 1 + z*z/m; c = (ph + z*z/(2*m))/dd
    h = z*((ph*(1-ph)/m + z*z/(4*m*m))**0.5)/dd
    return c-h, c+h

W, L = (d.result == "W").sum(), (d.result == "L").sum()
N = W + L
lo, hi = wilson(W, N); se = (0.25/N)**0.5 if N else float("nan")
print("\n" + "=" * 58)
print(f"SEASON  {W}-{L}  =  {W/N*100:.1f}%" if N else "SEASON  no decided games")
if N:
    print(f"        95% CI [{lo*100:.1f}%, {hi*100:.1f}%]   "
          f"z vs 52.38% break-even = {(W/N-0.5238)/se:+.2f}")
print("=" * 58)
print(f"   {'split':<26}{'record':>10}{'win%':>9}{'n':>6}")
print("   " + "-" * 51)
if "cls" in d:
    for c in sorted(d.cls.dropna().unique()):
        line(f"class {c}", d[d.cls == c])
for k in ("FAV", "DOG"):
    if "dir" in d: line(f"{k} side", d[d['dir'] == k])
if "edge" in d:
    e = d.edge.abs()
    for a, b, lab in [(0,4,"edge under 4"), (4,9,"edge 4-9"), (9,99,"edge 9+")]:
        line(lab, d[(e >= a) & (e < b)])
if "spread_size" in d:
    line("spread >= 22", d[d.spread_size >= 22])
    line("spread < 22",  d[d.spread_size < 22])

if {"model", "actual_margin", "market"} <= set(d.columns):
    v = d.dropna(subset=["model", "market", "actual_margin"])
    if len(v):
        me = (v.model - v.actual_margin).abs().mean()
        ke = (v.market - v.actual_margin).abs().mean()
        print(f"\n   model MAE {me:.2f}   market MAE {ke:.2f}   "
              f"model is {me-ke:+.2f} vs market   (n={len(v)})")
        sl = (v.model * v.actual_margin).sum() / (v.model ** 2).sum()
        print(f"   scale slope (actual on model) = {sl:.3f}   "
              f"{'-> still compressed' if sl > 1.08 else ''}")
print(f"\n   {int(N)} decided. ~1,400 needed to resolve a real edge.")
