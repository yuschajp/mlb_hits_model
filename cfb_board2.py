"""Board generator, v2. Replaces cfb_board.py -- no patching, drop it in beside.

WHAT CHANGED FROM v1, AND WHY
  1. NEUTRAL SITES.  v1 applied +2.0 home-field to every game including ones at
     Lambeau and Nissan Stadium. Mark a game neutral with a 4th field "N".
  2. MINIMUM EDGE.   Sub-4-point disagreements went 0-3-1 in week 1 and are
     inside the model's own 15-point MAE. They now print as REFERENCE, not plays.
  3. SCALE FACTOR.   Week-1 FBS games regressed actual-on-predicted at slope
     1.457, i.e. the ratings are compressed in the preseason-prior regime.
     Set SCALE from cfb_earlyscale.py. It multiplies the TEAM-RATING difference
     only -- never the FBS/FCS OFFSET (a fixed measured constant, not a shrunk
     estimate) and never HFA (measured directly in the fit).
  4. CLASS COLUMN.   FBS-vs-FCS is now tracked separately, because week 1 said
     the model beat the market there and lost to it everywhere else.

LINES FILE   data/cfb/lines_<DATE>.txt, one game per line:
     Away Team, Home Team, -11.5          home-perspective, negative = home fav
     Away Team, Home Team, -20.5, N       neutral site, no HFA
     Away Team @ Home Team -11.5          also accepted
  Blank lines and # comments ignored.
"""
import re, sys
import numpy as np, pandas as pd
from pathlib import Path

# ---------------------------------------------------------------- knobs -----
SCALE     = 1.00   # <-- set from cfb_earlyscale.py. 1.00 = no correction.
MIN_EDGE  = 4.0    # below this a disagreement is noise, not a bet
MAX_EDGE  = 25.0   # above this it is almost certainly an OFFSEASON CHANGE the
                   # 2025 ratings cannot see (new staff, portal haul, departures),
                   # not an edge. Flagged, not filtered -- it still logs and grades
                   # so we find out whether the flag is actually predictive.
OFFSET    = 32.6   # FBS -> FCS bridge, median of 126 bridge games (mean 31.2 +/- 2.4)
HFA       = 2.0    # home-field, points
# -----------------------------------------------------------------------------

DATE = sys.argv[1] if len(sys.argv) > 1 else None
if not DATE:
    sys.exit("usage: python3 cfb_board2.py YYYY-MM-DD")
SRC = Path(f"data/cfb/lines_{DATE}.txt")
if not SRC.exists():
    SRC.parent.mkdir(parents=True, exist_ok=True)
    SRC.write_text("# Away, Home, home_line[, N]   negative = home favoured, N = neutral\n")
    sys.exit(f"created {SRC} -- paste the slate into it, then re-run")

R = {d: pd.read_csv(f"data/cfb/ratings_2025_{d}.csv", index_col=0) for d in ("fbs", "fcs")}
_a = pd.read_csv("data/cfb/games_2025_all.csv")
gp = pd.concat([_a.home_team, _a.away_team]).value_counts()

base, div, weak = {}, {}, set()
for d in ("fbs", "fcs"):
    for t, v in R[d].net.items():
        base[t], div[t] = v, d
    weak |= set(R[d].net[R[d].net <= R[d].net.quantile(0.20)].index)
off = {t: (OFFSET if div[t] == "fbs" else 0.0) for t in base}


def find(name):
    n = name.strip()
    if n in base:
        return n
    low = n.lower()
    for t in base:
        if t.lower() == low:
            return t
    hits = [t for t in base if low in t.lower() or t.lower() in low]
    return hits[0] if len(hits) == 1 else None


rows, bad = [], []
for raw in SRC.read_text().splitlines():
    ln = raw.strip()
    if not ln or ln.startswith("#"):
        continue
    neutral = False
    if "," in ln:
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 3:
            bad.append((raw, "need Away, Home, line")); continue
        a, h, num = p[0], p[1], p[2]
        neutral = len(p) > 3 and p[3].upper().startswith("N")
    else:
        m = re.match(r"^(.*?)\s*@\s*(.*?)\s+([+-]?\d+(?:\.\d+)?)\s*(N)?$", ln, re.I)
        if not m:
            bad.append((raw, "unparsed")); continue
        a, h, num, nflag = m.groups()
        neutral = bool(nflag)
    try:
        line = float(num)
    except ValueError:
        bad.append((raw, f"bad number '{num}'")); continue
    ka, kh = find(a), find(h)
    if ka is None or kh is None:
        bad.append((raw, f"unknown team: {a if ka is None else h}")); continue

    # scale the shrunk ratings only; offset and HFA are measured, not shrunk
    diff  = (base[kh] - base[ka]) * SCALE + (off[kh] - off[ka])
    diff += 0.0 if neutral else HFA
    model = -diff

    cls  = "FCS" if div[kh] != div[ka] else ("FBS" if div[kh] == "fbs" else "fcs-only")
    flags = []
    if neutral:                                        flags.append("NEUTRAL")
    if kh in weak or ka in weak:                       flags.append("BLIND-SPOT")
    if gp.get(kh, 0) < 10 or gp.get(ka, 0) < 10:       flags.append("thin-data")
    if abs(model - line) >= MAX_EDGE:                  flags.append("INFO-GAP")

    edge = model - line
    fav  = kh if line < 0 else ka
    take = kh if model < line else ka
    rows.append(dict(away=ka, home=kh, cls=cls, model=round(model, 1), market=line,
                     edge=round(edge, 1), side=take,
                     dir=("FAV" if take == fav else "DOG"),
                     spread_size=abs(line), neutral=neutral,
                     playable=(abs(edge) >= MIN_EDGE and abs(edge) < MAX_EDGE),
                     flag=" ".join(flags)))

if bad:
    print("COULD NOT PARSE:")
    for r, why in bad:
        print(f"  {why:<28} | {r}")
    print()
if not rows:
    sys.exit("no usable games")

d = pd.DataFrame(rows)
d = d.reindex(d.edge.abs().sort_values(ascending=False).index)

hdr = (f"{'away':<22}{'home':<20}{'cls':<5}{'model':>8}{'mkt':>8}{'edge':>7} "
       f"{'take':<20}{'dir':<5}flag")
def show(sub):
    for _, r in sub.iterrows():
        print(f"{r.away[:21]:<22}{r.home[:19]:<20}{r.cls:<5}{r.model:>+8.1f}"
              f"{r.market:>+8.1f}{r.edge:>+7.1f} {r.side[:19]:<20}{r['dir']:<5}{r.flag}")

plays, ref = d[d.playable], d[~d.playable]
print(f"{DATE}   {len(d)} games   SCALE={SCALE:.2f}  edge window {MIN_EDGE:.0f}-{MAX_EDGE:.0f}\n")
print(f"PLAYS -- edge at or above {MIN_EDGE:.0f} points  ({len(plays)})")
print("-" * 100); print(hdr); print("-" * 100)
show(plays) if len(plays) else print("  none")
print(f"\nREFERENCE ONLY -- outside the {MIN_EDGE:.0f}-{MAX_EDGE:.0f} edge window, do not bet  ({len(ref)})")
print("-" * 100)
show(ref) if len(ref) else print("  none")

out = f"data/cfb/picks_{DATE}.csv"
d.to_csv(out, index=False)
print(f"\nlogged -> {out}")
print("SCALE 1.00 is the TESTED value: cfb_earlyscale.py returned slope 1.061,\n      95% CI [0.963, 1.161] on 575 early-season games. No correction justified.")
