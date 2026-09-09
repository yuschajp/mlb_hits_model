"""Measure the EARLY-SEASON scale factor -- the one number that decides whether
this week's 1.457 was real or eleven games of noise.

THE CLAIM UNDER TEST
    In weeks 0-3 every rating is a carry-forward from last season. Those ratings
    were produced by a ridge fit, so they are already shrunk toward the mean.
    Shrinkage is correct for ESTIMATING a rating and wrong for PREDICTING a
    margin: in a mismatch both teams get pulled toward the middle and the
    predicted gap collapses. If that is happening, actual margins will be
    systematically LARGER than predicted ones.

THE TEST
    For each season y: fit ratings on season y-1, predict weeks 1-3 of year y,
    regress ACTUAL margin on PREDICTED margin through the origin.
        slope = 1.0  -> correctly scaled, do nothing
        slope > 1.0  -> compressed; multiply early-season spreads by the slope
    No market data anywhere, so the book's opinion cannot contaminate it.

WHY SATURDAY'S TEST MISSED THIS
    cfb_shrink.py needs 80 training games, so it starts at week 3 and runs on
    in-season ratings. It measured a regime you do not bet in during September.

CAVEAT BUILT IN
    Noise in the predictor biases a slope toward zero, so whatever comes back is
    a FLOOR on the true factor, not a point estimate. A reported 1.2 could be 1.3.
"""
import glob, re, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from cfb_model import fit_ratings, team_pace, predict_game, PARAMS

P = dict(PARAMS, ridge_lambda=2.0, cap_ppd=5.0, yoy_regression=0.0)
MAX_WEEK = 3          # the preseason-prior regime
NEED = ["home_team", "away_team", "home_points", "away_points",
        "home_drives", "away_drives", "week"]


def load_season(year):
    """Find whatever games file exists for this season and return FBS-vs-FBS rows."""
    for pat in (f"data/cfb/games_{year}_fbs.csv", f"data/cfb/games_{year}_all.csv"):
        try:
            g = pd.read_csv(pat)
        except Exception:
            continue
        missing = [c for c in NEED if c not in g.columns]
        if missing:
            print(f"  {pat}: missing columns {missing} -- skipped")
            continue
        g = g.dropna(subset=NEED)
        try:
            dv = pd.read_csv(f"data/cfb/division_{year}.csv",
                             index_col=0)["division"].astype(str).str.lower()
            fbs = set(dv[dv.str.contains("fbs")].index)
            g = g[g.home_team.isin(fbs) & g.away_team.isin(fbs)]
        except Exception:
            pass                      # a _fbs.csv is already filtered
        if len(g):
            return g, pat
    return None, None


years = sorted({int(m.group(1)) for f in glob.glob("data/cfb/games_*.csv")
                for m in [re.search(r"games_(\d{4})_", f)] if m})
print(f"seasons found: {years}")
pairs = [(y - 1, y) for y in years if (y - 1) in years]
if not pairs:
    sys.exit("need two consecutive seasons of games_*.csv to run this test")

rows = []
for prev, cur in pairs:
    gp, sp = load_season(prev)
    gc, sc = load_season(cur)
    if gp is None or gc is None:
        print(f"  {prev}->{cur}: missing a season, skipped");  continue
    early = gc[gc.week <= MAX_WEEK]
    if len(early) < 20:
        print(f"  {prev}->{cur}: only {len(early)} early games, skipped");  continue

    R = fit_ratings(gp, params=P)
    PC = team_pace(gp, params=P)
    n_ok = 0
    for _, r in early.iterrows():
        if r.home_team not in R.index or r.away_team not in R.index:
            continue
        pr = predict_game(R, PC, r.home_team, r.away_team,
                          neutral=bool(r.get("neutral", False)), params=P)
        rows.append(dict(season=cur, week=int(r.week),
                         pred=-pr["pred_spread"],
                         act=r.home_points - r.away_points))
        n_ok += 1
    print(f"  {prev} -> {cur} weeks 1-{MAX_WEEK}: {n_ok} games predicted")

d = pd.DataFrame(rows)
if len(d) < 40:
    sys.exit(f"only {len(d)} usable games -- not enough to conclude anything")

slope = lambda s: (s.pred * s.act).sum() / (s.pred ** 2).sum()
rng = np.random.default_rng(0)
bs = [slope(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
s = slope(d)

print("\n" + "=" * 68)
print(f"EARLY-SEASON SCALE TEST   n={len(d)} games, weeks 1-{MAX_WEEK}, "
      f"seasons {sorted(d.season.unique())}")
print("=" * 68)
print(f"  slope (actual on predicted) : {s:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
print(f"  this week's 11-game estimate: 1.457")
if lo > 1.05:
    print(f"\n  -> CONFIRMED. Multiply early-season model spreads by {s:.2f}.")
    print(f"     Remember attenuation: {s:.2f} is a floor, the true factor is >= this.")
elif hi < 1.05:
    print("\n  -> NOT COMPRESSED. This week's 1.457 was eleven games of noise.")
    print("     Drop the idea and stop revisiting it.")
else:
    print("\n  -> INCONCLUSIVE. CI straddles 1.0; do not apply a correction yet.")

print(f"\n  {'|pred| bucket':<16}{'n':>6}{'slope':>9}{'mean|pred|':>12}{'mean|act|':>11}")
print("  " + "-" * 54)
for a, b in [(0, 7), (7, 14), (14, 21), (21, 99)]:
    m = d[(d.pred.abs() >= a) & (d.pred.abs() < b)]
    if len(m) < 15:
        continue
    print(f"  {f'{a}-{b}':<16}{len(m):>6}{slope(m):>9.3f}"
          f"{m.pred.abs().mean():>12.1f}{m.act.abs().mean():>11.1f}")
print("  A slope that RISES with bucket size means the compression is worst in")
print("  mismatches -- which is where the board makes most of its picks.")

print(f"\n  {'week':<8}{'n':>6}{'slope':>9}")
print("  " + "-" * 23)
for wk, m in d.groupby("week"):
    if len(m) >= 15:
        print(f"  {int(wk):<8}{len(m):>6}{slope(m):>9.3f}")
print("=" * 68)
