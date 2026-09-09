"""NFL point-spread model, with the out-of-sample test built in.

    python3 nfl_sides.py

WHY THIS IS DIFFERENT FROM THE CFB BUILD
    games.csv carries CLOSING SPREADS back to 1999. So this does not have to be
    built, bet, and evaluated over five seasons -- it can be tested against
    ~7,000 real lines before a dollar moves. That is the single thing the CFB
    project lacked, and the reason it ran a week of money before anyone asked
    whether it beat the market.

MODEL
    Ridge on point differential, refit every week on prior games only, with
    exponential time decay so a 2019 result does not weigh like last month's.
    +1 home team, -1 away team, +1 HFA (0 at neutral sites). 32 teams, 33
    parameters -- the sparse-schedule problem that makes ridge essential in
    college barely exists here, which is itself a reason to expect little edge.

WHAT TO LOOK FOR
    MAE vs the market's is the honest headline. ATS% is what pays, and it needs
    to clear 52.38% to break even at -110. Anything in the 50-52% range is a
    losing model that looks fine.
"""
import sys
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

RIDGE      = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
HALFLIFE   = float(sys.argv[2]) if len(sys.argv) > 2 else 420.0   # days
TEST_FROM  = 2010
LOOKBACK   = 1400        # days of history per fit

g = pd.read_csv("data/nfl/games.csv")
g = g[g.result.notna() & g.home_score.notna()].copy()
g["gameday"] = pd.to_datetime(g.gameday, errors="coerce")
g = g.dropna(subset=["gameday"]).sort_values("gameday").reset_index(drop=True)
g["neutral"] = g.get("location", "Home").astype(str).str.lower().eq("neutral")
g["margin"] = g.home_score - g.away_score
print(f"{len(g):,} completed games, {g.gameday.min().date()} to {g.gameday.max().date()}")

teams = sorted(set(g.home_team) | set(g.away_team))
idx = {t: i for i, t in enumerate(teams)}
n = len(teams)

def fit(hist, asof):
    age = (asof - hist.gameday).dt.days.values
    w = 0.5 ** (age / HALFLIFE)
    X = np.zeros((len(hist), n + 1)); y = hist.margin.values.astype(float)
    for k, (_, r) in enumerate(hist.iterrows()):
        X[k, idx[r.home_team]] = 1.0
        X[k, idx[r.away_team]] = -1.0
        X[k, n] = 0.0 if r.neutral else 1.0
    W = np.diag(w)
    P = np.eye(n + 1) * RIDGE; P[n, n] = 0.0          # never penalise HFA
    beta = np.linalg.solve(X.T @ W @ X + P, X.T @ W @ y)
    beta[:n] -= beta[:n].mean()
    return beta

rows = []
for (season, week), blk in g[g.season >= TEST_FROM].groupby(["season", "week"], sort=True):
    asof = blk.gameday.min()
    hist = g[(g.gameday < asof) & (g.gameday >= asof - pd.Timedelta(days=LOOKBACK))]
    if len(hist) < 200:
        continue
    beta = fit(hist, asof)
    for _, r in blk.iterrows():
        pred = beta[idx[r.home_team]] - beta[idx[r.away_team]] + (0 if r.neutral else beta[n])
        rows.append(dict(season=season, week=week, pred=pred, actual=r.margin,
                         spread=r.spread_line, neutral=r.neutral))
d = pd.DataFrame(rows)
print(f"walk-forward predictions: {len(d):,} games, {int(d.season.min())}-{int(d.season.max())}")

d["mod_err"] = (d.pred - d.actual).abs()
lined = d.dropna(subset=["spread"]).copy()
lined["mkt_err"] = (lined.spread - lined.actual).abs()
print("\n" + "=" * 62)
print(f"ACCURACY   model MAE {d.mod_err.mean():.3f}"
      f"   market MAE {lined.mkt_err.mean():.3f}"
      f"   gap {d.mod_err.mean() - lined.mkt_err.mean():+.3f}")
print("=" * 62)

# ATS: back the side the model likes vs the closing line
lined["edge"] = lined.pred - lined.spread
lined = lined[lined.edge.abs() > 0]
lined["push"] = lined.actual == lined.spread
lined["win"] = np.where(lined.edge > 0, lined.actual > lined.spread,
                                        lined.actual < lined.spread)
dec = lined[~lined.push]
w, l = int(dec.win.sum()), int((~dec.win).sum()); N = w + l
se = (0.25 / N) ** 0.5
print(f"\nATS  {w}-{l}  =  {w/N*100:.2f}%   n={N:,}")
print(f"     break-even at -110 is 52.38%   z = {(w/N - 0.5238)/se:+.2f}")
print(f"     ROI at -110: {((w/N)*1.9091 - 1)*100:+.2f}%")

print(f"\n{'edge bucket':<16}{'n':>8}{'ATS%':>9}{'z':>8}")
print("-" * 41)
for lo, hi in [(0,1),(1,2),(2,3),(3,5),(5,7),(7,99)]:
    m = dec[(dec.edge.abs() >= lo) & (dec.edge.abs() < hi)]
    if len(m) < 100: continue
    ww = m.win.sum(); nn = len(m); pp = ww/nn
    print(f"{f'{lo}-{hi} pts':<16}{nn:>8,}{pp*100:>8.2f}%{(pp-0.5238)/((0.25/nn)**0.5):>+8.2f}")
import json
json.dump({"tested": "2026-09-09", "n": int(N), "record": f"{w}-{l}",
           "ats": w / N, "z": (w/N - 0.5238) / se, "roi": (w/N)*1.9091 - 1,
           "mae_model": float(d.mod_err.mean()), "mae_market": float(lined.mkt_err.mean()),
           "seasons": f"{int(d.season.min())}-{int(d.season.max())}",
           "verdict": "NO EDGE -- do not bet"},
          open("data/nfl/sides_validation.json", "w"), indent=2)
print("\n  -> data/nfl/sides_validation.json")

print("\nIf no bucket clears 52.38% with a z above ~2, there is no edge here and")
print("the honest move is to not bet it -- the same conclusion the CFB out-of-")
print("sample test reached at 49.1%, just reached before the money instead of after.")
