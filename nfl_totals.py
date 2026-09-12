"""NFL totals model, with the out-of-sample test built in.

Same discipline as nfl_sides.py: games.csv carries CLOSING TOTALS back to 1999,
so "does this beat the market" is answerable before a dollar moves.

MODEL
    Ridge on points scored, two rows per game (home, away). Each row:
        points = intercept + off[team] + def[opponent] + hfa*(is_home)
    Refit weekly on prior games with exponential time decay. Predicted total is
    the sum of both sides. Over/under graded against the closing line.

WHAT WOULD MAKE THIS WORTH BETTING
    MAE below the market's, and an O/U rate above 52.38% with a z above ~2.
    The spread version of this model came back 50.12% on 4,254 games. Totals
    markets are somewhat softer than sides, so this is not a foregone
    conclusion -- but it is not a good prior either.
"""
import sys
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

RIDGE, HALFLIFE, LOOKBACK, TEST_FROM = 8.0, 420.0, 1400, 2010

g = pd.read_csv("data/nfl/games.csv")
g = g[g.result.notna() & g.home_score.notna() & g.total_line.notna()].copy()
g["gameday"] = pd.to_datetime(g.gameday, errors="coerce")
g = g.dropna(subset=["gameday"]).sort_values("gameday").reset_index(drop=True)
g["neutral"] = g.get("location", "Home").astype(str).str.lower().eq("neutral")
g["total"] = g.home_score + g.away_score
print(f"{len(g):,} completed games with a closing total, "
      f"{g.gameday.min().date()} to {g.gameday.max().date()}")

teams = sorted(set(g.home_team) | set(g.away_team))
idx = {t: i for i, t in enumerate(teams)}
n = len(teams)
# columns: [intercept | off(n) | def(n) | hfa]
NC = 1 + 2 * n + 1


def fit(hist, asof):
    rows, y, w = [], [], []
    age = (asof - hist.gameday).dt.days.values
    dec = 0.5 ** (age / HALFLIFE)
    for k, (_, r) in enumerate(hist.iterrows()):
        for side in ("home", "away"):
            opp = "away" if side == "home" else "home"
            v = np.zeros(NC); v[0] = 1.0
            v[1 + idx[r[f"{side}_team"]]] = 1.0
            v[1 + n + idx[r[f"{opp}_team"]]] = 1.0
            if not r.neutral:
                v[-1] = 1.0 if side == "home" else 0.0
            rows.append(v); y.append(r[f"{side}_score"]); w.append(dec[k])
    X = np.array(rows); y = np.array(y, dtype=float); W = np.diag(w)
    P = np.eye(NC) * RIDGE; P[0, 0] = 0.0; P[-1, -1] = 0.0   # free intercept/HFA
    return np.linalg.solve(X.T @ W @ X + P, X.T @ W @ y)


def pred_total(b, h, a, neutral):
    base = 2 * b[0] + b[1 + idx[h]] + b[1 + n + idx[a]] \
           + b[1 + idx[a]] + b[1 + n + idx[h]]
    return base + (0.0 if neutral else b[-1])


rows = []
for (season, week), blk in g[g.season >= TEST_FROM].groupby(["season", "week"], sort=True):
    asof = blk.gameday.min()
    hist = g[(g.gameday < asof) & (g.gameday >= asof - pd.Timedelta(days=LOOKBACK))]
    if len(hist) < 200:
        continue
    b = fit(hist, asof)
    for _, r in blk.iterrows():
        rows.append(dict(season=season, pred=pred_total(b, r.home_team, r.away_team, r.neutral),
                         actual=r.total, line=r.total_line))
d = pd.DataFrame(rows)
print(f"walk-forward: {len(d):,} games, {int(d.season.min())}-{int(d.season.max())}\n")

mod, mkt = np.abs(d.pred - d.actual).mean(), np.abs(d.line - d.actual).mean()
print("=" * 62)
print(f"ACCURACY   model MAE {mod:.3f}   market MAE {mkt:.3f}   gap {mod-mkt:+.3f}")
print(f"           constant ({d.actual.mean():.1f}) MAE "
      f"{np.abs(d.actual.mean()-d.actual).mean():.3f}")
print("=" * 62)

dd = d[d.actual != d.line].copy()
dd["over"] = dd.pred > dd.line
dd["win"] = np.where(dd.over, dd.actual > dd.line, dd.actual < dd.line)
w, l = int(dd.win.sum()), int((~dd.win).sum()); N = w + l
se = (0.25 / N) ** 0.5
print(f"\nO/U  {w}-{l}  =  {w/N*100:.2f}%   n={N:,}")
print(f"     break-even 52.38%   z = {(w/N-0.5238)/se:+.2f}   "
      f"ROI {((w/N)*1.9091-1)*100:+.2f}%")
print(f"\n{'edge bucket':<16}{'n':>8}{'O/U%':>9}{'z':>8}")
print("-" * 41)
for lo, hi in [(0,1),(1,2),(2,3),(3,5),(5,99)]:
    m = dd[(np.abs(dd.pred-dd.line) >= lo) & (np.abs(dd.pred-dd.line) < hi)]
    if len(m) < 100: continue
    pp = m.win.mean()
    print(f"{f'{lo}-{hi} pts':<16}{len(m):>8,}{pp*100:>8.2f}%{(pp-0.5238)/((0.25/len(m))**0.5):>+8.2f}")
import json
json.dump({"n": int(N), "record": f"{w}-{l}", "ou": w/N, "z": (w/N-0.5238)/se,
           "mae_model": float(mod), "mae_market": float(mkt),
           "verdict": "NO EDGE -- do not bet" if (w/N-0.5238)/se < 2 else "review"},
          open("data/nfl/totals_validation.json", "w"), indent=2)
print("\n-> data/nfl/totals_validation.json")
