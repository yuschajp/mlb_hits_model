"""In-season refit: rebuild ratings from THIS season's games, 2025 as prior.

    python3 cfb_fetch.py 2026     # first -- needs CFBD, run from your terminal
    python3 cfb_refit.py          # then this

WHY
  The board has been running on ratings fitted to 2025 and carried forward. Six
  weeks into 2026 that is the single biggest reason model MAE is 14.0 against
  the market's 12.0. Teams are not what they were last November.

  fit_ratings already takes a `prior` with weight prior_weight_g, which is the
  right mechanism: early in the season the prior dominates, and as 2026 games
  accumulate the data takes over. No new maths, just the correct inputs.

WHAT THIS DOES *NOT* DO
  It does not write new ratings unless they are measurably better. The refit is
  validated WALK-FORWARD: for each slate, fit on games played strictly BEFORE
  that slate, predict it, and compare MAE against (a) the 2025 carryover the
  board uses now and (b) the market. Fitting and scoring on the same games would
  guarantee an improvement and prove nothing -- that mistake has already been
  made once on this project with the early-season SCALE factor.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from cfb_model import fit_ratings, team_pace, predict_game, net_points, PARAMS

P = dict(PARAMS, ridge_lambda=2.0, cap_ppd=5.0, yoy_regression=0.0)
OUT = Path("data/cfb")
NEED = ["home_team","away_team","home_points","away_points","home_drives","away_drives"]

g26 = OUT / "games_2026_all.csv"
if not g26.exists():
    sys.exit("missing data/cfb/games_2026_all.csv\n"
             "  run this in YOUR terminal first (needs CFBD):\n"
             "      python3 cfb_fetch.py 2026")

cur = pd.read_csv(g26).dropna(subset=NEED)
prev = pd.read_csv(OUT / "games_2025_all.csv").dropna(subset=NEED)
if "week" not in cur.columns:
    sys.exit("games_2026_all.csv has no `week` column -- cannot order slates")

try:
    dv = pd.read_csv(OUT / "division_2025.csv", index_col=0)["division"].astype(str).str.lower()
    fbs = set(dv[dv.str.contains("fbs")].index)
except Exception:
    fbs = set()

prior25 = fit_ratings(prev, params=P)
weeks = sorted(cur.week.unique())
print(f"2026 games: {len(cur)}   weeks {weeks[0]}-{weeks[-1]}")
print()
print(f"  {'wk':<5}{'n':>5}{'carryover MAE':>16}{'refit MAE':>12}{'market':>9}")
print("  " + "-" * 48)

rows = []
for wk in weeks:
    tr = cur[cur.week < wk]
    te = cur[cur.week == wk]
    if len(tr) < 40 or not len(te):
        continue
    R = fit_ratings(tr, params=P, prior=prior25)
    PC = team_pace(tr, params=P)
    PC25 = team_pace(prev, params=P)
    for _, r in te.iterrows():
        if r.home_team not in R.index or r.away_team not in R.index: continue
        if r.home_team not in prior25.index or r.away_team not in prior25.index: continue
        nt = bool(r.get("neutral", False))
        new = predict_game(R, PC, r.home_team, r.away_team, neutral=nt, params=P)["pred_spread"]
        old = predict_game(prior25, PC25, r.home_team, r.away_team, neutral=nt, params=P)["pred_spread"]
        act = r.away_points - r.home_points
        rows.append(dict(week=wk, new=new, old=old, act=act,
                         both_fbs=(r.home_team in fbs and r.away_team in fbs)))
    b = pd.DataFrame([x for x in rows if x["week"] == wk])
    if len(b):
        print(f"  {int(wk):<5}{len(b):>5}{(b.old-b.act).abs().mean():>16.2f}"
              f"{(b.new-b.act).abs().mean():>12.2f}{'':>9}")

d = pd.DataFrame(rows)
if len(d) < 50:
    sys.exit(f"\nonly {len(d)} out-of-sample games -- too few to decide. Re-run later in the season.")

mo = (d.old - d.act).abs().mean()
mn = (d.new - d.act).abs().mean()
print()
print("=" * 62)
print(f"WALK-FORWARD RESULT   n={len(d)} out-of-sample games")
print("=" * 62)
print(f"  2025 carryover (what the board uses now) : MAE {mo:.2f}")
print(f"  2026 in-season refit                     : MAE {mn:.2f}")
print(f"  improvement                              : {mo-mn:+.2f} points")

rng = np.random.default_rng(0)
diff = (d.old - d.act).abs() - (d.new - d.act).abs()
bs = [diff.iloc[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  95% CI on the improvement                : [{lo:+.2f}, {hi:+.2f}]")
print()

if lo > 0:
    print("  -> CONFIRMED better out-of-sample. Writing new ratings files.")
    R = fit_ratings(cur, params=P, prior=prior25)
    net = net_points(R, params=P)
    for tag, keep in (("fbs", lambda t: t in fbs), ("fcs", lambda t: t not in fbs)):
        sub = net[[keep(t) for t in net.index]]
        if not len(sub): continue
        f = OUT / f"ratings_2026_{tag}.csv"
        pd.DataFrame({"net": sub}).to_csv(f)
        print(f"     wrote {f}  ({len(sub)} teams)")
    print()
    print("  TO USE THEM, point the board at 2026 instead of 2025:")
    print("     in cfb_board2.py change  ratings_2025_{d}.csv  ->  ratings_2026_{d}.csv")
    print("  and RE-MEASURE the FBS->FCS OFFSET on 2026 bridge games before you")
    print("  trust it -- OFFSET 38.6 was fitted against the 2025-based ratings and")
    print("  does not carry over to a different rating scale.")
elif hi < 0:
    print("  -> WORSE out-of-sample. Keep the 2025 carryover. Nothing written.")
else:
    print("  -> INCONCLUSIVE; CI contains zero. Nothing written. Re-run in 2-3 weeks")
    print("     when more 2026 games have accumulated.")
