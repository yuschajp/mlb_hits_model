"""Grade the anytime-TD board -- including against the BOOK's price.

    python3 nfl_fetch.py 2026        # refresh player stats after games
    python3 nfl_grade.py             # grade every ungraded td_value_*.csv

This is the measurement missing from every model in this stack. Calibration
says the model's probabilities are honest. It does NOT say they beat a price.
The NFL spread model was honest too and returned 50.12% on 4,254 games.

WHAT IT REPORTS
  1. Calibration -- model vs actual, and BOOK vs actual. If the book is better
     calibrated than the model, there is no edge to find, full stop.
  2. Brier for both. Lower wins.
  3. ROI of actually betting the model's positive-edge rows AT THE BOOK'S PRICE.
     This is the only number that decides anything.

EDGE IS ALLOCATION, NOT LEVEL
  The model splits each team's touchdowns over its own player list; the book
  uses a longer one, which inflates every model probability ~15%. Both sides
  are renormalised per game before the edge is computed -- otherwise every
  player shows fake value and the ROI below is meaningless.
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
MIN_EDGE = 0.04

src = Path("data/nfl/td_value.csv")
if not src.exists():
    sys.exit("no data/nfl/td_value.csv -- run nfl_td_value.py first")
v = pd.read_csv(src)

pw = Path(f"data/nfl/player_week_{SEASON}.csv")
if not pw.exists():
    sys.exit(f"no {pw} -- run:  python3 nfl_fetch.py {SEASON}")
p = pd.read_csv(pw, low_memory=False)
p = p[p.season_type.eq("REG")].copy()
for c in ("rushing_tds", "receiving_tds"):
    p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
p["td"] = p.rushing_tds + p.receiving_tds
wk = int(p.week.max())
act = (p[p.week == wk].groupby("player_id")
       .agg(td=("td", "sum"), name=("player_display_name", "last")).reset_index())
print(f"grading week {wk} of {SEASON}: {len(act)} players with a box score")

v = v.merge(act[["player_id", "td"]].rename(columns={"player_id": "pid"}),
            left_on="player_id", right_on="pid", how="left") \
     if "player_id" in v.columns else v.merge(
        act.assign(key=act.name.str.lower().str.replace(r"[^a-z]", "", regex=True))
           [["key", "td"]],
        left_on=v.player_display_name.str.lower().str.replace(r"[^a-z]", "", regex=True),
        right_on="key", how="left")
v = v.dropna(subset=["td"])
if not len(v):
    sys.exit("no players matched -- check that the week has been played and fetched")
v["hit"] = (v.td > 0).astype(int)

# renormalise per game: compare ALLOCATION, not level
mt = v.groupby("matchup").p_td.transform("sum")
bt = v.groupby("matchup").p_book.transform("sum")
v["p_model"] = v.p_td * (bt / mt)
v["edge"] = v.p_model - v.p_book
v["dec"] = 1.0 / v.p_book_raw          # the book's actual payout

n = len(v)
print("\n" + "=" * 64)
print(f"CALIBRATION   n={n}")
print("=" * 64)
print(f"  {'source':<10}{'predicted':>12}{'actual':>10}{'Brier':>10}")
for lab, col in (("model", "p_model"), ("book", "p_book")):
    print(f"  {lab:<10}{v[col].mean()*100:>11.2f}%{v.hit.mean()*100:>9.2f}%"
          f"{np.mean((v[col]-v.hit)**2):>10.5f}")
bm, bb = np.mean((v.p_model-v.hit)**2), np.mean((v.p_book-v.hit)**2)
print(f"\n  -> {'MODEL' if bm < bb else 'BOOK'} is better calibrated "
      f"(difference {abs(bm-bb):.5f})")
if bb < bm:
    print("     The book predicts this market better than the model does.")
    print("     That is close to fatal for the idea of an edge here.")

print("\n" + "=" * 64)
print("ROI -- betting the model's positive-edge rows at the book's price")
print("=" * 64)
print(f"  {'edge bucket':<16}{'n':>6}{'hit%':>8}{'book%':>8}{'ROI':>10}")
for lo, hi, lab in [(MIN_EDGE, 0.08, f"{MIN_EDGE:.0%}-8%"), (0.08, 0.15, "8-15%"),
                    (0.15, 1.0, "15%+"), (-1.0, -MIN_EDGE, "negative (fade)")]:
    s = v[(v.edge >= lo) & (v.edge < hi)]
    if len(s) < 5: continue
    roi = (s.hit * (s.dec - 1) - (1 - s.hit)).mean() * 100
    print(f"  {lab:<16}{len(s):>6}{s.hit.mean()*100:>7.1f}%{s.p_book.mean()*100:>7.1f}%{roi:>9.1f}%")
bet = v[v.edge >= MIN_EDGE]
if len(bet):
    roi = (bet.hit * (bet.dec - 1) - (1 - bet.hit)).mean() * 100
    se = (bet.hit * (bet.dec - 1) - (1 - bet.hit)).std(ddof=1) / np.sqrt(len(bet)) * 100
    print(f"\n  ALL edges >= {MIN_EDGE:.0%}:  n={len(bet)}  "
          f"ROI {roi:+.1f}%  +/- {1.96*se:.1f} (95%)")
    print(f"  {'PROFITABLE on this sample' if roi-1.96*se > 0 else 'not distinguishable from zero'}")
out = Path(f"data/nfl/graded_td_{SEASON}_wk{wk}.csv")
v.to_csv(out, index=False)
print(f"\n-> {out}   ({n} rows)")
print("One week is one week. This needs a few hundred graded rows before the")
print("ROI line means anything -- but it is the first real reading in the stack.")
