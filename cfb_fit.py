"""Fit ratings and write the files the board reads.

    python3 cfb_fit.py 2025

Writes:
    data/cfb/ratings_<year>_fbs.csv    index=team, column net (POINTS)
    data/cfb/ratings_<year>_fcs.csv
    and prints the measured FBS->FCS OFFSET to paste into cfb_board2.py

WHY TWO SCALES AND NOT ONE JOINT FIT
    A single ridge across all 266 teams put NDSU 10th overall with a gap to the
    FBS field of 13.5 points, against a true gap around 31. Cross-division games
    are rare and lopsided, so the joint fit has almost no information tying the
    two populations together and quietly collapses them onto one scale.
    Instead: fit each division on its OWN games, then measure the bridge from
    the cross-division games directly, where it is observable.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from cfb_model import fit_ratings, net_points, PARAMS

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
OUT = Path("data/cfb")

g = pd.read_csv(OUT / f"games_{YEAR}_all.csv")
dv = pd.read_csv(OUT / f"division_{YEAR}.csv", index_col=0)["division"] \
       .astype(str).str.lower()
div = dv.to_dict()

g["hdiv"] = g.home_team.map(div)
g["adiv"] = g.away_team.map(div)
g = g.dropna(subset=["hdiv", "adiv"])

within = {d: g[(g.hdiv == d) & (g.adiv == d)] for d in ("fbs", "fcs")}
cross = g[g.hdiv != g.adiv]
print(f"{YEAR}: {len(g)} games  |  fbs-only {len(within['fbs'])}  "
      f"fcs-only {len(within['fcs'])}  cross {len(cross)}")

nets = {}
for d in ("fbs", "fcs"):
    sub = within[d]
    if len(sub) < 50:
        sys.exit(f"only {len(sub)} {d} games -- not enough to fit")
    R = fit_ratings(sub, params=PARAMS)
    net = net_points(R, PARAMS).sort_values(ascending=False)
    nets[d] = net
    p = OUT / f"ratings_{YEAR}_{d}.csv"
    net.rename("net").to_frame().to_csv(p, index_label="team")
    print(f"  {d}: {len(net)} teams, hfa {R.attrs['hfa_ppd']*PARAMS['league_drives']:+.2f} pts -> {p}")
    print(f"     top 5:    " + ", ".join(f"{t} {v:+.1f}" for t, v in net.head(5).items()))
    print(f"     bottom 3: " + ", ".join(f"{t} {v:+.1f}" for t, v in net.tail(3).items()))

# ---- measure the bridge where it is observable: on cross-division games ------
res = []
for _, r in cross.iterrows():
    hd, ad = r.hdiv, r.adiv
    if r.home_team not in nets[hd].index or r.away_team not in nets[ad].index:
        continue
    hfa = 0.0 if r.get("neutral", False) else PARAMS["hfa_points"]
    fbs_is_home = (hd == "fbs")
    rating_gap = (nets[hd][r.home_team] - nets[ad][r.away_team])
    actual = r.home_points - r.away_points
    # actual = rating_gap + hfa + OFFSET*(+1 if fbs at home else -1)
    resid = actual - rating_gap - hfa
    res.append(resid if fbs_is_home else -resid)

res = np.array(res)
if len(res) < 20:
    print(f"\n  only {len(res)} usable cross-division games -- keeping OFFSET as-is")
else:
    med, mean = float(np.median(res)), float(res.mean())
    se = res.std(ddof=1) / np.sqrt(len(res))
    print(f"\n  FBS->FCS OFFSET from {len(res)} bridge games")
    print(f"    median {med:.1f}   mean {mean:.1f} +/- {1.96*se:.1f} (95%)")
    print(f"    -> set OFFSET = {med:.1f} in cfb_board2.py")
    print(f"    (median, not mean: blowout margins are right-skewed)")
