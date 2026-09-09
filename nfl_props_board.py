"""Fair lines for receptions / receiving yards / rushing yards.

    python3 nfl_props_board.py                       # fair lines only
    python3 nfl_props_board.py data/nfl/props.txt    # + P(over) vs real lines

Optional props file, one per line:   Player Name, market, line
    Ja'Marr Chase, receptions, 6.5
    Bijan Robinson, rushing_yards, 78.5

OUTPUT
    fair   -- the median of the predicted distribution. Compare to the book:
              book line well BELOW fair is a lean over, and vice versa.
    p25/p75 -- the middle half of the outcome range. A book line inside that
              band is a coin flip whatever the fair number says, which is most
              of them and is why most props are not bets.

The distributions behind these passed a randomized-PIT calibration test on
held-out 2024-25 (max decile deviation 0.9-1.6pp), so P(over) is trustworthy.
KNOWN GAP: no game-script adjustment yet. A team trailing by two scores throws
more and runs less, and the spread predicts that. Usage here is recent-form
only, so expect the biggest errors in games with large spreads.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
USAGE_SEASON, OPP_WINDOW, K_OPP = 2025, 4, 0.25
MARKETS = {"receptions": ("receptions", "targets", 2.5, 25.0),
           "receiving_yards": ("receiving_yards", "targets", 2.5, 40.0),
           "rushing_yards": ("rushing_yards", "carries", 5.0, 40.0)}
BINS = [0, 2, 4, 6, 9, 99]

h = pd.concat([pd.read_csv(f"data/nfl/player_week_{y}.csv", low_memory=False)
               for y in SEASONS], ignore_index=True)
h = h[h.season_type.eq("REG") & h.position_group.isin(["RB", "WR", "TE"])].copy()
for c in ("carries", "targets", "receptions", "receiving_yards", "rushing_yards"):
    h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0.0)

car = h.groupby(["player_id", "player_display_name", "position_group"]).agg(
    c=("carries", "sum"), t=("targets", "sum"), rec=("receptions", "sum"),
    ry=("receiving_yards", "sum"), ru=("rushing_yards", "sum")).reset_index()

u = h[h.season == USAGE_SEASON].sort_values(["player_id", "week"])
u = u.groupby("player_id").tail(OPP_WINDOW)
use = u.groupby("player_id").agg(g=("week", "size"), team25=("team", "last"),
                                 mc=("carries", "mean"), mt=("targets", "mean")).reset_index()
d = car.merge(use[use.g >= 2], on="player_id", how="inner")
pm = d.groupby("position_group")[["mc", "mt"]].transform("mean")
W = np.minimum(d.g, OPP_WINDOW)
d["e_targets"] = (d.mt * W + K_OPP * pm.mt) / (W + K_OPP)
d["e_carries"] = (d.mc * W + K_OPP * pm.mc) / (W + K_OPP)

rp = Path("data/nfl/roster_weekly_2026.csv")
if rp.exists():
    r = pd.read_csv(rp, low_memory=False)
    idc = "gsis_id" if "gsis_id" in r.columns else "player_id"
    if "status" in r.columns:
        r = r[r.status.astype(str).str.upper().eq("ACT")]
    r = r.sort_values("week").groupby(idc).tail(1)[[idc, "team"]]
    d = d.merge(r.rename(columns={idc: "player_id", "team": "team26"}), on="player_id", how="left")
    d = d[d.team26.notna()]; d["team"] = d.team26
else:
    d["team"] = d.team25

# empirical ratio distributions, bucketed by usage, from history
hist = h.copy()
gb = hist.groupby("player_id", sort=False)
for c in ("carries", "targets", "receptions", "receiving_yards", "rushing_yards"):
    hist[f"pr_{c}"] = gb[c].cumsum() - hist[c]
for c in ("carries", "targets"):
    hist[f"rec_{c}"] = (gb[c].shift(1).groupby(hist.player_id)
                        .rolling(OPP_WINDOW, min_periods=1).mean()
                        .reset_index(level=0, drop=True))
hist["games_prior"] = gb.cumcount()
hpm = hist.groupby("position_group")[["carries", "targets"]].transform("mean")
HW = np.minimum(hist.games_prior, OPP_WINDOW)
hist["e_targets"] = (hist.rec_targets.fillna(hpm.targets) * HW + K_OPP * hpm.targets) / (HW + K_OPP)
hist["e_carries"] = (hist.rec_carries.fillna(hpm.carries) * HW + K_OPP * hpm.carries) / (HW + K_OPP)
hist = hist[hist.games_prior >= 4]

RATIOS, RATE = {}, {}
for name, (stat, opp, minopp, k) in MARKETS.items():
    mu_lg = hist[f"pr_{stat}"].sum() / max(hist[f"pr_{opp}"].sum(), 1)
    rate = (hist[f"pr_{stat}"] + k * mu_lg) / (hist[f"pr_{opp}"] + k)
    mu = hist[f"e_{opp}"] * rate
    sub = hist[(hist[f"e_{opp}"] >= minopp) & (mu > 0.05)].copy()
    sub["ratio"] = sub[stat] / mu[sub.index]
    sub["b"] = pd.cut(sub[f"e_{opp}"], BINS, labels=False, include_lowest=True)
    RATIOS[name] = {b: np.sort(g.ratio.values) for b, g in sub.groupby("b") if len(g) > 200}
    RATE[name] = (mu_lg, k)

rows = []
for name, (stat, opp, minopp, k) in MARKETS.items():
    mu_lg, _ = RATE[name]
    tot = {"receptions": d.rec, "receiving_yards": d.ry, "rushing_yards": d.ru}[name]
    den = d.t if opp == "targets" else d.c
    rate = (tot + k * mu_lg) / (den + k)
    mu = d[f"e_{opp}"] * rate
    b = pd.cut(d[f"e_{opp}"], BINS, labels=False, include_lowest=True)
    for i, r in d.iterrows():
        if r[f"e_{opp}"] < minopp or mu[i] <= 0.05:
            continue
        R = RATIOS[name].get(b[i])
        if R is None:
            continue
        q = np.quantile(R, [0.25, 0.50, 0.75]) * mu[i]
        rows.append(dict(player=r.player_display_name, pos=r.position_group,
                         team=r.team, market=name, exp_opp=round(r[f"e_{opp}"], 1),
                         mu=round(mu[i], 1), fair=round(q[1], 1),
                         p25=round(q[0], 1), p75=round(q[2], 1), bucket=int(b[i])))
b = pd.DataFrame(rows)

lines = {}
if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
    for ln in Path(sys.argv[1]).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 3:
            lines[(f[0].lower(), f[1])] = float(f[2])
if lines:
    povers = []
    for _, r in b.iterrows():
        L = lines.get((r.player.lower(), r.market))
        if L is None:
            povers.append(np.nan); continue
        R = RATIOS[r.market][r.bucket]
        povers.append(float((R * r.mu > L).mean()))
    b["line"] = [lines.get((r.player.lower(), r.market), np.nan) for _, r in b.iterrows()]
    b["p_over"] = povers

b = b.sort_values(["market", "fair"], ascending=[True, False])
for m in MARKETS:
    s = b[b.market == m]
    if not len(s): continue
    print(f"\n{m.upper()}   {len(s)} players")
    cols = f"{'player':<24}{'tm':<5}{'opp':>6}{'fair':>8}{'p25':>8}{'p75':>8}"
    if "p_over" in s: cols += f"{'line':>8}{'P(ovr)':>9}"
    print(cols); print("-" * len(cols))
    for _, r in s.head(15).iterrows():
        line = f"{r.player[:23]:<24}{r.team:<5}{r.exp_opp:>6.1f}{r.fair:>8.1f}{r.p25:>8.1f}{r.p75:>8.1f}"
        if "p_over" in s and not pd.isna(r.get("p_over")):
            line += f"{r.line:>8.1f}{r.p_over*100:>8.1f}%"
        print(line)
b.to_csv("data/nfl/props_board.csv", index=False)
print(f"\n{len(b)} player-markets -> data/nfl/props_board.csv")
