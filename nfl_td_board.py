"""Anytime-TD board from a lines file.

    python3 nfl_td_board.py data/nfl/lines_2026-w1.txt

Lines file:  away, home, home_spread, total[, N]
    home_spread POSITIVE = home favoured (nflverse convention, NOT the CFB one)

NEUTRAL SITES NEED NO SPECIAL CASE HERE. The CFB board computes its own spread
and so had to know about HFA; this model consumes the MARKET's spread, which
already prices the venue. The N flag is printed for context only.

ROSTERS: uses roster_weekly_2026 so free agents land on the right team. Without
it the board falls back to 2025 team assignments and says so loudly.

WEEK 1 CAVEAT, STATED UP FRONT: usage comes entirely from last season. Rookies
have no row at all and are simply absent; players with changed roles are priced
on the old one. This is the same preseason-prior regime where the CFB model was
weakest, and it is the worst week of the year for this model.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/nfl/lines_2026-w1.txt")
SEASONS = list(range(2018, 2026))
USAGE_SEASON = 2025
OPP_WINDOW, K_OPP = 4, 0.25
TD_A, TD_B = -0.651, 0.1361     # team offensive TDs = a + b * implied points
MIN_GAMES = 2

rows = []
for ln in SRC.read_text().splitlines():
    ln = ln.strip()
    if not ln or ln.startswith("#"): continue
    f = [x.strip() for x in ln.split(",")]
    rows.append(dict(away=f[0], home=f[1], spread=float(f[2]), total=float(f[3]),
                     neutral=len(f) > 4 and f[4].upper().startswith("N")))
games = pd.DataFrame(rows)
teams = pd.concat([games.home, games.away])
dup = teams[teams.duplicated()].unique()
if len(dup):
    sys.exit(f"ERROR: {SRC.name} has these teams in more than one game: "
             f"{', '.join(dup[:8])}\n"
             "A player would be priced once per game and the board would be\n"
             "meaningless. Re-run nfl_lines.py to window it to one slate.")
print(f"{len(games)} games from {SRC.name}\n")

hist = pd.concat([pd.read_csv(f"data/nfl/player_week_{y}.csv", low_memory=False)
                  for y in SEASONS], ignore_index=True)
hist = hist[hist.season_type.eq("REG") &
            hist.position_group.isin(["RB", "WR", "TE", "QB"])].copy()
for c in ("carries", "targets", "rushing_tds", "receiving_tds"):
    hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0.0)

# --- rates: career totals, empirical-Bayes shrunk to position ---------------
car = hist.groupby(["player_id", "player_display_name", "position_group"]).agg(
    c=("carries", "sum"), t=("targets", "sum"),
    rtd=("rushing_tds", "sum"), ctd=("receiving_tds", "sum")).reset_index()

def eb_k(x, n):
    ok = n > 20
    if ok.sum() < 30: return 100.0
    r = (x[ok] / n[ok]).astype(float); m, v = r.mean(), r.var()
    return float(np.clip(m * (1 - m) / v - 1, 20, 2000)) if v > 0 else 100.0

parts = []
for pos, s in car.groupby("position_group"):
    mu_r = s.rtd.sum() / max(s.c.sum(), 1); mu_c = s.ctd.sum() / max(s.t.sum(), 1)
    kr, kc = eb_k(s.rtd.values, s.c.values), eb_k(s.ctd.values, s.t.values)
    s = s.copy()
    s["r_rush"] = (s.rtd + kr * mu_r) / (s.c + kr)
    s["r_rec"] = (s.ctd + kc * mu_c) / (s.t + kc)
    parts.append(s)
rates = pd.concat(parts, ignore_index=True)

# --- usage: last OPP_WINDOW games of the most recent season ----------------
u = hist[hist.season == USAGE_SEASON].sort_values(["player_id", "week"])
u = u.groupby("player_id").tail(OPP_WINDOW)
use = u.groupby("player_id").agg(g=("week", "size"), team25=("team", "last"),
                                 mc=("carries", "mean"), mt=("targets", "mean")).reset_index()
use = use[use.g >= MIN_GAMES]

d = rates.merge(use, on="player_id", how="inner")
pm = d.groupby("position_group")[["mc", "mt"]].transform("mean")
W = np.minimum(d.g, OPP_WINDOW)
d["e_car"] = (d.mc * W + K_OPP * pm.mc) / (W + K_OPP)
d["e_tar"] = (d.mt * W + K_OPP * pm.mt) / (W + K_OPP)

# --- current team ----------------------------------------------------------
rp = Path("data/nfl/roster_weekly_2026.csv")
if rp.exists():
    r = pd.read_csv(rp, low_memory=False)
    idc = "gsis_id" if "gsis_id" in r.columns else "player_id"
    if "status" in r.columns:                 # drop IR / practice squad / cut
        before = len(r); r = r[r.status.astype(str).str.upper().eq("ACT")]
        print(f"rosters: {before - len(r)} non-active rows dropped")
    r = r.sort_values("week").groupby(idc).tail(1)[[idc, "team"]]
    d = d.merge(r.rename(columns={idc: "player_id", "team": "team26"}),
                on="player_id", how="left")
    d = d[d.team26.notna()]                   # not on a 2026 active roster -> gone
    d["team"] = d.team26
    d["moved"] = d.team26 != d.team25
    print(f"rosters: 2026 applied, {len(d):,} active players, "
          f"{int(d.moved.sum())} on a NEW TEAM (usage is from their old offence)\n")
else:
    d["team"] = d.team25; d["moved"] = False
    print("!! roster_weekly_2026.csv MISSING -- using 2025 teams. Free agents will\n"
          "   be on the wrong side. Fetch it before trusting this board.\n")

# --- environment: implied team points -> expected team TDs ------------------
env = []
for _, g in games.iterrows():
    env.append(dict(team=g.home, opp=g.away, site="H" if not g.neutral else "N",
                    implied=g.total / 2 + g.spread / 2, game=f"{g.away}@{g.home}"))
    env.append(dict(team=g.away, opp=g.home, site="A" if not g.neutral else "N",
                    implied=g.total / 2 - g.spread / 2, game=f"{g.away}@{g.home}"))
env = pd.DataFrame(env)
env["team_td"] = TD_A + TD_B * env.implied

b = d.merge(env, on="team", how="inner")
b["lam_raw"] = b.e_car * b.r_rush + b.e_tar * b.r_rec
b = b[b.lam_raw > 0]
tot = b.groupby("team").lam_raw.transform("sum")
b["lam"] = b.lam_raw * b.team_td / tot
b["p_td"] = 1 - np.exp(-b.lam)
b["fair"] = np.where(b.p_td > 0, np.where(b.p_td >= .5, -100*b.p_td/(1-b.p_td),
                                          100*(1-b.p_td)/b.p_td), np.nan)
b = b.sort_values("p_td", ascending=False)

print(f"{'player':<24}{'pos':<5}{'tm':<5}{'game':<10}{'car':>6}{'tar':>6}"
      f"{'impl':>7}{'P(TD)':>8}{'fair':>8}  flag")
print("-" * 94)
for _, r in b.head(45).iterrows():
    print(f"{r.player_display_name[:23]:<24}{r.position_group:<5}{r.team:<5}"
          f"{r.game:<10}{r.e_car:>6.1f}{r.e_tar:>6.1f}{r.implied:>7.1f}"
          f"{r.p_td*100:>7.1f}%{r.fair:>+8.0f}  {'NEW TEAM' if r.moved else ''}")
mv = b.head(45).moved.sum()
print(f"\n{mv} of the top 45 changed teams -- their usage estimate comes from a\n"
      f"different offence and is the least reliable number on this board.")
b.to_csv("data/nfl/td_board_w1.csv", index=False)
print(f"\n{len(b)} players priced -> data/nfl/td_board_w1.csv")
print("Compare P(TD) to the book's anytime-TD price. Bet only where the model is\n"
      "meaningfully higher -- and remember this has NEVER been tested against a\n"
      "real price. Paper-trade it before it sees money.")
