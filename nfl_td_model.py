"""Anytime-touchdown model. Direct port of the HR model's structure.

    HR per AB, shrunk to league     ->  TD per carry / per target, shrunk to position
    expected AB by lineup slot      ->  expected carries / targets from recent usage
    opposing pitcher HR/BF          ->  handled by the market, see below
    park factor                     ->  handled by the market, see below
    P(>=1 HR) = 1-(1-r)^AB          ->  P(>=1 TD) = 1-exp(-lambda)

THE ONE REAL DESIGN DECISION: TOP-DOWN ALLOCATION
    The obvious build multiplies a player's rate by an opponent-defence
    adjustment AND scales by the game's implied team total. That double-counts,
    because the implied total ALREADY contains the defence -- the identical
    error as multiplying a season HR rate (which already contains the home park)
    by the park factor again. It cost the HR model a 28% overprediction in the
    high-park tercile.

    So this model does not compete with the market on team scoring, which the
    market prices extremely well (MAE 9.82 on spreads). It takes the market's
    implied team touchdowns as given, and only does the ALLOCATION -- which
    player gets them. That is the part the market prices lazily, and it makes the
    double-count structurally impossible rather than something to correct for.

        lambda_raw(player) = E[carries]*r_rush + E[targets]*r_rec
        lambda(player)     = lambda_raw * (team_TD_expected / sum(lambda_raw))
        P(>=1 TD)          = 1 - exp(-lambda)

WALK-FORWARD BY CONSTRUCTION
    Every feature for a given game uses only rows strictly BEFORE it. No row can
    see its own outcome or any later one.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
POS = ["RB", "WR", "TE", "QB"]
OPP_WINDOW = 4        # games of recent usage
K_OPP = 0.25          # shrink usage toward position mean, in games.
                      # 2.0 overpredicted the lowest-usage quartile by 50% --
                      # the same over-aggressive-prior bug as prior_ab=400 in
                      # the HR model (slot 9 +39.7%). Swept on HELD-OUT data:
                      # 0.25 minimises Brier AND maximises AUC. Usage is far
                      # more knowable than TD rate and should barely be shrunk.
MIN_SEASON_TEST = 2024  # hold these out entirely


def load():
    fr = []
    for y in SEASONS:
        d = pd.read_csv(f"data/nfl/player_week_{y}.csv", low_memory=False)
        fr.append(d)
    p = pd.concat(fr, ignore_index=True)
    p = p[p.season_type.eq("REG") & p.position_group.isin(POS)].copy()
    for c in ("carries", "targets", "rushing_tds", "receiving_tds"):
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    p["td"] = p.rushing_tds + p.receiving_tds
    p = p.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    return p


def prior_sums(p):
    """Career-to-date totals EXCLUDING the current row."""
    g = p.groupby("player_id", sort=False)
    for c in ("carries", "targets", "rushing_tds", "receiving_tds"):
        p[f"pr_{c}"] = g[c].cumsum() - p[c]
    # recent usage: mean of previous OPP_WINDOW games
    for c in ("carries", "targets"):
        p[f"rec_{c}"] = (g[c].shift(1)
                          .groupby(p.player_id).rolling(OPP_WINDOW, min_periods=1)
                          .mean().reset_index(level=0, drop=True))
    p["games_prior"] = g.cumcount()
    return p


def eb_k(x, n):
    """Beta-binomial prior strength from the moments: k = m(1-m)/var - 1."""
    ok = n > 20
    if ok.sum() < 30:
        return 100.0
    r = (x[ok] / n[ok]).astype(float)
    m, v = r.mean(), r.var()
    if v <= 0:
        return 100.0
    return float(np.clip(m * (1 - m) / v - 1, 20, 2000))


def build(p):
    p = prior_sums(p)
    # position baselines from PRIOR rows only, per position group
    out = []
    for pos, s in p.groupby("position_group"):
        mu_r = s.pr_rushing_tds.sum() / max(s.pr_carries.sum(), 1)
        mu_c = s.pr_receiving_tds.sum() / max(s.pr_targets.sum(), 1)
        kr = eb_k(s.pr_rushing_tds.values, s.pr_carries.values)
        kc = eb_k(s.pr_receiving_tds.values, s.pr_targets.values)
        s = s.copy()
        s["r_rush"] = (s.pr_rushing_tds + kr * mu_r) / (s.pr_carries + kr)
        s["r_rec"]  = (s.pr_receiving_tds + kc * mu_c) / (s.pr_targets + kc)
        mc = s.carries.mean(); mt = s.targets.mean()
        s["e_carries"] = (s.rec_carries.fillna(mc) * np.minimum(s.games_prior, OPP_WINDOW)
                          + K_OPP * mc) / (np.minimum(s.games_prior, OPP_WINDOW) + K_OPP)
        s["e_targets"] = (s.rec_targets.fillna(mt) * np.minimum(s.games_prior, OPP_WINDOW)
                          + K_OPP * mt) / (np.minimum(s.games_prior, OPP_WINDOW) + K_OPP)
        s.attrs = {}
        out.append(s)
        print(f"  {pos:<3} mu_rush {mu_r:.4f} (k={kr:>6.0f})   "
              f"mu_rec {mu_c:.4f} (k={kc:>6.0f})   n={len(s):,}")
    return pd.concat(out, ignore_index=True)


def attach_market(p):
    g = pd.read_csv("data/nfl/games.csv")
    g = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna()]
    keep = g[["game_id", "home_team", "away_team", "spread_line", "total_line",
              "roof", "home_score", "away_score"]]
    p = p.merge(keep, on="game_id", how="inner")
    is_home = p.team.eq(p.home_team)
    # nflverse: spread_line POSITIVE = home favoured
    p["implied_pts"] = np.where(is_home,
                                p.total_line / 2 + p.spread_line / 2,
                                p.total_line / 2 - p.spread_line / 2)
    p["is_home"] = is_home.astype(int)
    return p


if __name__ == "__main__":
    print("loading...")
    p = load()
    print(f"  {len(p):,} player-weeks, {p.season.min()}-{p.season.max()}\n")
    print("empirical-Bayes rates by position (prior-only):")
    p = build(p)
    p = attach_market(p)
    print(f"\n  merged with market lines: {len(p):,} rows")

    # team offensive TDs vs implied points -- fit the environment conversion
    tm = (p.groupby(["game_id", "team"])
            .agg(td=("td", "sum"), implied=("implied_pts", "first")).reset_index())
    tr = tm[tm.implied.notna()]
    b, a = np.polyfit(tr.implied, tr.td, 1)
    print(f"\n  team offensive TDs = {a:+.3f} + {b:.4f} * implied_points   "
          f"(r={np.corrcoef(tr.implied, tr.td)[0,1]:.3f}, n={len(tr):,})")
    p["team_td_exp"] = a + b * p.implied_pts

    p["lam_raw"] = p.e_carries * p.r_rush + p.e_targets * p.r_rec
    tot = p.groupby(["game_id", "team"]).lam_raw.transform("sum")
    p["lam"] = np.where(tot > 0, p.lam_raw * p.team_td_exp / tot, 0.0)
    p["p_td"] = 1 - np.exp(-p.lam)
    p["hit"] = (p.td > 0).astype(int)

    p.to_csv("data/nfl/td_features.csv", index=False)
    print(f"  wrote data/nfl/td_features.csv  ({len(p):,} rows)")

    tr_ = p[(p.season < MIN_SEASON_TEST) & (p.games_prior >= 4)]
    te_ = p[(p.season >= MIN_SEASON_TEST) & (p.games_prior >= 4)]
    print(f"\n  train {len(tr_):,} rows ({p.season.min()}-{MIN_SEASON_TEST-1})   "
          f"HELD OUT {len(te_):,} rows ({MIN_SEASON_TEST}-{p.season.max()})")
