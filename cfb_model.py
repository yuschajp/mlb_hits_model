"""College football score model -- opponent-adjusted points per drive.

STRUCTURAL PARALLEL TO THE HR MODEL
    HR rate per AB      ->  points per drive (efficiency)
    expected AB by slot ->  expected drives  (pace)
    pitcher adjustment  ->  opponent defensive rating
    park factor         ->  home-field advantage
    shrink to league    ->  ridge penalty + preseason prior

WHY RIDGE AND NOT SEASON AVERAGES
    130+ FBS teams play 12 games each, mostly inside their own conference. The
    opponent graph is sparse, so raw points-per-game is dominated by schedule.
    A joint fit solves every team's offence and defence at once against a shared
    scale. That is the whole ball game in CFB and has no baseball equivalent.

PARAMS ARE TUNED, NOT DEFAULT
    ridge_lambda 12.0 -> 2.0   over-shrunk; 2.0 measured better on walk-forward
    cap_ppd       4.0 -> 5.0   4.0 clipped legitimate blowouts
    yoy_regression .38 -> 0.0  optimum measured at ZERO. Ridge already shrinks;
                               regressing again double-shrinks -- the same error
                               as the park-factor double count in the HR model.
"""
import numpy as np
import pandas as pd

PARAMS = dict(
    ridge_lambda   = 2.0,
    hfa_points     = 2.6,
    cap_ppd        = 5.0,
    prior_weight_g = 4.0,
    yoy_regression = 0.0,
    league_ppd     = 2.05,
    league_drives  = 12.2,
)


def fit_ratings(games, params=PARAMS, prior=None):
    """Joint ridge fit of per-team offensive and defensive points-per-drive.

    games needs: home_team away_team home_points away_points
                 home_drives away_drives [neutral]
    Returns a frame indexed by team with off_ppd / def_ppd, both centred so
    0 = league average. Positive off = good offence; positive def = good defence.
    """
    teams = sorted(set(games.home_team) | set(games.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    rows, y, w = [], [], []
    for _, g in games.iterrows():
        for side in ("home", "away"):
            opp = "away" if side == "home" else "home"
            pts, drv = g[f"{side}_points"], g[f"{side}_drives"]
            if not drv or drv <= 0:
                continue
            ppd = min(pts / drv, params["cap_ppd"])
            r = np.zeros(2 * n + 1)
            r[idx[g[f"{side}_team"]]] = 1.0
            r[n + idx[g[f"{opp}_team"]]] = -1.0
            if not g.get("neutral", False):
                r[2 * n] = 1.0 if side == "home" else -1.0
            rows.append(r); y.append(ppd - params["league_ppd"]); w.append(drv)

    X = np.array(rows); y = np.array(y); w = np.array(w, dtype=float)
    W = np.diag(w / w.mean())

    P = np.eye(2 * n + 1) * params["ridge_lambda"]
    P[2 * n, 2 * n] = 0.0                       # never penalise HFA

    b = np.zeros(2 * n + 1)
    if prior is not None:
        pw = params["prior_weight_g"]
        for t, i in idx.items():
            if t in prior.index:
                b[i] = prior.loc[t, "off_ppd"] * pw
                b[n + i] = prior.loc[t, "def_ppd"] * pw
                P[i, i] += pw; P[n + i, n + i] += pw

    beta = np.linalg.solve(X.T @ W @ X + P, X.T @ W @ y + b)
    off = beta[:n] - beta[:n].mean()
    dfn = beta[n:2 * n] - beta[n:2 * n].mean()

    out = pd.DataFrame({"off_ppd": off, "def_ppd": dfn}, index=teams)
    out.attrs["hfa_ppd"] = beta[2 * n]
    return out


def team_pace(games, params=PARAMS):
    """Drives per game, shrunk toward the league mean."""
    rec = {}
    for _, g in games.iterrows():
        for side in ("home", "away"):
            rec.setdefault(g[f"{side}_team"], []).append(g[f"{side}_drives"])
    k = 4.0
    return pd.Series({t: (sum(v) + k * params["league_drives"]) / (len(v) + k)
                      for t, v in rec.items()})


def predict_game(ratings, pace, home, away, neutral=False, params=PARAMS):
    """Predicted score. Efficiency x opportunity -- rate x expected drives."""
    hfa_ppd = ratings.attrs.get("hfa_ppd",
                                params["hfa_points"] / params["league_drives"])
    drives = 0.5 * (pace.get(home, params["league_drives"]) +
                    pace.get(away, params["league_drives"]))

    def side(off_team, def_team, is_home):
        ppd = (params["league_ppd"] + ratings.loc[off_team, "off_ppd"]
               - ratings.loc[def_team, "def_ppd"])
        if not neutral:
            ppd += hfa_ppd if is_home else -hfa_ppd
        return max(0.0, ppd) * drives

    ph, pa = side(home, away, True), side(away, home, False)
    return dict(home_team=home, away_team=away,
                pred_home=round(ph, 1), pred_away=round(pa, 1),
                pred_spread=round(pa - ph, 1),     # negative = home favoured
                pred_total=round(ph + pa, 1),
                exp_drives=round(drives, 1))


def net_points(ratings, params=PARAMS):
    """Team rating in POINTS. net[A] - net[B] = expected neutral-field margin."""
    return ((ratings.off_ppd + ratings.def_ppd) * params["league_drives"]).round(3)


def carry_forward(final_ratings, params=PARAMS):
    r = final_ratings.copy()
    k = 1.0 - params["yoy_regression"]
    r["off_ppd"] *= k; r["def_ppd"] *= k
    return r
