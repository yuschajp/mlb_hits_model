#!/usr/bin/env python3
"""
Per-plate-appearance HR model.

Design notes:
  - Target is HR on a single PA (base rate ~3%), not HR-in-game.
    Game-level probability is derived at the end via 1-(1-p)^E[PA].
  - EVERY feature is built from STRICTLY PRIOR SEASONS. A feature computed
    on the same season it predicts leaks the outcome into the input and
    produces a fake-good result. This is the single easiest way to fool
    yourself in this problem.
  - Rate features are shrunk toward league mean via empirical Bayes using
    published stabilization points. A batter with 40 PA of history should
    not be trusted like one with 2000.
  - Logistic regression, not GBM. With a ~3% positive rate and heavy
    irreducible noise, interpretability is worth more than flexibility,
    and you need to be able to see which coefficients are doing work.

Usage:
    python3 hr_pa_model.py --data statcast_2021_2025.parquet --test-season 2025
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

# Stabilization points (PA or batted-ball events) from public research.
# Used as the shrinkage constant k in: shrunk = (x + k*league) / (n + k)
STAB_HR_PA = 170       # HR rate per PA
STAB_BARREL_BBE = 150  # barrel rate per batted-ball event
STAB_PITCHER_HR = 250  # pitcher HR per batter faced


def shrink(successes, trials, league_rate, k):
    """Empirical-Bayes shrinkage toward league rate with constant k."""
    return (successes + k * league_rate) / (trials + k)


def load(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    return df


def to_plate_appearances(df):
    """
    Collapse pitch-level Statcast to one row per plate appearance.

    Statcast is pitch-level. The final pitch of a PA carries the outcome in
    `events`. Rows where `events` is null are mid-PA pitches and are dropped.
    """
    required = ["game_date", "events", "batter", "pitcher", "stand",
                "p_throws", "home_team", "game_pk", "at_bat_number"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Missing expected Statcast columns: {missing}")

    pa = df[df["events"].notna()].copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa["season"] = pa["game_date"].dt.year
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)

    # Batted-ball events: needed as the denominator for barrel rate.
    pa["is_bbe"] = pa["launch_speed"].notna().astype(int) if "launch_speed" in pa else 0

    # Statcast ships a barrel flag on batted balls. Fall back to the
    # published EV/LA definition if the column is absent.
    if "barrel" in pa.columns:
        pa["is_barrel"] = pa["barrel"].fillna(0).astype(int)
    elif {"launch_speed", "launch_angle"}.issubset(pa.columns):
        ev = pa["launch_speed"].astype("float64")
        la = pa["launch_angle"].astype("float64")
        # Barrel window widens with exit velo: at 98 mph it is 26-30 deg,
        # and expands roughly 1 deg down / 1.5 deg up per additional mph.
        lo, hi = 26 - (ev - 98), 30 + 1.5 * (ev - 98)
        pa["is_barrel"] = (
            ((ev >= 98) & (la >= lo) & (la <= hi)).fillna(False).astype(int)
        )
        print("  [note] No `barrel` column; using simplified EV/LA proxy.")
    else:
        pa["is_barrel"] = 0
        print("  [warn] No launch data. Barrel feature will be inert.")

    pa["platoon_adv"] = (pa["stand"] != pa["p_throws"]).astype(int)
    return pa


def prior_season_features(pa, target_season):
    """
    Build batter / pitcher / park features from seasons STRICTLY BEFORE
    target_season. Returns three lookup frames.
    """
    hist = pa[pa["season"] < target_season]
    if hist.empty:
        sys.exit(f"No seasons before {target_season} to build features from.")

    lg_hr_pa = hist["is_hr"].mean()
    bbe = hist[hist["is_bbe"] == 1]
    lg_barrel = bbe["is_barrel"].mean() if len(bbe) else 0.0

    bat = hist.groupby("batter").agg(
        pa_n=("is_hr", "size"), hr_n=("is_hr", "sum")
    )
    bat_bbe = bbe.groupby("batter").agg(
        bbe_n=("is_barrel", "size"), barrel_n=("is_barrel", "sum")
    )
    bat = bat.join(bat_bbe).fillna({"bbe_n": 0, "barrel_n": 0})
    bat["bat_hr_rate"] = shrink(bat["hr_n"], bat["pa_n"], lg_hr_pa, STAB_HR_PA)
    bat["bat_barrel_rate"] = shrink(
        bat["barrel_n"], bat["bbe_n"], lg_barrel, STAB_BARREL_BBE
    )

    pit = hist.groupby("pitcher").agg(
        bf_n=("is_hr", "size"), hr_allowed=("is_hr", "sum")
    )
    pit["pit_hr_rate"] = shrink(
        pit["hr_allowed"], pit["bf_n"], lg_hr_pa, STAB_PITCHER_HR
    )

    # Park factor: HR rate at each home park relative to league, shrunk.
    # Crude -- it does not control for which hitters play there. Good
    # enough to test for signal; replace with a regressed version later.
    park = hist.groupby("home_team").agg(
        park_pa=("is_hr", "size"), park_hr=("is_hr", "sum")
    )
    park["park_factor"] = shrink(
        park["park_hr"], park["park_pa"], lg_hr_pa, 5000
    ) / lg_hr_pa

    return (
        bat[["bat_hr_rate", "bat_barrel_rate"]],
        pit[["pit_hr_rate"]],
        park[["park_factor"]],
        lg_hr_pa,
        lg_barrel,
    )


def assemble(pa, season, bat, pit, park, lg_hr_pa, lg_barrel):
    """Attach prior-season features to one season's PAs."""
    d = pa[pa["season"] == season].copy()
    d = d.join(bat, on="batter").join(pit, on="pitcher").join(park, on="home_team")

    # Players with no prior history fall back to league mean -- this is the
    # correct prior for an unknown, not a reason to drop the row.
    d["bat_hr_rate"] = d["bat_hr_rate"].fillna(lg_hr_pa)
    d["bat_barrel_rate"] = d["bat_barrel_rate"].fillna(lg_barrel)
    d["pit_hr_rate"] = d["pit_hr_rate"].fillna(lg_hr_pa)
    d["park_factor"] = d["park_factor"].fillna(1.0)

    # Log-odds space: these are multiplicative effects on a rate, so logs
    # are the natural additive scale for a linear model.
    d["log_bat_hr"] = np.log(d["bat_hr_rate"].clip(lower=1e-4))
    d["log_bat_barrel"] = np.log(d["bat_barrel_rate"].clip(lower=1e-4))
    d["log_pit_hr"] = np.log(d["pit_hr_rate"].clip(lower=1e-4))
    d["log_park"] = np.log(d["park_factor"].clip(lower=0.5))
    return d


FEATURES = ["log_bat_hr", "log_bat_barrel", "log_pit_hr", "log_park", "platoon_adv"]


def evaluate(y, p, label):
    """
    Report metrics that actually discriminate on a rare target.

    Brier score is near-useless here: predicting the base rate for every
    row scores well and carries zero information. Log loss and top-decile
    lift answer the real question -- do the players ranked highest homer
    more than the players ranked lowest?
    """
    base = np.full_like(p, y.mean())
    ll_model = log_loss(y, p, labels=[0, 1])
    ll_base = log_loss(y, base, labels=[0, 1])
    auc = roc_auc_score(y, p)

    order = np.argsort(p)
    n = len(p)
    bot = y.iloc[order[: n // 10]].mean()
    top = y.iloc[order[-(n // 10):]].mean()

    print(f"\n--- {label} (n={n:,}, base rate={y.mean():.4f}) ---")
    print(f"  log loss   model {ll_model:.5f}  |  base {ll_base:.5f}  "
          f"|  improvement {(ll_base - ll_model) / ll_base * 100:+.2f}%")
    print(f"  AUC        {auc:.4f}   (0.500 = no discrimination)")
    print(f"  top decile HR rate {top:.4f}  vs bottom decile {bot:.4f}  "
          f"| lift {top / bot:.2f}x" if bot > 0 else "")

    print("  calibration:")
    dfc = pd.DataFrame({"p": p, "y": y.values})
    dfc["bucket"] = pd.qcut(dfc["p"], 10, duplicates="drop")
    for b, g in dfc.groupby("bucket", observed=True):
        print(f"    {str(b):<22} n={len(g):>7,}  pred={g['p'].mean():.4f}  "
              f"actual={g['y'].mean():.4f}")
    return ll_model, ll_base, auc


def ablate(train, test, y_tr, y_te):
    """Drop each feature and watch the metric move. Anything that doesn't
    move it is not earning its place in the model."""
    print("\n--- feature ablation (log loss on test; higher = feature mattered) ---")
    full = LogisticRegression(max_iter=1000).fit(train[FEATURES], y_tr)
    ll_full = log_loss(y_te, full.predict_proba(test[FEATURES])[:, 1], labels=[0, 1])
    print(f"  all features: {ll_full:.5f}")
    for f in FEATURES:
        sub = [c for c in FEATURES if c != f]
        m = LogisticRegression(max_iter=1000).fit(train[sub], y_tr)
        ll = log_loss(y_te, m.predict_proba(test[sub])[:, 1], labels=[0, 1])
        print(f"  without {f:<18} {ll:.5f}  ({ll - ll_full:+.6f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Statcast parquet or csv")
    ap.add_argument("--test-season", type=int, required=True)
    args = ap.parse_args()

    print("Loading...")
    pa = to_plate_appearances(load(args.data))
    print(f"  {len(pa):,} plate appearances, seasons "
          f"{pa['season'].min()}-{pa['season'].max()}")

    test_season = args.test_season
    train_seasons = sorted(s for s in pa["season"].unique() if s < test_season)
    if len(train_seasons) < 2:
        sys.exit("Need at least 2 prior seasons: one to build features, one to fit.")

    print(f"\nFeatures built from seasons < each target season (no leakage).")
    print(f"Train: {train_seasons[1:]}  Test: {test_season}")

    # Each training season gets features from ITS OWN prior seasons.
    train_parts = []
    for s in train_seasons[1:]:
        bat, pit, park, lg, lgb = prior_season_features(pa, s)
        train_parts.append(assemble(pa, s, bat, pit, park, lg, lgb))
    train = pd.concat(train_parts, ignore_index=True)

    bat, pit, park, lg, lgb = prior_season_features(pa, test_season)
    test = assemble(pa, test_season, bat, pit, park, lg, lgb)

    y_tr, y_te = train["is_hr"], test["is_hr"]

    model = LogisticRegression(max_iter=1000).fit(train[FEATURES], y_tr)

    print("\n--- coefficients (log-odds per unit) ---")
    for f, c in zip(FEATURES, model.coef_[0]):
        print(f"  {f:<18} {c:+.4f}")
    print(f"  {'intercept':<18} {model.intercept_[0]:+.4f}")

    evaluate(y_te, model.predict_proba(test[FEATURES])[:, 1], f"TEST {test_season}")
    ablate(train, test, y_tr, y_te)

    print("\nTo convert per-PA to per-game: 1-(1-p_pa)**expected_PA")
    print("Expected PA by lineup slot runs ~4.6 (leadoff) down to ~3.8 (ninth).")
    print("\nRead the ablation before trusting anything. If AUC is near 0.50")
    print("or no feature moves log loss, there is no signal here yet and")
    print("more data will not create any.")


if __name__ == "__main__":
    main()
