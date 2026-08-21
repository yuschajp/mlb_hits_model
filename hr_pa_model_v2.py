#!/usr/bin/env python3
"""
Per-PA HR model, v2.

Changes from v1:

  1. PARK FACTOR IS DECONFOUNDED. v1 computed HR rate at each stadium
     using all PAs there, which includes the home team's own hitters in
     roughly half of them. Coors got credit for altitude AND for whoever
     Colorado employs. v2 uses VISITING batters only: across a season
     every park hosts a near-random cross-section of the league, so the
     residual difference is much closer to the park itself.

  2. RECENCY FEATURES ADDED, AS A TEST. HR counts over trailing 2/7/14
     day windows -- exactly the "hot streak" signal hand-tuned models
     lean on. Included specifically so the ablation can measure whether
     it carries information once true talent (barrel rate) is already
     in the model. Computed causally: strictly prior game dates only,
     same-day PAs excluded, since a bet is placed pregame.

  3. Per-game conversion using batting-order position derived from
     first-appearance order within each game.

Usage:
    python3 hr_pa_model_v2.py --data data/statcast_2021_2025.parquet \\
        --test-season 2025
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

STAB_HR_PA = 170
STAB_BARREL_BBE = 150
STAB_PITCHER_HR = 250
STAB_PARK = 3000        # park factors are noisy; shrink hard

RECENCY_WINDOWS = (2, 7, 14)


def shrink(successes, trials, league_rate, k):
    return (successes + k * league_rate) / (trials + k)


def load(path):
    return (pd.read_parquet(path) if path.endswith(".parquet")
            else pd.read_csv(path, low_memory=False))


def to_plate_appearances(df):
    required = ["game_date", "events", "batter", "pitcher", "stand",
                "p_throws", "home_team", "game_pk", "at_bat_number"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Missing Statcast columns: {missing}")

    pa = df[df["events"].notna()].copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa["season"] = pa["game_date"].dt.year
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_bbe"] = pa["launch_speed"].notna().astype(int)

    if "barrel" in pa.columns:
        pa["is_barrel"] = pa["barrel"].fillna(0).astype(int)
    else:
        ev = pa["launch_speed"].astype("float64")
        la = pa["launch_angle"].astype("float64")
        lo, hi = 26 - (ev - 98), 30 + 1.5 * (ev - 98)
        pa["is_barrel"] = (((ev >= 98) & (la >= lo) & (la <= hi))
                           .fillna(False).astype(int))
        print("  [note] No `barrel` column; using EV/LA proxy.")

    pa["platoon_adv"] = (pa["stand"] != pa["p_throws"]).astype(int)

    # Batter is on the away team when their team is not the home team.
    # Statcast has inning_topbot: top = away team batting.
    if "inning_topbot" in pa.columns:
        pa["batter_is_away"] = (pa["inning_topbot"] == "Top").astype(int)
    else:
        print("  [warn] No inning_topbot; park factors will stay confounded.")
        pa["batter_is_away"] = 0

    return pa


def add_recency(pa):
    """
    Trailing HR counts per batter, using strictly prior game dates.

    Same-day PAs are excluded entirely -- not just earlier PAs in the
    same game. A bet is placed before first pitch, so anything from
    today's game is unavailable at decision time. Including it would be
    leakage of exactly the kind that makes a model look predictive when
    it is only looking at the answer.
    """
    pa = pa.sort_values(["batter", "game_date"]).reset_index(drop=True)
    cols = {w: np.zeros(len(pa), dtype=np.int16) for w in RECENCY_WINDOWS}

    for _, idx in pa.groupby("batter", sort=False).indices.items():
        dates = pa["game_date"].values[idx]
        cum = np.concatenate([[0], np.cumsum(pa["is_hr"].values[idx])])
        # First row index sharing each PA's date -> cumulative HR before today
        before_today = np.searchsorted(dates, dates, side="left")
        for w in RECENCY_WINDOWS:
            start = np.searchsorted(
                dates, dates - np.timedelta64(w, "D"), side="left"
            )
            cols[w][idx] = cum[before_today] - cum[start]

    for w in RECENCY_WINDOWS:
        pa[f"hr_last_{w}d"] = cols[w]
    return pa


def add_lineup_slot(pa):
    """Batting order position from first-appearance order within a game."""
    pa = pa.sort_values(["game_pk", "batter_is_away", "at_bat_number"])
    first = (pa.groupby(["game_pk", "batter_is_away", "batter"], sort=False)
               ["at_bat_number"].transform("min"))
    pa["_first_ab"] = first
    slot = (pa.groupby(["game_pk", "batter_is_away"], sort=False)["_first_ab"]
              .rank(method="dense").astype(int))
    pa["lineup_slot"] = slot.clip(upper=9)
    return pa.drop(columns="_first_ab")


def prior_features(pa, target_season):
    hist = pa[pa["season"] < target_season]
    if hist.empty:
        sys.exit(f"No seasons before {target_season}.")

    lg_hr = hist["is_hr"].mean()
    bbe = hist[hist["is_bbe"] == 1]
    lg_barrel = bbe["is_barrel"].mean()

    bat = hist.groupby("batter").agg(pa_n=("is_hr", "size"),
                                     hr_n=("is_hr", "sum"))
    bb = bbe.groupby("batter").agg(bbe_n=("is_barrel", "size"),
                                   barrel_n=("is_barrel", "sum"))
    bat = bat.join(bb).fillna({"bbe_n": 0, "barrel_n": 0})
    bat["bat_hr_rate"] = shrink(bat["hr_n"], bat["pa_n"], lg_hr, STAB_HR_PA)
    bat["bat_barrel_rate"] = shrink(bat["barrel_n"], bat["bbe_n"],
                                    lg_barrel, STAB_BARREL_BBE)

    pit = hist.groupby("pitcher").agg(bf_n=("is_hr", "size"),
                                      hr_allowed=("is_hr", "sum"))
    pit["pit_hr_rate"] = shrink(pit["hr_allowed"], pit["bf_n"],
                                lg_hr, STAB_PITCHER_HR)

    # VISITING batters only -- see module docstring.
    away = hist[hist["batter_is_away"] == 1]
    lg_away_hr = away["is_hr"].mean() if len(away) else lg_hr
    park = away.groupby("home_team").agg(n=("is_hr", "size"),
                                         hr=("is_hr", "sum"))
    park["park_factor"] = shrink(park["hr"], park["n"],
                                 lg_away_hr, STAB_PARK) / lg_away_hr

    return (bat[["bat_hr_rate", "bat_barrel_rate"]], pit[["pit_hr_rate"]],
            park[["park_factor"]], lg_hr, lg_barrel)


def assemble(pa, season, bat, pit, park, lg_hr, lg_barrel):
    d = pa[pa["season"] == season].copy()
    d = d.join(bat, on="batter").join(pit, on="pitcher").join(park, on="home_team")
    d["bat_hr_rate"] = d["bat_hr_rate"].fillna(lg_hr)
    d["bat_barrel_rate"] = d["bat_barrel_rate"].fillna(lg_barrel)
    d["pit_hr_rate"] = d["pit_hr_rate"].fillna(lg_hr)
    d["park_factor"] = d["park_factor"].fillna(1.0)

    d["log_bat_hr"] = np.log(d["bat_hr_rate"].clip(lower=1e-4))
    d["log_bat_barrel"] = np.log(d["bat_barrel_rate"].clip(lower=1e-4))
    d["log_pit_hr"] = np.log(d["pit_hr_rate"].clip(lower=1e-4))
    d["log_park"] = np.log(d["park_factor"].clip(lower=0.5))
    return d


BASE_FEATURES = ["log_bat_hr", "log_bat_barrel", "log_pit_hr",
                 "log_park", "platoon_adv"]
RECENCY_FEATURES = [f"hr_last_{w}d" for w in RECENCY_WINDOWS]
ALL_FEATURES = BASE_FEATURES + RECENCY_FEATURES


def evaluate(y, p, label):
    base = np.full_like(p, y.mean())
    ll_m = log_loss(y, p, labels=[0, 1])
    ll_b = log_loss(y, base, labels=[0, 1])
    order = np.argsort(p)
    n = len(p)
    bot = y.iloc[order[: n // 10]].mean()
    top = y.iloc[order[-(n // 10):]].mean()

    print(f"\n--- {label} (n={n:,}, base={y.mean():.4f}) ---")
    print(f"  log loss  {ll_m:.5f} vs base {ll_b:.5f}  "
          f"({(ll_b - ll_m) / ll_b * 100:+.2f}%)")
    print(f"  AUC       {roc_auc_score(y, p):.4f}")
    print(f"  decile 10 {top:.4f}  decile 1 {bot:.4f}  lift {top / bot:.2f}x")

    dfc = pd.DataFrame({"p": p, "y": y.values})
    dfc["b"] = pd.qcut(dfc["p"], 10, duplicates="drop")
    print("  calibration:")
    for b, g in dfc.groupby("b", observed=True):
        print(f"    n={len(g):>7,}  pred={g['p'].mean():.4f}  "
              f"actual={g['y'].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test-season", type=int, required=True)
    args = ap.parse_args()

    print("Loading...")
    pa = to_plate_appearances(load(args.data))
    print(f"  {len(pa):,} PAs, {pa['season'].min()}-{pa['season'].max()}")

    print("Building recency features (causal, prior dates only)...")
    pa = add_recency(pa)
    pa = add_lineup_slot(pa)

    ts = args.test_season
    train_seasons = sorted(s for s in pa["season"].unique() if s < ts)
    if len(train_seasons) < 2:
        sys.exit("Need >=2 prior seasons.")

    parts = []
    for s in train_seasons[1:]:
        parts.append(assemble(pa, s, *prior_features(pa, s)))
    train = pd.concat(parts, ignore_index=True)
    test = assemble(pa, ts, *prior_features(pa, ts))
    y_tr, y_te = train["is_hr"], test["is_hr"]

    print(f"\nTrain {train_seasons[1:]}  Test {ts}")

    model = LogisticRegression(max_iter=1000).fit(train[ALL_FEATURES], y_tr)
    print("\n--- coefficients ---")
    for f, c in zip(ALL_FEATURES, model.coef_[0]):
        print(f"  {f:<18} {c:+.4f}")

    p_te = model.predict_proba(test[ALL_FEATURES])[:, 1]
    evaluate(y_te, p_te, f"TEST {ts}")

    print("\n--- ablation (test log loss; larger delta = feature mattered) ---")
    ll_full = log_loss(y_te, p_te, labels=[0, 1])
    print(f"  all: {ll_full:.6f}")
    for f in ALL_FEATURES:
        sub = [c for c in ALL_FEATURES if c != f]
        m = LogisticRegression(max_iter=1000).fit(train[sub], y_tr)
        ll = log_loss(y_te, m.predict_proba(test[sub])[:, 1], labels=[0, 1])
        print(f"  without {f:<18} {ll:.6f}  ({ll - ll_full:+.6f})")

    # The direct test: does recency add anything on top of true talent?
    m_base = LogisticRegression(max_iter=1000).fit(train[BASE_FEATURES], y_tr)
    p_base = m_base.predict_proba(test[BASE_FEATURES])[:, 1]
    ll_base_only = log_loss(y_te, p_base, labels=[0, 1])
    print(f"\n--- recency block test ---")
    print(f"  base features only:      {ll_base_only:.6f}  "
          f"AUC {roc_auc_score(y_te, p_base):.4f}")
    print(f"  base + recency:          {ll_full:.6f}  "
          f"AUC {roc_auc_score(y_te, p_te):.4f}")
    print(f"  recency contributes:     {ll_base_only - ll_full:+.6f} log loss")

    print("\n--- per-game conversion ---")
    exp_pa = {1: 4.65, 2: 4.55, 3: 4.45, 4: 4.35, 5: 4.25,
              6: 4.15, 7: 4.05, 8: 3.95, 9: 3.85}
    test = test.assign(p_pa=p_te)
    for slot in range(1, 10):
        s = test[test["lineup_slot"] == slot]
        if s.empty:
            continue
        pg = 1 - (1 - s["p_pa"]) ** exp_pa[slot]
        print(f"  slot {slot}: mean p_game {pg.mean():.4f}  n={len(s):,}")


if __name__ == "__main__":
    main()
