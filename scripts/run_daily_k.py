"""
run_daily_k.py

Generates strikeout Over/Under probability predictions for every probable
starting pitcher today, then logs them to the K ledger.

Run with: python3 scripts/run_daily_k.py

The default line is 6.5 -- the most common strikeout total line offered by
sportsbooks. If you want predictions at a different line (e.g. 5.5 or 7.5),
use --line 5.5. The model computes the full distribution so any line can
be priced from a single run.

--- season_ip / recent_ip columns ---

Added after a bias diagnostic (diagnose_k_lambda_bias.py) on 68 graded
games found lambda_k is overpredicted specifically for below-average
season_k_per_9 pitchers (bias mostly vanishes once season_k_per_9 crosses
~8.9, the league average) -- consistent with the empirical-Bayes
shrinkage in k_model.py's stabilized_k_per_9 (prior_innings=50 for
season, 20 for recent) pulling weak-K pitchers too hard toward league
average. That hypothesis was only checkable indirectly through K/9
buckets because the ledger didn't record how many innings backed each
prediction -- shrinkage strength depends directly on innings pitched
(weight = ip / (ip + prior_innings)), not on K/9 itself. These two new
columns let future diagnostic runs regress bias directly against innings
pitched, which is the actual lever, instead of inferring it through K/9
as a proxy.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import mlb_api_client as client
from src.k_model import k_over_probability, LEAGUE_AVG_K_PER_9, LEAGUE_AVG_INNINGS
from src.park_factors import get_park_factor, load_park_factors

K_LEDGER_PATH = Path(__file__).resolve().parents[1] / "data" / "ledger" / "k_predictions_log.csv"
K_COLUMNS = [
    "date", "game_pk", "pitcher_id", "pitcher_name", "team", "opponent",
    "venue", "park_factor", "line",
    "season_k_per_9", "recent_k_per_9", "opp_k_rate",
    "season_ip", "recent_ip",
    "expected_innings", "lambda_k", "p_over", "p_under",
    "actual_ks", "graded",
]


def _upsert_predictions(rows, ledger_path):
    """Upsert keyed on (date, game_pk, pitcher_id, line)."""
    ledger_path = Path(ledger_path)
    df_new = pd.DataFrame(rows)
    for col in K_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[K_COLUMNS]
    df_new["graded"] = df_new["graded"].fillna(False)
    df_new["date"]   = pd.to_datetime(df_new["date"])

    key_cols = ["date", "game_pk", "pitcher_id", "line"]

    if ledger_path.exists():
        df_existing = pd.read_csv(ledger_path, parse_dates=["date"])
        # Old rows won't have season_ip/recent_ip -- add as NaN so
        # concat doesn't break. Diagnostic scripts should dropna on
        # these columns when they need them, same as any other column.
        for col in K_COLUMNS:
            if col not in df_existing.columns:
                df_existing[col] = None
        df_existing = df_existing[K_COLUMNS]
        new_keys  = set(map(tuple, df_new[key_cols].astype(str).to_numpy()))
        exist_keys = df_existing[key_cols].astype(str).apply(tuple, axis=1)
        df_existing = df_existing[~exist_keys.isin(new_keys)]
        combined = pd.concat([df_existing, df_new], ignore_index=True)
        combined.to_csv(ledger_path, mode="w", header=True, index=False)
    else:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(ledger_path, mode="w", header=True, index=False)


def build_k_prediction(game, pitcher, side, opp_team_id, line, park_factors):
    """
    Build a single strikeout prediction for one pitcher in one game.
    Returns a dict row or None if insufficient data.
    """
    pid = pitcher["player_id"]

    # Season K stats
    season_ks, season_ip = client.get_pitcher_season_k_stats(pid)
    season_k_per_9 = (season_ks / season_ip * 9) if season_ip > 0 else LEAGUE_AVG_K_PER_9

    # Recent K stats (last 35 days ≈ 5-6 starts)
    recent_ks, recent_ip = client.get_pitcher_recent_k_stats(pid, days=35)

    # Game log for average innings per start
    start_log = client.get_pitcher_start_log(pid)
    n_starts   = len(start_log)
    avg_ip     = (sum(s["ip"] for s in start_log) / n_starts) if n_starts > 0 else LEAGUE_AVG_INNINGS

    # Opponent team K rate
    opp_k_rate = client.get_team_k_rate(opp_team_id)

    # Park factor (hits-based as proxy -- no K-specific park factors yet)
    park_factor = get_park_factor(
        game["home_team"] if side == "home" else game["away_team"],
        factors=park_factors
    )

    p_over, p_under, lam = k_over_probability(
        season=(season_ks, season_ip),
        recent=(recent_ks, recent_ip),
        opp_k_rate=opp_k_rate,
        avg_innings=avg_ip,
        n_starts=n_starts,
        line=line,
    )

    return {
        "date":            date.today().isoformat(),
        "game_pk":         game["game_pk"],
        "pitcher_id":      pid,
        "pitcher_name":    pitcher["name"],
        "team":            game["away_team"] if side == "away" else game["home_team"],
        "opponent":        game["home_team"] if side == "away" else game["away_team"],
        "venue":           game["venue"],
        "park_factor":     park_factor,
        "line":            line,
        "season_k_per_9":  round(season_k_per_9, 3),
        "recent_k_per_9":  round((recent_ks / recent_ip * 9) if recent_ip > 0 else LEAGUE_AVG_K_PER_9, 3),
        "opp_k_rate":      round(opp_k_rate, 4),
        "season_ip":       round(season_ip, 1),
        "recent_ip":       round(recent_ip, 1),
        "expected_innings": round(avg_ip, 2),
        "lambda_k":        lam,
        "p_over":          p_over,
        "p_under":         p_under,
        "graded":          False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", type=float, default=6.5,
                        help="Strikeout over/under line to predict (default: 6.5)")
    args = parser.parse_args()

    park_factors = load_park_factors()
    games = client.get_schedule()
    print(f"Found {len(games)} games today ({date.today().isoformat()}).\n")

    finished = [g for g in games if g["status"] == "Final"]
    games    = [g for g in games if g["status"] != "Final"]
    if finished:
        print(f"Skipping {len(finished)} already-final game(s).\n")

    # Skip exhibition/All-Star games -- MLB Stats API has no season hitting
    # stats for composite All-Star rosters, so team-stat lookups 404.
    exhibition = [g for g in games if "All-Star" in g["home_team"] or "All-Star" in g["away_team"]]
    games = [g for g in games if g not in exhibition]
    if exhibition:
        print(f"Skipping {len(exhibition)} exhibition/All-Star game(s).\n")

    all_rows = []
    for game in games:
        pitchers = client.get_probable_pitchers(game["game_pk"])
        if not pitchers.get("home") and not pitchers.get("away"):
            print(f"  [skip] {game['away_team']} @ {game['home_team']}: no probable pitchers posted yet")
            continue

        print(f"{game['away_team']} @ {game['home_team']} ({game['venue']})")

        # Away pitcher faces home team batters
        if pitchers.get("away"):
            p = pitchers["away"]
            # Get home team ID for K rate lookup
            schedule_raw = client._get("/schedule", {
                "sportId": 1,
                "gamePk": game["game_pk"],
                "hydrate": "team",
            })
            home_team_id = None
            try:
                game_data = schedule_raw["dates"][0]["games"][0]
                home_team_id = game_data["teams"]["home"]["team"]["id"]
            except (KeyError, IndexError):
                pass

            if home_team_id:
                row = build_k_prediction(game, p, "away", home_team_id, args.line, park_factors)
                if row:
                    all_rows.append(row)
                    print(f"  {p['name']:<25} (away starter)  λ={row['lambda_k']:.1f}  "
                          f"P(over {args.line})={row['p_over']:.1%}")

        # Home pitcher faces away team batters
        if pitchers.get("home"):
            p = pitchers["home"]
            schedule_raw = client._get("/schedule", {
                "sportId": 1,
                "gamePk": game["game_pk"],
                "hydrate": "team",
            })
            away_team_id = None
            try:
                game_data = schedule_raw["dates"][0]["games"][0]
                away_team_id = game_data["teams"]["away"]["team"]["id"]
            except (KeyError, IndexError):
                pass

            if away_team_id:
                row = build_k_prediction(game, p, "home", away_team_id, args.line, park_factors)
                if row:
                    all_rows.append(row)
                    print(f"  {p['name']:<25} (home starter)  λ={row['lambda_k']:.1f}  "
                          f"P(over {args.line})={row['p_over']:.1%}")

    if not all_rows:
        print("\nNo probable starters available yet -- try again closer to first pitch.")
        return

    _upsert_predictions(all_rows, K_LEDGER_PATH)

    ranked = sorted(all_rows, key=lambda r: r["p_over"], reverse=True)
    print(f"\nTop strikeout over candidates today (line={args.line}):")
    for r in ranked[:10]:
        print(f"  {r['pitcher_name']:<25} {r['team']:<20}  "
              f"λ={r['lambda_k']:.1f}  P(over {args.line})={r['p_over']:.1%}  "
              f"vs {r['opponent']} ({r['venue']})")

    print(f"\nLogged {len(all_rows)} K predictions to {K_LEDGER_PATH}")


if __name__ == "__main__":
    main()
