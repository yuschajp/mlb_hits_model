"""
run_daily_wc.py

Generates World Cup match outcome predictions for today's and upcoming
matches, based on team ratings computed from all tournament results so far.

Run with: python3 scripts/run_daily_wc.py

Requires FOOTBALL_DATA_API_KEY environment variable.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import wc_data_client as client
from src.wc_model import compute_team_ratings, predict_match

WC_LEDGER  = Path(__file__).resolve().parents[1] / "data" / "ledger" / "wc_predictions_log.csv"
WC_COLUMNS = [
    "date", "match_id", "home_team", "away_team", "stage", "group",
    "utc_date", "xg_home", "xg_away",
    "p_home", "p_draw", "p_away", "over_2_5", "btts",
    "actual_home_goals", "actual_away_goals", "actual_result", "graded",
]


def _upsert(rows, ledger_path):
    ledger_path = Path(ledger_path)
    df_new = pd.DataFrame(rows)
    for col in WC_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[WC_COLUMNS]
    df_new["graded"] = df_new["graded"].fillna(False)
    df_new["date"]   = pd.to_datetime(df_new["date"])

    key_cols = ["match_id"]
    if ledger_path.exists():
        df_existing = pd.read_csv(ledger_path, parse_dates=["date"])
        new_keys  = set(df_new[key_cols].astype(str).squeeze())
        exist_keys = df_existing[key_cols].astype(str).squeeze()
        df_existing = df_existing[~exist_keys.isin(new_keys)]
        combined = pd.concat([df_existing, df_new], ignore_index=True)
        combined.to_csv(ledger_path, mode="w", header=True, index=False)
    else:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(ledger_path, mode="w", header=True, index=False)


def main():
    print("Fetching World Cup match data...\n")

    # Build team ratings from all completed matches
    completed = client.get_all_completed_matches()
    print(f"  {len(completed)} completed matches found for rating computation.")
    ratings = compute_team_ratings(completed)
    print(f"  Computed ratings for {len(ratings)} teams.\n")

    # Get today's matches + upcoming
    todays   = client.get_todays_matches()
    upcoming = client.get_upcoming_matches(days_ahead=3)

    # Deduplicate
    all_match_ids = set()
    all_matches   = []
    for m in todays + upcoming:
        if m["match_id"] not in all_match_ids:
            all_match_ids.add(m["match_id"])
            all_matches.append(m)

    if not all_matches:
        print("No upcoming World Cup matches found.")
        return

    print(f"Generating predictions for {len(all_matches)} match(es):\n")
    rows = []
    for m in all_matches:
        pred = predict_match(m["home_team"], m["away_team"], ratings)
        row = {
            "date":       date.today().isoformat(),
            "match_id":   m["match_id"],
            "home_team":  m["home_team"],
            "away_team":  m["away_team"],
            "stage":      m.get("stage", ""),
            "group":      m.get("group", ""),
            "utc_date":   m.get("utc_date", ""),
            "xg_home":    pred["xg_home"],
            "xg_away":    pred["xg_away"],
            "p_home":     pred["home"],
            "p_draw":     pred["draw"],
            "p_away":     pred["away"],
            "over_2_5":   pred["over_2_5"],
            "btts":       pred["btts"],
            "graded":     False,
        }
        rows.append(row)

        home_rating = ratings.get(m["home_team"], {})
        away_rating = ratings.get(m["away_team"], {})
        print(f"  {m['home_team']:<28} vs {m['away_team']}")
        print(f"    xG: {pred['xg_home']:.2f} - {pred['xg_away']:.2f}  |  "
              f"Home {pred['home']:.1%}  Draw {pred['draw']:.1%}  Away {pred['away']:.1%}  |  "
              f"O2.5: {pred['over_2_5']:.1%}")
        print(f"    Games played: {home_rating.get('games', 0)} / {away_rating.get('games', 0)}")
        print()

    _upsert(rows, WC_LEDGER)
    print(f"Logged {len(rows)} WC predictions to {WC_LEDGER}")


if __name__ == "__main__":
    main()
