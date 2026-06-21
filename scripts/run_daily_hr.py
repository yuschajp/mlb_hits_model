"""
run_daily_hr.py

Home run version of run_daily.py -- same daily workflow, same data
source, different model and ledger:
    1. pull today's MLB schedule (skipping already-final games)
    2. pull confirmed starting lineups + probable starters for each game
    3. compute HR probability for every confirmed starting batter
    4. upsert predictions into the HR ledger (separate file from the hits ledger)
    5. print a ranked summary

Run with: python3 scripts/run_daily_hr.py
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import mlb_api_client as client
from src.hr_model import hr_probability
from src.ledger import append_predictions, hr_columns
from src.park_factors import get_park_factor, load_park_factors

HR_LEDGER_PATH = Path(__file__).resolve().parents[1] / "data" / "ledger" / "hr_predictions_log.csv"
HR_PARK_FACTORS_PATH = Path(__file__).resolve().parents[1] / "data" / "hr_park_factors.csv"


def build_hr_predictions_for_game(game, hr_park_factors):
    game_pk = game["game_pk"]
    lineups = client.get_confirmed_lineup(game_pk)
    pitchers = client.get_probable_pitchers(game_pk)

    if not lineups["home"] and not lineups["away"]:
        print(f"  [skip] {game['away_team']} @ {game['home_team']}: lineups not posted yet")
        return []

    rows = []
    matchups = [
        ("home", game["home_team"], "away", game["away_team"]),
        ("away", game["away_team"], "home", game["home_team"]),
    ]
    for batting_side, team_name, pitching_side, opp_team_name in matchups:
        opp_pitcher = pitchers.get(pitching_side)
        if opp_pitcher is None:
            print(f"  [skip] {team_name} batters: no probable pitcher found for {opp_team_name}")
            continue

        park_factor = get_park_factor(team_name if batting_side == "home" else opp_team_name,
                                       factors=hr_park_factors)

        for batter in lineups[batting_side]:
            season = client.get_season_hr_stats(batter["player_id"])
            recent = client.get_recent_hr_stats(batter["player_id"], days=30)
            vs_hand = client.get_hr_splits_vs_hand(batter["player_id"], opp_pitcher["hand"])
            pitcher_line = client.get_pitcher_hr_stats_against(opp_pitcher["player_id"])

            p_hr, adjusted_hr_rate, expected_ab = hr_probability(
                season=season, recent=recent, vs_hand=vs_hand,
                lineup_spot=batter["lineup_spot"],
                pitcher_hr_allowed=pitcher_line[0], pitcher_ab_faced=pitcher_line[1],
                park_factor=park_factor,
            )

            rows.append({
                "date": date.today().isoformat(),
                "game_pk": game_pk,
                "player_id": batter["player_id"],
                "player_name": batter["name"],
                "team": team_name,
                "opponent": opp_team_name,
                "lineup_spot": batter["lineup_spot"],
                "venue": game["venue"],
                "park_factor": park_factor,
                "opponent_pitcher": opp_pitcher["name"],
                "p_hr": round(p_hr, 4),
                "adjusted_hr_rate": round(adjusted_hr_rate, 4),
                "expected_ab": expected_ab,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-raw", action="store_true", help="Print raw schedule JSON for today and exit.")
    args = parser.parse_args()

    if args.dump_raw:
        import json
        raw = client._get("/schedule", {"sportId": 1, "date": date.today().strftime("%Y-%m-%d")})
        print(json.dumps(raw, indent=2)[:4000])
        return

    hr_park_factors = load_park_factors(path=HR_PARK_FACTORS_PATH, value_col="hr_park_factor")
    games = client.get_schedule()
    print(f"Found {len(games)} games today ({date.today().isoformat()}).\n")

    finished = [g for g in games if g["status"] == "Final"]
    games = [g for g in games if g["status"] != "Final"]
    if finished:
        print(f"Skipping {len(finished)} already-final game(s).\n")

    all_rows = []
    for game in games:
        print(f"{game['away_team']} @ {game['home_team']} ({game['venue']}) [{game['status']}]")
        all_rows.extend(build_hr_predictions_for_game(game, hr_park_factors))

    if not all_rows:
        print("\nNo confirmed lineups available yet -- try again closer to first pitch.")
        return

    append_predictions(all_rows, HR_LEDGER_PATH, columns=hr_columns())

    ranked = sorted(all_rows, key=lambda r: r["p_hr"], reverse=True)
    print(f"\nTop 10 highest home-run-probability batters today:")
    for r in ranked[:10]:
        print(f"  {r['player_name']:<22} {r['team']:<20} P(HR)={r['p_hr']:.1%}  "
              f"vs {r['opponent_pitcher']} ({r['venue']})")

    print(f"\nLogged {len(all_rows)} HR predictions to {HR_LEDGER_PATH}")


if __name__ == "__main__":
    main()
