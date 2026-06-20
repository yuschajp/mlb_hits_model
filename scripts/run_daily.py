"""
run_daily.py

The daily entrypoint -- run this once a day (e.g. via cron, mid-morning,
after most lineups are posted) to:
    1. pull today's MLB schedule
    2. pull confirmed starting lineups + probable starters for each game
    3. compute hit probability for every confirmed starting batter
    4. append predictions to the ledger
    5. print a ranked summary

Run with: python3 scripts/run_daily.py
Inspect raw API responses first with: python3 scripts/run_daily.py --dump-raw

IMPORTANT: this needs live internet access (this sandbox doesn't have any,
so it's untested against the real API -- see mlb_api_client.py's docstring
for what to verify before trusting the output).
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import mlb_api_client as client
from src.hit_model import hit_probability
from src.ledger import append_predictions
from src.park_factors import get_park_factor, load_park_factors

LEDGER_PATH = Path(__file__).resolve().parents[1] / "data" / "ledger" / "predictions_log.csv"


def build_predictions_for_game(game, park_factors):
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
                                       factors=park_factors)

        for batter in lineups[batting_side]:
            season = client.get_season_hitting_stats(batter["player_id"])
            recent = client.get_recent_hitting_stats(batter["player_id"], days=30)
            vs_hand = client.get_splits_vs_hand(batter["player_id"], opp_pitcher["hand"])
            pitcher_line = client.get_pitcher_stats_against(opp_pitcher["player_id"])

            p_hit, adjusted_ba, expected_ab = hit_probability(
                season=season, recent=recent, vs_hand=vs_hand,
                lineup_spot=batter["lineup_spot"],
                pitcher_hits_allowed=pitcher_line[0], pitcher_ab_faced=pitcher_line[1],
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
                "p_hit": round(p_hit, 4),
                "adjusted_ba": round(adjusted_ba, 4),
                "expected_ab": expected_ab,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-raw", action="store_true",
                         help="Print raw schedule JSON for today and exit.")
    parser.add_argument("--dump-boxscore", type=int, metavar="GAME_PK",
                         help="Print raw boxscore JSON for a given game_pk and exit. "
                              "Use this to verify the battingOrder/pitchHand field paths.")
    parser.add_argument("--dump-player-stats", type=int, metavar="PLAYER_ID",
                         help="Print raw season/recent/splits stats JSON for a player and exit. "
                              "Use this to verify the people/stats field paths, especially sitCodes.")
    args = parser.parse_args()

    import json

    if args.dump_raw:
        raw = client._get("/schedule", {"sportId": 1, "date": date.today().strftime("%Y-%m-%d")})
        print(json.dumps(raw, indent=2)[:4000])
        return

    if args.dump_boxscore:
        raw = client._get(f"/game/{args.dump_boxscore}/boxscore")
        print(json.dumps(raw, indent=2)[:6000])
        print("\n... (truncated -- look for teams.home.players.{ID...}.battingOrder and .pitchHand)")
        return

    if args.dump_player_stats:
        pid = args.dump_player_stats
        print("--- season hitting ---")
        print(json.dumps(client._get(f"/people/{pid}/stats", {"stats": "season", "group": "hitting"}), indent=2)[:2000])
        print("\n--- vs RHP split (sitCodes=vr) ---")
        print(json.dumps(client._get(f"/people/{pid}/stats", {"stats": "statSplits", "sitCodes": "vr", "group": "hitting"}), indent=2)[:2000])
        return

    park_factors = load_park_factors()
    games = client.get_schedule()
    print(f"Found {len(games)} games today ({date.today().isoformat()}).\n")

    all_rows = []
    for game in games:
        print(f"{game['away_team']} @ {game['home_team']} ({game['venue']})")
        all_rows.extend(build_predictions_for_game(game, park_factors))

    if not all_rows:
        print("\nNo confirmed lineups available yet -- try again closer to first pitch.")
        return

    append_predictions(all_rows, LEDGER_PATH)

    ranked = sorted(all_rows, key=lambda r: r["p_hit"], reverse=True)
    print(f"\nTop 10 highest hit-probability batters today:")
    for r in ranked[:10]:
        print(f"  {r['player_name']:<22} {r['team']:<20} P(hit)={r['p_hit']:.1%}  "
              f"vs {r['opponent_pitcher']} ({r['venue']})")

    print(f"\nLogged {len(all_rows)} predictions to {LEDGER_PATH}")


if __name__ == "__main__":
    main()
