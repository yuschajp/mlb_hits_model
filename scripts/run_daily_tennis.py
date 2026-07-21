"""
run_daily_tennis.py

Generates match win probability predictions for today's Wimbledon
matches with prop betting predictions (games O/U, set scores, etc).
"""

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import tennis_data_client as client
from src.tennis_model import compute_elo_ratings, predict_match_with_props

TENNIS_LEDGER = Path(__file__).resolve().parents[1] / "data" / "ledger" / "tennis_predictions_log.csv"
TENNIS_COLUMNS = [
    "date", "tour", "tournament", "round", "player_a", "player_b",
    "elo_a", "elo_b", "p_a_wins", "p_b_wins",
    "game_win_prob_a", "game_total_expected", "game_total_over_p", "game_total_under_p",
    "p_a_2_0", "p_a_2_1", "p_b_2_0", "p_b_2_1",
    "sets_2", "sets_3",
    "winner", "graded",
]


def _match_key(row):
    return (row["date"], row["round"], frozenset([row["player_a"], row["player_b"]]))


def _upsert(rows):
    TENNIS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    existing_keys = set()

    if TENNIS_LEDGER.exists():
        with open(TENNIS_LEDGER) as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                existing_keys.add(_match_key(row))

    new_keys = {_match_key(r): r for r in rows}
    kept_existing = [r for r in existing_rows if _match_key(r) not in new_keys]

    def fill_missing_cols(row):
        return {col: row.get(col, "") for col in TENNIS_COLUMNS}

    kept_existing = [fill_missing_cols(r) for r in kept_existing]
    rows = [fill_missing_cols(r) for r in rows]

    with open(TENNIS_LEDGER, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TENNIS_COLUMNS)
        writer.writeheader()
        writer.writerows(kept_existing)
        writer.writerows(rows)

    added = sum(1 for k in new_keys if k not in existing_keys)
    updated = sum(1 for k in new_keys if k in existing_keys)
    return added, updated


def run_for_tour(tour):
    print(f"\n{'='*60}")
    print(f"  {tour.upper()} — Wimbledon Predictions")
    print(f"{'='*60}\n")

    print("Fetching recent match history for Elo computation...")
    recent_matches = client.get_recent_matches(tour=tour)
    print(f"  {len(recent_matches)} matches in recent history.")

    if not recent_matches:
        print(f"  No historical data available for {tour} -- skipping Elo computation.")
        print(f"  (Live Wimbledon matches below will use default 1500 Elo for all players.)")

    print("Computing overall and grass-surface Elo ratings...")
    overall_elo, surface_elo, match_counts, surface_counts = compute_elo_ratings(
        recent_matches, surface="Grass"
    )
    print(f"  Rated {len(overall_elo)} players, "
          f"{len(surface_counts)} with grass-court history.\n")

    print("Fetching Wimbledon matches from ESPN (full tournament draw)...")
    all_wimbledon_matches = client.get_summer_major_draw(tour=tour)

    if not all_wimbledon_matches:
        print("  No Wimbledon matches found via ESPN.")
        return []

    today_str_compact = date.today().isoformat()
    todays_matches = [m for m in all_wimbledon_matches if m["date"] == today_str_compact]
    other_matches  = [m for m in all_wimbledon_matches if m["date"] != today_str_compact]

    print(f"  {len(all_wimbledon_matches)} total matches in tournament so far "
          f"({len(todays_matches)} dated today, {len(other_matches)} other days).\n")

    wimbledon_matches = todays_matches if todays_matches else all_wimbledon_matches[-40:]
    if not todays_matches:
        print(f"  No matches dated today specifically -- showing the {len(wimbledon_matches)} most recent instead.\n")

    rows = []
    completed = [m for m in wimbledon_matches if m.get("completed")]
    pending   = [m for m in wimbledon_matches if not m.get("completed")]

    print(f"  Found {len(completed)} completed and {len(pending)} in-progress/upcoming match(es):\n")

    for m in completed:
        winner, loser = m["winner"], m["loser"]
        pred = predict_match_with_props(winner, loser, overall_elo, surface_elo, surface_counts, 
                                        best_of=3, num_sims=500)
        props = pred.get("props", {})
        game_ou = props.get("game_total_ou", {})
        set_scores = props.get("set_scores", {})
        sets_played = props.get("sets_played", {})

        row = {
            "date":       m["date"],
            "tour":       tour,
            "tournament": "Wimbledon",
            "round":      m.get("round", ""),
            "player_a":   winner,
            "player_b":   loser,
            "elo_a":      pred["elo_a"],
            "elo_b":      pred["elo_b"],
            "p_a_wins":   pred["p_a_wins"],
            "p_b_wins":   pred["p_b_wins"],
            "game_win_prob_a": props.get("game_win_prob_a", ""),
            "game_total_expected": game_ou.get("expected_total", ""),
            "game_total_over_p": game_ou.get("over", ""),
            "game_total_under_p": game_ou.get("under", ""),
            "p_a_2_0": set_scores.get("p_a_2_0", ""),
            "p_a_2_1": set_scores.get("p_a_2_1", ""),
            "p_b_2_0": set_scores.get("p_b_2_0", ""),
            "p_b_2_1": set_scores.get("p_b_2_1", ""),
            "sets_2": sets_played.get("sets_2", ""),
            "sets_3": sets_played.get("sets_3", ""),
            "winner":     winner,
            "graded":     True,
        }
        rows.append(row)

        favored = winner if pred["p_a_wins"] > 0.5 else loser
        upset = "⚡ UPSET" if pred["p_a_wins"] < 0.5 else ""
        print(f"  [{row['round']}] {winner} def. {loser}")
        print(f"    Elo: {pred['elo_a']:.0f} vs {pred['elo_b']:.0f}  |  "
              f"Model favored {favored} ({max(pred['p_a_wins'], pred['p_b_wins']):.1%})  {upset}")
        
        expected_games = game_ou.get('expected_total', 0)
        if expected_games:
            over_p = game_ou.get('over', 0)
            print(f"    Games: ~{expected_games:.0f} (O/U {over_p:.1%})")

    for m in pending:
        p1, p2 = m["player1"], m["player2"]
        if not p1 or not p2:
            continue
        pred = predict_match_with_props(p1, p2, overall_elo, surface_elo, surface_counts,
                                        best_of=3, num_sims=500)
        props = pred.get("props", {})
        game_ou = props.get("game_total_ou", {})
        set_scores = props.get("set_scores", {})
        sets_played = props.get("sets_played", {})

        row = {
            "date":       m["date"],
            "tour":       tour,
            "tournament": "Wimbledon",
            "round":      m.get("round", ""),
            "player_a":   p1,
            "player_b":   p2,
            "elo_a":      pred["elo_a"],
            "elo_b":      pred["elo_b"],
            "p_a_wins":   pred["p_a_wins"],
            "p_b_wins":   pred["p_b_wins"],
            "game_win_prob_a": props.get("game_win_prob_a", ""),
            "game_total_expected": game_ou.get("expected_total", ""),
            "game_total_over_p": game_ou.get("over", ""),
            "game_total_under_p": game_ou.get("under", ""),
            "p_a_2_0": set_scores.get("p_a_2_0", ""),
            "p_a_2_1": set_scores.get("p_a_2_1", ""),
            "p_b_2_0": set_scores.get("p_b_2_0", ""),
            "p_b_2_1": set_scores.get("p_b_2_1", ""),
            "sets_2": sets_played.get("sets_2", ""),
            "sets_3": sets_played.get("sets_3", ""),
            "winner":     None,
            "graded":     False,
        }
        rows.append(row)

        favored = p1 if pred["p_a_wins"] > 0.5 else p2
        print(f"  [{row['round']}] {p1} vs {p2}  (upcoming/in-progress)")
        print(f"    Elo: {pred['elo_a']:.0f} vs {pred['elo_b']:.0f}  |  "
              f"Model favors {favored} ({max(pred['p_a_wins'], pred['p_b_wins']):.1%})")
        
        expected_games = game_ou.get('expected_total', 0)
        if expected_games:
            over_p = game_ou.get('over', 0)
            p_2_0 = set_scores.get('p_a_2_0', 0) + set_scores.get('p_b_2_0', 0)
            p_3 = sets_played.get('sets_3', 0)
            print(f"    Games: ~{expected_games:.0f} (O/U {over_p:.1%})")
            print(f"    Sets: 2-0 or 0-2 = {p_2_0:.1%}, goes to 3 = {p_3:.1%}")

    added, updated = _upsert(rows)
    print(f"\n{added} new prediction(s), {updated} existing prediction(s) refreshed "
          f"(e.g. pending -> completed) for {tour.upper()}.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", choices=["atp", "wta", "both"], default="both")
    args = parser.parse_args()

    tours = ["atp", "wta"] if args.tour == "both" else [args.tour]

    for tour in tours:
        run_for_tour(tour)


if __name__ == "__main__":
    main()
