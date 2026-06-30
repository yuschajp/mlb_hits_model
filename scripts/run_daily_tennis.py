"""
run_daily_tennis.py

Generates match win probability predictions for today's Wimbledon
matches (or any active grass-court tournament) using surface-specific
Elo ratings computed from recent ATP/WTA history.

Run with: python3 scripts/run_daily_tennis.py
Specify tour: python3 scripts/run_daily_tennis.py --tour wta
Both tours:   python3 scripts/run_daily_tennis.py --tour both
"""

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import tennis_data_client as client
from src.tennis_model import compute_elo_ratings, predict_match

TENNIS_LEDGER = Path(__file__).resolve().parents[1] / "data" / "ledger" / "tennis_predictions_log.csv"
TENNIS_COLUMNS = [
    "date", "tour", "tournament", "round", "player_a", "player_b",
    "elo_a", "elo_b", "p_a_wins", "p_b_wins",
    "winner", "graded",
]


def _upsert(rows):
    TENNIS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()

    if TENNIS_LEDGER.exists():
        with open(TENNIS_LEDGER) as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["date"], row["player_a"], row["player_b"], row["round"]))

    new_rows = [
        r for r in rows
        if (r["date"], r["player_a"], r["player_b"], r["round"]) not in existing_keys
    ]

    write_header = not TENNIS_LEDGER.exists()
    with open(TENNIS_LEDGER, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TENNIS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    return len(new_rows)


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
    all_wimbledon_matches = client.get_wimbledon_draw(tour=tour)

    if not all_wimbledon_matches:
        print("  No Wimbledon matches found via ESPN.")
        return []

    today_str_compact = date.today().isoformat()
    # ESPN returns the whole tournament (qualifying through final) in one call.
    # Show today's matches plus any recent completed ones not yet graded.
    todays_matches = [m for m in all_wimbledon_matches if m["date"] == today_str_compact]
    other_matches  = [m for m in all_wimbledon_matches if m["date"] != today_str_compact]

    print(f"  {len(all_wimbledon_matches)} total matches in tournament so far "
          f"({len(todays_matches)} dated today, {len(other_matches)} other days).\n")

    wimbledon_matches = todays_matches if todays_matches else all_wimbledon_matches[-40:]
    if not todays_matches:
        print(f"  No matches dated today specifically -- showing the {len(wimbledon_matches)} most recent instead.\n")

    today_str = date.today().isoformat()
    rows = []

    completed = [m for m in wimbledon_matches if m.get("completed")]
    pending   = [m for m in wimbledon_matches if not m.get("completed")]

    print(f"  Found {len(completed)} completed and {len(pending)} in-progress/upcoming match(es):\n")

    for m in completed:
        winner, loser = m["winner"], m["loser"]
        pred = predict_match(winner, loser, overall_elo, surface_elo, surface_counts)

        row = {
            "date":       today_str,
            "tour":       tour,
            "tournament": "Wimbledon",
            "round":      m.get("round", ""),
            "player_a":   winner,
            "player_b":   loser,
            "elo_a":      pred["elo_a"],
            "elo_b":      pred["elo_b"],
            "p_a_wins":   pred["p_a_wins"],
            "p_b_wins":   pred["p_b_wins"],
            "winner":     winner,
            "graded":     True,
        }
        rows.append(row)

        favored = winner if pred["p_a_wins"] > 0.5 else loser
        upset = "⚡ UPSET" if pred["p_a_wins"] < 0.5 else ""
        print(f"  [{row['round']}] {winner} def. {loser}")
        print(f"    Elo: {pred['elo_a']:.0f} vs {pred['elo_b']:.0f}  |  "
              f"Model favored {favored} ({max(pred['p_a_wins'], pred['p_b_wins']):.1%})  {upset}")

    for m in pending:
        p1, p2 = m["player1"], m["player2"]
        if not p1 or not p2:
            continue
        pred = predict_match(p1, p2, overall_elo, surface_elo, surface_counts)

        row = {
            "date":       today_str,
            "tour":       tour,
            "tournament": "Wimbledon",
            "round":      m.get("round", ""),
            "player_a":   p1,
            "player_b":   p2,
            "elo_a":      pred["elo_a"],
            "elo_b":      pred["elo_b"],
            "p_a_wins":   pred["p_a_wins"],
            "p_b_wins":   pred["p_b_wins"],
            "winner":     None,
            "graded":     False,
        }
        rows.append(row)

        favored = p1 if pred["p_a_wins"] > 0.5 else p2
        print(f"  [{row['round']}] {p1} vs {p2}  (upcoming/in-progress)")
        print(f"    Elo: {pred['elo_a']:.0f} vs {pred['elo_b']:.0f}  |  "
              f"Model favors {favored} ({max(pred['p_a_wins'], pred['p_b_wins']):.1%})")

    added = _upsert(rows)
    print(f"\nLogged {added} new prediction(s) for {tour.upper()}.")
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
