"""
run_daily_tennis.py

Generates match win probability predictions for today's Wimbledon
matches (or any active grass-court tournament) using surface-specific
Elo ratings computed from recent ATP/WTA history.

Run with: python3 scripts/run_daily_tennis.py
Specify tour: python3 scripts/run_daily_tennis.py --tour wta
Both tours:   python3 scripts/run_daily_tennis.py --tour both

--- Bug fixes ---

1. WRONG DATE STAMPED ON EVERY ROW: previously every row (completed or
   pending) was stamped with today_str -- the date the SCRIPT ran, not
   the date the match actually happened. Combined with the fallback
   "no matches dated today -- show the most recent 40 instead", this
   meant re-running the script on a later day could re-log an
   already-completed match from days earlier as if it happened "today",
   inflating today_count and total_graded with the same real match
   counted again. Fixed: the row's date now comes from the match data
   itself (comp["date"] from ESPN, already present on every match dict),
   which is stable regardless of when the script happens to run.

2. ORDER-DEPENDENT DEDUP KEY: the upsert key was (date, player_a,
   player_b, round). For a pending match, player_a/player_b come from
   ESPN's competitor order (p1, p2); once the match completes,
   player_a/player_b become (winner, loser), which may not be the same
   order as (p1, p2). That meant a pending row and its own completed
   result could both persist as separate ledger rows instead of the
   completed one properly superseding the pending one. Fixed: the dedup
   key now uses an order-independent (frozenset of the two player names)
   so a completed result correctly replaces its own earlier pending row.

NOTE: these fixes prevent NEW duplication going forward. They do not
retroactively clean up rows already duplicated in the ledger under the
old logic -- see dedupe_ledger.py (or ask for a tennis-specific one-time
cleanup) if the existing CSV needs a pass to remove already-created
duplicates.
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


def _match_key(row):
    """
    Order-independent dedup key: same real-world match should collide
    regardless of which player got labeled player_a vs player_b, and
    regardless of whether this is the pending or completed version of
    that match.
    """
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

    # Keep existing rows EXCEPT any that a new row is about to supersede
    # (e.g. a completed result replacing its own earlier pending row).
    kept_existing = [r for r in existing_rows if _match_key(r) not in new_keys]

    write_header = True
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
        print(f"  (Each match below is still logged under its OWN actual date, not today's date,")
        print(f"  so this doesn't re-stamp old results as if they happened today.)\n")

    rows = []

    completed = [m for m in wimbledon_matches if m.get("completed")]
    pending   = [m for m in wimbledon_matches if not m.get("completed")]

    print(f"  Found {len(completed)} completed and {len(pending)} in-progress/upcoming match(es):\n")

    for m in completed:
        winner, loser = m["winner"], m["loser"]
        pred = predict_match(winner, loser, overall_elo, surface_elo, surface_counts)

        row = {
            "date":       m["date"],  # actual match date, not today's run date
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
            "date":       m["date"],  # actual scheduled/current match date, not today's run date
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
