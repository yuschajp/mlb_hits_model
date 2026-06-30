"""
run_daily_wc_gs.py

Generates World Cup anytime goalscorer predictions for today's matches.

Run with: python3 scripts/run_daily_wc_gs.py

No API key required -- uses openfootball worldcup.json.
"""

import csv
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import wc_data_client as client
from src.wc_goalscorer import extract_goalscorers, predict_match_scorers

GS_LEDGER  = Path(__file__).resolve().parents[1] / "data" / "ledger" / "wc_gs_predictions_log.csv"
GS_COLUMNS = [
    "date", "home_team", "away_team", "player", "team", "opponent",
    "tournament_goals", "tournament_minutes",
    "lambda", "p_scores", "actual_scored", "graded",
]


def _upsert(rows):
    GS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()

    if GS_LEDGER.exists():
        with open(GS_LEDGER) as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["date"], row["player"], row["home_team"], row["away_team"]))

    new_rows = [
        r for r in rows
        if (r["date"], r["player"], r["home_team"], r["away_team"]) not in existing_keys
    ]

    write_header = not GS_LEDGER.exists()
    with open(GS_LEDGER, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    return len(new_rows)


def main():
    print("Fetching World Cup data...\n")

    # Get all raw match data including goals1/goals2 arrays
    raw_data = client._fetch()
    all_raw_matches = raw_data.get("matches", [])

    # Build player stats from completed matches (those with score.ft)
    completed_raw = [
        m for m in all_raw_matches
        if isinstance(m.get("score"), dict) and m["score"].get("ft") is not None
    ]

    # Enrich with goals arrays
    completed_enriched = []
    for m in completed_raw:
        ft = m["score"]["ft"]
        team1 = m.get("team1", "")
        team2 = m.get("team2", "")
        if isinstance(team1, dict): team1 = team1.get("name", "")
        if isinstance(team2, dict): team2 = team2.get("name", "")
        completed_enriched.append({
            "home_team":  team1,
            "away_team":  team2,
            "home_goals": int(ft[0]),
            "away_goals": int(ft[1]),
            "goals1":     m.get("goals1", []),
            "goals2":     m.get("goals2", []),
        })

    print(f"  {len(completed_enriched)} completed matches for player rating computation.")
    player_stats, team_defense = extract_goalscorers(completed_enriched)
    print(f"  Tracked {len(player_stats)} players who scored in this tournament.\n")

    # Get today's matches
    todays_matches = client.get_todays_matches()
    upcoming       = client.get_upcoming_matches(days_ahead=0)
    all_today      = {m["match_id"]: m for m in todays_matches + upcoming}.values()

    if not all_today:
        print("No matches today.")
        return

    all_rows = []
    today_str = date.today().isoformat()

    for match in all_today:
        home = match["home_team"]
        away = match["away_team"]

        # Skip placeholder teams
        if "/" in home or "/" in away or home.startswith("1") or away.startswith("1"):
            continue

        print(f"  {home} vs {away}")
        preds = predict_match_scorers(home, away, player_stats, team_defense, min_p=0.10)

        for p in preds:
            all_rows.append({
                "date":                today_str,
                "home_team":           home,
                "away_team":           away,
                "player":              p["player"],
                "team":                p["team"],
                "opponent":            p["opponent"],
                "tournament_goals":    p["goals"],
                "tournament_minutes":  p["minutes"],
                "lambda":              p["lambda"],
                "p_scores":            p["p_scores"],
                "graded":              False,
            })

        # Print top scorers for this match
        for p in preds[:6]:
            print(f"    {p['player']:<28} {p['team']:<22} "
                  f"goals={p['goals']}  λ={p['lambda']:.3f}  "
                  f"P(scores)={p['p_scores']:.1%}")
        print()

    added = _upsert(all_rows)
    print(f"Logged {added} new goalscorer predictions to {GS_LEDGER}")


if __name__ == "__main__":
    main()
