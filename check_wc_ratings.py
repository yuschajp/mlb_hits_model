import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from src import wc_data_client as client
from src.wc_model import compute_team_ratings, expected_goals, TOURNAMENT_AVG_GOALS, PRIOR_GAMES

completed = client.get_all_completed_matches()
print(f"Total completed matches: {len(completed)}\n")

ratings = compute_team_ratings(completed)

for team in ["Belgium", "Senegal"]:
    r = ratings.get(team, {})
    print(f"{team}: attack={r.get('attack')}  defense={r.get('defense')}  games={r.get('games')}")
    weight = r.get("games", 0) / (r.get("games", 0) + PRIOR_GAMES) if r.get("games") else 0
    print(f"  shrinkage weight on observed data: {weight:.3f}  "
          f"(tournament avg = {TOURNAMENT_AVG_GOALS})")

print("\nUnderlying matches for each team:")
for team in ["Belgium", "Senegal"]:
    print(f"\n{team}:")
    for m in completed:
        if m["home_team"] == team:
            print(f"  (H) {team} {m['home_goals']}-{m['away_goals']} {m['away_team']}")
        elif m["away_team"] == team:
            print(f"  (A) {m['home_team']} {m['home_goals']}-{m['away_goals']} {team}")

print("\nRecomputed xG for Belgium (home) vs Senegal (away):")
xg_h, xg_a = expected_goals("Belgium", "Senegal", ratings)
print(f"  xG Belgium={xg_h}  xG Senegal={xg_a}")
