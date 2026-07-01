import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from src import wc_data_client as client
from src.wc_goalscorer import extract_goalscorers, opponent_defense_factor

raw_data = client._fetch()
all_raw_matches = raw_data.get("matches", [])
completed_raw = [
    m for m in all_raw_matches
    if isinstance(m.get("score"), dict) and m["score"].get("ft") is not None
]
completed_enriched = []
for m in completed_raw:
    ft = m["score"]["ft"]
    team1 = m.get("team1", "")
    team2 = m.get("team2", "")
    if isinstance(team1, dict): team1 = team1.get("name", "")
    if isinstance(team2, dict): team2 = team2.get("name", "")
    completed_enriched.append({
        "home_team": team1, "away_team": team2,
        "home_goals": int(ft[0]), "away_goals": int(ft[1]),
        "goals1": m.get("goals1", []), "goals2": m.get("goals2", []),
    })

player_stats, team_defense = extract_goalscorers(completed_enriched)

print("All team_defense keys:")
for k in sorted(team_defense.keys()):
    print(f"  {repr(k)}")

print("\nToday's match team names:")
todays_matches = client.get_todays_matches()
upcoming = client.get_upcoming_matches(days_ahead=0)
all_today = {m["match_id"]: m for m in todays_matches + upcoming}.values()
for m in all_today:
    print(f"  home={repr(m['home_team'])}  away={repr(m['away_team'])}")
    print(f"    home in team_defense: {m['home_team'] in team_defense}")
    print(f"    away in team_defense: {m['away_team'] in team_defense}")

print("\nDefense factor comparison:")
for team in ["Bosnia & Herzegovina", "Senegal", "DR Congo", "USA", "Belgium", "England"]:
    d = team_defense.get(team, {})
    factor = opponent_defense_factor(team, team_defense)
    print(f"  {team:<22} matches={d.get('matches')}  conceded={d.get('goals_conceded')}  "
          f"factor={factor:.6f}")
