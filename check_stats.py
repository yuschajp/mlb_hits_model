import sys
sys.path.insert(0, ".")
from src import mlb_api_client as c

print("Season hitting (Jacob Gonzalez, 694378):", c.get_season_hitting_stats(694378))
print("Last 30 days (694378):                 ", c.get_recent_hitting_stats(694378, days=30))
print("Splits vs RHP (694378):                ", c.get_splits_vs_hand(694378, "R"))
print("Pitcher stats against (Troy Melton, 675512):", c.get_pitcher_stats_against(675512))
