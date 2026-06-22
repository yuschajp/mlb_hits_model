import sys
sys.path.insert(0, ".")
from src import mlb_api_client as c

lineups = c.get_confirmed_lineup(824263)
print("HOME LINEUP:")
for p in lineups["home"]:
    print(" ", p)
print("\nAWAY LINEUP:")
for p in lineups["away"]:
    print(" ", p)

pitchers = c.get_probable_pitchers(824263)
print("\nPITCHERS:", pitchers)
