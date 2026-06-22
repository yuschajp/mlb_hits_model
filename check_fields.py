import sys, json
sys.path.insert(0, ".")
from src import mlb_api_client as c

raw = c._get("/game/824263/boxscore")
s = json.dumps(raw)

for key in ["batSide", "pitchHand", "battingOrder"]:
    idx = s.find(key)
    print(key, "->", s[idx:idx+90] if idx != -1 else "NOT FOUND")
