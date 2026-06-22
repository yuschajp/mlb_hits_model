import sys, json
sys.path.insert(0, ".")
from src import mlb_api_client as c

raw = c._get("/people/694378")
print(json.dumps(raw, indent=2)[:1500])
