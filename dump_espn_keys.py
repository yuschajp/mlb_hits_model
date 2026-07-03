import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from src import tennis_data_client as client

data = client._espn_get("atp", "scoreboard", {})
events = data.get("events", [])
wimbledon_event = next(
    (e for e in events if "wimbledon" in e.get("name", "").lower()), None
)

grouping = next(
    g for g in wimbledon_event.get("groupings", [])
    if "men's singles" in g.get("grouping", {}).get("displayName", "").lower()
)
comp = grouping.get("competitions", [])[0]

print("Top-level keys in a competition object:")
print(list(comp.keys()))
print()

# Print any key whose name suggests round/stage info, and its value
for key in comp.keys():
    if any(word in key.lower() for word in ["round", "type", "stage", "phase"]):
        print(f"comp['{key}'] =")
        print(json.dumps(comp[key], indent=2))
        print()

print("Full raw competition object (untruncated):")
print(json.dumps(comp, indent=2))
