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

if not wimbledon_event:
    print("No Wimbledon event found in ATP scoreboard response.")
    sys.exit(0)

print("Event name:", wimbledon_event.get("name"))
print("Number of groupings:", len(wimbledon_event.get("groupings", [])))
print()

for grouping in wimbledon_event.get("groupings", []):
    g_name = grouping.get("grouping", {}).get("displayName", "")
    comps = grouping.get("competitions", [])
    print(f"=== Grouping: {g_name}  ({len(comps)} competitions) ===")

    if comps:
        # Dump the FULL first competition so we can see every available
        # field -- this is what we need to find the real round label.
        print(json.dumps(comps[0], indent=2)[:3000])
        print("... (truncated)")
    print()
