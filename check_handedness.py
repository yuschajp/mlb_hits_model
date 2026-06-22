import sys, json
sys.path.insert(0, ".")
from src import mlb_api_client as c

result = c.get_handedness_batch([694378])
print(result)
