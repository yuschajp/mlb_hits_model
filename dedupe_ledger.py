import pandas as pd
from pathlib import Path

path = Path("data/ledger/predictions_log.csv")
df = pd.read_csv(path, parse_dates=["date"])
before = len(df)

df = df.drop_duplicates(subset=["date", "game_pk", "player_id"], keep="last")
after = len(df)

df.to_csv(path, index=False)
print(f"Removed {before - after} duplicate row(s). Ledger now has {after} rows.")
