"""Pull NFL data from nflverse. No API key, no rate limits, no 401s.

    python3 nfl_fetch.py 2019 2020 2021 2022 2023 2024 2025

Writes:
    data/nfl/games.csv               every game 1999-present WITH CLOSING LINES
    data/nfl/player_week_<year>.csv  weekly player box lines

WHY games.csv MATTERS MORE THAN IT LOOKS
    It carries spread_line and total_line going back to 1999. The CFB build had
    no line history, so "does this model beat the market" could not be asked
    until real money had already been bet. Here it can be answered first.
    Note the sign convention: nflverse spread_line is HOME-perspective and
    POSITIVE means the home team is favoured -- the opposite of the CFB board.
    Everything downstream normalises to the CFB convention (negative = home fav)
    so the two projects do not drift apart.
"""
import io, sys, urllib.request
from pathlib import Path

OUT = Path("data/nfl"); OUT.mkdir(parents=True, exist_ok=True)
GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
REL = "https://github.com/nflverse/nflverse-data/releases/download"
PLAYER = [f"{REL}/stats_player/stats_player_week_{{y}}.csv",
          f"{REL}/player_stats/player_stats_{{y}}.csv"]   # fallback, older tag


def grab(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def save(data, path):
    path.write_bytes(data)
    n = data.count(b"\n")
    print(f"  {path}  {len(data)/1e6:.1f} MB  {n:,} rows")
    return n


print("games.csv (schedules + closing lines)")
try:
    save(grab(GAMES), OUT / "games.csv")
except Exception as e:
    sys.exit(f"  FAILED: {e}")

years = [int(y) for y in sys.argv[1:]] or [2025]
print("\nweekly player stats")
for y in years:
    got = False
    for pat in PLAYER:
        url = pat.format(y=y)
        try:
            save(grab(url), OUT / f"player_week_{y}.csv"); got = True; break
        except Exception as e:
            last = f"{url.rsplit('/', 1)[-1]}: {e}"
    if not got:
        print(f"  {y}: FAILED -- {last}")

print("\nsanity")
import csv
with open(OUT / "games.csv") as f:
    rows = list(csv.DictReader(f))
done = [r for r in rows if r.get("result") not in ("", "NA", None)]
lined = [r for r in done if r.get("spread_line") not in ("", "NA", None)]
seasons = sorted({int(r["season"]) for r in rows})
print(f"  {len(rows):,} games, seasons {seasons[0]}-{seasons[-1]}")
print(f"  {len(done):,} completed, {len(lined):,} of those carry a closing spread")
if lined:
    import statistics as st
    err = [abs(float(r["spread_line"]) - float(r["result"])) for r in lined[-2000:]]
    print(f"  market MAE on the last {len(err):,} lined games: {st.mean(err):.2f}")
    print("  (CFB market MAE was 11.92 -- this is the bar to beat, and it is lower)")
