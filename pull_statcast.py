#!/usr/bin/env python3
"""
Pull Statcast pitch-level data, one season per file.

Statcast rate-limits hard, so this goes season by season, caches each to
its own parquet, and skips anything already on disk. If it dies partway
through, just run it again -- completed seasons are not re-fetched.

Expect ~700k-750k pitches per season, roughly 20-40 min for five seasons.

    python3 pull_statcast.py --start 2021 --end 2025
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from pybaseball import statcast, cache
except ImportError:
    sys.exit("pip install pybaseball --break-system-packages")

OUT_DIR = Path("data/statcast")

# Regular season only. Spring training and postseason have different
# roster/usage patterns and would contaminate rate estimates.
SEASON_WINDOWS = {
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-26", "2026-09-27"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--combine", action="store_true",
                    help="also write a single merged file at the end")
    args = ap.parse_args()

    cache.enable()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for season in range(args.start, args.end + 1):
        path = OUT_DIR / f"statcast_{season}.parquet"
        if path.exists():
            print(f"{season}: cached ({path.stat().st_size / 1e6:.0f} MB), skipping")
            continue

        if season not in SEASON_WINDOWS:
            print(f"{season}: no date window defined, skipping")
            continue

        start, end = SEASON_WINDOWS[season]
        print(f"{season}: fetching {start} to {end} ... this takes a while")
        try:
            df = statcast(start_dt=start, end_dt=end)
        except Exception as e:
            print(f"{season}: FAILED ({e}). Re-run to retry this season.")
            continue

        if df is None or df.empty:
            print(f"{season}: returned no rows")
            continue

        df.to_parquet(path, index=False)
        print(f"{season}: {len(df):,} pitches -> {path}")

    if args.combine:
        files = sorted(OUT_DIR.glob("statcast_*.parquet"))
        if not files:
            sys.exit("Nothing to combine.")
        print(f"\nCombining {len(files)} seasons...")
        merged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out = Path("data") / f"statcast_{args.start}_{args.end}.parquet"
        merged.to_parquet(out, index=False)
        print(f"{len(merged):,} pitches -> {out}")
        print(f"\nNext: python3 hr_pa_model.py --data {out} --test-season {args.end}")


if __name__ == "__main__":
    main()
