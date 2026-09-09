"""Rewrite data/cfb/division_<year>.csv using the classification field on each
team record, instead of trusting the API's division= filter (which it ignores).

    export CFBD_KEY=...
    python3 cfb_divfix.py 2021 2022 2023 2024 2025

One /teams call per year -- no need to re-pull games or drives.
"""
import csv, json, os, sys, time, urllib.request
from collections import Counter
from pathlib import Path

KEY = os.environ.get("CFBD_KEY")
if not KEY:
    sys.exit("Set CFBD_KEY first")
OUT = Path("data/cfb")


def get(path, **p):
    q = "&".join(f"{k}={v}" for k, v in p.items() if v is not None)
    req = urllib.request.Request(f"https://api.collegefootballdata.com{path}?{q}",
        headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            if a == 2: raise
            print(f"  retry {a+1} after {e}"); time.sleep(2)


for year in (sys.argv[1:] or ["2025"]):
    year = int(year)
    rows = get("/teams", year=year)
    if not rows:
        print(f"{year}: no teams returned"); continue
    if year == int(sys.argv[1] if len(sys.argv) > 1 else 2025):
        print(f"  team record fields: {sorted(rows[0].keys())}\n")

    div = {}
    for t in rows:
        nm = t.get("school") or t.get("team")
        cl = (t.get("classification") or "").strip().lower()
        if nm and cl in ("fbs", "fcs"):
            div[nm] = cl

    # any team that actually played in our games file but was not classified
    gp = OUT / f"games_{year}_all.csv"
    unknown = set()
    if gp.exists():
        with open(gp) as f:
            for r in csv.DictReader(f):
                for k in ("home_team", "away_team"):
                    if r[k] not in div:
                        unknown.add(r[k])

    p = OUT / f"division_{year}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["team", "division"])
        for t in sorted(div):
            w.writerow([t, div[t]])

    c = Counter(div.values())
    print(f"{year}: fbs {c['fbs']}  fcs {c['fcs']}  -> {p}")
    if unknown:
        print(f"   {len(unknown)} teams in games but unclassified "
              f"(likely D-II/III opponents, will be dropped): "
              f"{', '.join(sorted(unknown)[:6])}{' ...' if len(unknown) > 6 else ''}")
