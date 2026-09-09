"""Pull a season from the CFBD API into the shape fit_ratings wants.

    export CFBD_KEY=your_free_key          # collegefootballdata.com/key
    python3 cfb_fetch.py 2025

Writes, per year:
    data/cfb/games_<year>_all.csv   both divisions, with drive counts
    data/cfb/division_<year>.csv    team -> fbs / fcs

BOTH DIVISIONS, deliberately. The FBS-only pull cannot rate an FCS opponent, and
week 1 said the FBS/FCS bridge games were where the model beat the market. Games
where either side has zero recorded drives are dropped -- the model divides by
drives, so a zero is not a small number, it is a missing measurement.
"""
import csv, json, os, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://api.collegefootballdata.com"
KEY = os.environ.get("CFBD_KEY")


def get(path, **params):
    if not KEY:
        sys.exit("Set CFBD_KEY first:  export CFBD_KEY=your_key\n"
                 "Free key: https://collegefootballdata.com/key")
    q = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    req = urllib.request.Request(f"{BASE}{path}?{q}", headers={
        "Authorization": f"Bearer {KEY}", "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1} after {e}")
            time.sleep(2)


def pick(d, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def build(year):
    out = Path("data/cfb"); out.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {year}...")

    div = {}
    teams = get("/teams", year=year)          # division= is IGNORED by the API;
    for t in teams:                            # read classification off the record
        nm = pick(t, "school", "team")
        cl = str(pick(t, "classification", default="")).strip().lower()
        if nm and cl in ("fbs", "fcs"):
            div[nm] = cl
    from collections import Counter as _C
    print(f"  teams: {len(teams)} returned, classified {dict(_C(div.values()))}")

    games = get("/games", year=year, seasonType="regular")
    print(f"  games: {len(games)}")
    seen, uniq = set(), []
    for g in games:
        gid = pick(g, "id", "gameId")
        if gid is not None and gid not in seen:
            seen.add(gid); uniq.append(g)
    print(f"  unique games: {len(uniq)}")

    drives = get("/drives", year=year, seasonType="regular")
    print(f"  drives: {len(drives)}")
    dcount = defaultdict(int)
    for dr in drives:
        gid, off = pick(dr, "gameId", "game_id"), pick(dr, "offense")
        if gid is not None and off:
            dcount[(gid, off)] += 1

    rows, skipped = [], 0
    for g in uniq:
        gid = pick(g, "id", "gameId")
        h, a = pick(g, "homeTeam", "home_team"), pick(g, "awayTeam", "away_team")
        hp, ap = pick(g, "homePoints", "home_points"), pick(g, "awayPoints", "away_points")
        wk = pick(g, "week", default=0)
        if None in (gid, h, a, hp, ap):
            skipped += 1; continue
        hd, ad = dcount.get((gid, h), 0), dcount.get((gid, a), 0)
        if hd == 0 or ad == 0:
            skipped += 1; continue
        rows.append(dict(game_id=gid, week=int(wk), home_team=h, away_team=a,
                         home_points=hp, away_points=ap,
                         home_drives=hd, away_drives=ad,
                         neutral=bool(pick(g, "neutralSite", "neutral_site",
                                           default=False))))
    if not rows:
        sys.exit("  no usable games -- check the key and the year")

    gp = out / f"games_{year}_all.csv"
    with open(gp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # do NOT default unclassified teams to fcs -- that is what broke the split.
    missing = {r[k] for r in rows for k in ("home_team", "away_team")} - set(div)
    if missing:
        print(f"  {len(missing)} teams unclassified, left out of division file")
    dp = out / f"division_{year}.csv"
    with open(dp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["team", "division"])
        for t in sorted(div):
            w.writerow([t, div[t]])

    dr = [r["home_drives"] for r in rows] + [r["away_drives"] for r in rows]
    pts = [r["home_points"] for r in rows] + [r["away_points"] for r in rows]
    print(f"\n  wrote {len(rows)} games ({skipped} skipped) -> {gp}")
    print(f"  wrote {len(div)} teams -> {dp}")
    print(f"  drives/team/game {sum(dr)/len(dr):.1f}  (sanity 11-13)")
    print(f"  points/team/game {sum(pts)/len(pts):.1f}  (sanity 27-30)")
    print(f"  points per drive {sum(pts)/sum(dr):.2f}  (sanity 2.0-2.3)")


if __name__ == "__main__":
    for y in (sys.argv[1:] or ["2025"]):
        build(int(y)); print()
