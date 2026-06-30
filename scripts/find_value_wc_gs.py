"""
find_value_wc_gs.py

Compares World Cup goalscorer predictions against anytime goalscorer
odds from The Odds API.

Sport key: soccer_fifa_world_cup
Market:    anytime_goalscorer (where available)

Run with:
    export ODDS_API_KEY=your_key
    python3 scripts/find_value_wc_gs.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

GS_LEDGER       = Path(__file__).resolve().parents[1] / "data" / "ledger" / "wc_gs_predictions_log.csv"
ODDS_BASE       = "https://api.the-odds-api.com/v4"
WC_SPORT_KEY    = "soccer_fifa_world_cup"
EDGE_THRESHOLD  = 0.05
SUSPICIOUS_EDGE = 0.20
TIMEOUT         = 15


def _api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Set ODDS_API_KEY before running.")
    return key


def american_to_implied(american):
    if american > 0:
        return 100 / (american + 100)
    return (-american) / (-american + 100)


def normalize(name):
    return name.lower().strip().split()[-1]


def match_player(odds_name, model_players):
    """Match by last name."""
    on = normalize(odds_name)
    for p in model_players:
        if normalize(p) == on:
            return p
    return None


def get_wc_events():
    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/{WC_SPORT_KEY}/events",
            params={"apiKey": _api_key()},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [warn] {e}")
        return []


def get_goalscorer_odds(event_id):
    """
    Returns {player_name: {"price": int, "book": str}} for anytime goalscorer market.
    """
    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/{WC_SPORT_KEY}/events/{event_id}/odds",
            params={
                "apiKey":      _api_key(),
                "regions":     "us",
                "markets":     "anytime_goalscorer",
                "oddsFormat":  "american",
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    best = {}
    for bk in data.get("bookmakers", []):
        bk_key = bk.get("key")
        for market in bk.get("markets", []):
            if market.get("key") != "anytime_goalscorer":
                continue
            for outcome in market.get("outcomes", []):
                name  = outcome.get("name", "")
                price = outcome.get("price")
                if not name or price is None:
                    continue
                if name not in best or price > best[name]["price"]:
                    best[name] = {"price": price, "book": bk_key}
    return best


def main():
    if not GS_LEDGER.exists():
        print("No goalscorer predictions. Run scripts/run_daily_wc_gs.py first.")
        return

    df = pd.read_csv(GS_LEDGER)
    today = date.today()
    today_df = df[pd.to_datetime(df["date"]).dt.date == today]

    if today_df.empty:
        print("No goalscorer predictions for today. Run scripts/run_daily_wc_gs.py first.")
        return

    print(f"Found {len(today_df)} goalscorer predictions for today.\n")

    events = get_wc_events()
    if not events:
        print(f"No events found for {WC_SPORT_KEY}.")
        return

    print(f"Checking {len(events)} WC events for anytime goalscorer odds...\n")

    value_rows     = []
    suspicious_rows = []
    no_market      = 0

    # Group predictions by match
    matches = today_df.groupby(["home_team", "away_team"])

    for (home, away), match_df in matches:
        # Find the event on The Odds API
        event = None
        for e in events:
            eh = e.get("home_team", "").lower()
            ea = e.get("away_team", "").lower()
            if (home.lower() in eh or eh in home.lower()) and \
               (away.lower() in ea or ea in away.lower()):
                event = e
                break

        if not event:
            print(f"  [no match] {home} vs {away} not found in odds feed")
            continue

        gs_odds = get_goalscorer_odds(event["id"])
        if not gs_odds:
            no_market += 1
            print(f"  [no market] {home} vs {away}: no anytime goalscorer odds available")
            continue

        print(f"  {home} vs {away} — {len(gs_odds)} players with odds")

        model_players = match_df["player"].tolist()
        pred_by_player = {r["player"]: r for _, r in match_df.iterrows()}

        for odds_name, info in gs_odds.items():
            matched = match_player(odds_name, model_players)
            if not matched:
                continue

            pred     = pred_by_player[matched]
            model_p  = float(pred["p_scores"])
            market_p = american_to_implied(info["price"])
            edge     = model_p - market_p

            row = {
                "player":   matched,
                "team":     pred["team"],
                "match":    f"{home} vs {away}",
                "model_p":  round(model_p, 4),
                "market_p": round(market_p, 4),
                "edge":     round(edge, 4),
                "price":    info["price"],
                "book":     info["book"],
                "lambda":   round(float(pred["lambda"]), 3),
            }

            if edge >= SUSPICIOUS_EDGE:
                suspicious_rows.append(row)
            elif edge >= EDGE_THRESHOLD:
                value_rows.append(row)

    print(f"\n  ({no_market} match(es) had no goalscorer market)")

    if suspicious_rows:
        print(f"\n⚠️  {len(suspicious_rows)} player(s) with edge >= {SUSPICIOUS_EDGE:.0%} -- VERIFY:\n")
        for r in sorted(suspicious_rows, key=lambda x: -x["edge"]):
            print(f"  {r['player']:<28} {r['match']:<35} "
                  f"model={r['model_p']:.1%}  market={r['market_p']:.1%}  "
                  f"edge=+{r['edge']:.1%}  ({r['price']:+d} @ {r['book']})")

    if not value_rows:
        print("\nNo clean goalscorer value found above the edge threshold today.")
        return

    value_rows.sort(key=lambda r: -r["edge"])
    print(f"\n{len(value_rows)} player(s) where model beats market by >= {EDGE_THRESHOLD:.0%}:\n")
    for r in value_rows:
        print(f"  {r['player']:<28} {r['match']:<35} "
              f"λ={r['lambda']:.3f}  model={r['model_p']:.1%}  "
              f"market={r['market_p']:.1%}  edge=+{r['edge']:.1%}  "
              f"({r['price']:+d} @ {r['book']})")


if __name__ == "__main__":
    main()
