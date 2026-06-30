"""
find_value_tennis.py

Compares Wimbledon match predictions against live odds from The Odds API.

Sport keys: tennis_atp_wimbledon, tennis_wta_wimbledon

Run with:
    export ODDS_API_KEY=your_key
    python3 scripts/find_value_tennis.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

TENNIS_LEDGER  = Path(__file__).resolve().parents[1] / "data" / "ledger" / "tennis_predictions_log.csv"
ODDS_BASE      = "https://api.the-odds-api.com/v4"
SPORT_KEYS     = {"atp": "tennis_atp_wimbledon", "wta": "tennis_wta_wimbledon"}
EDGE_THRESHOLD = 0.05
SUSPICIOUS     = 0.30
TIMEOUT        = 15


def _api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Set ODDS_API_KEY before running.")
    return key


def american_to_implied(american):
    if american > 0:
        return 100 / (american + 100)
    return (-american) / (-american + 100)


def get_events(sport_key):
    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/{sport_key}/events",
            params={"apiKey": _api_key()},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [warn] {e}")
        return []


def get_match_odds(sport_key, event_id):
    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/{sport_key}/events/{event_id}/odds",
            params={
                "apiKey": _api_key(),
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
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
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name  = outcome.get("name", "")
                price = outcome.get("price")
                if not name or price is None:
                    continue
                if name not in best or price > best[name]["price"]:
                    best[name] = {"price": price, "book": bk_key}
    return best


def normalize(name):
    return name.lower().strip().split()[-1]


def match_player(odds_name, model_name):
    return normalize(odds_name) == normalize(model_name)


def main():
    if not TENNIS_LEDGER.exists():
        print("No tennis predictions found. Run scripts/run_daily_tennis.py first.")
        return

    df = pd.read_csv(TENNIS_LEDGER)
    today = date.today()
    today_df = df[pd.to_datetime(df["date"]).dt.date == today]

    if today_df.empty:
        print("No tennis predictions for today.")
        return

    for tour in ["atp", "wta"]:
        tour_df = today_df[today_df["tour"] == tour]
        if tour_df.empty:
            continue

        sport_key = SPORT_KEYS[tour]
        print(f"\n{'='*60}")
        print(f"  {tour.upper()} VALUE")
        print(f"{'='*60}\n")

        events = get_events(sport_key)
        if not events:
            print(f"  No events found for {sport_key} (market may not be available pre-tournament)")
            continue

        value_rows     = []
        suspicious_rows = []

        for _, pred in tour_df.iterrows():
            player_a, player_b = pred["player_a"], pred["player_b"]

            event = None
            for e in events:
                eh, ea = e.get("home_team",""), e.get("away_team","")
                if (match_player(eh, player_a) or match_player(eh, player_b)) and \
                   (match_player(ea, player_a) or match_player(ea, player_b)):
                    event = e
                    break

            if not event:
                continue

            odds = get_match_odds(sport_key, event["id"])
            if not odds:
                continue

            for player, prob_col in [(player_a, "p_a_wins"), (player_b, "p_b_wins")]:
                matched_odds = next((v for k, v in odds.items() if match_player(k, player)), None)
                if not matched_odds:
                    continue

                model_p  = float(pred[prob_col])
                market_p = american_to_implied(matched_odds["price"])
                edge     = model_p - market_p

                row = {
                    "player":   player,
                    "opponent": player_b if player == player_a else player_a,
                    "model_p":  round(model_p, 4),
                    "market_p": round(market_p, 4),
                    "edge":     round(edge, 4),
                    "price":    matched_odds["price"],
                    "book":     matched_odds["book"],
                }
                if edge >= SUSPICIOUS:
                    suspicious_rows.append(row)
                elif edge >= EDGE_THRESHOLD:
                    value_rows.append(row)

        if suspicious_rows:
            print(f"⚠️  {len(suspicious_rows)} with edge >= {SUSPICIOUS:.0%} -- VERIFY:\n")
            for r in sorted(suspicious_rows, key=lambda x: -x["edge"]):
                print(f"  {r['player']:<25} vs {r['opponent']:<25} "
                      f"model={r['model_p']:.1%}  market={r['market_p']:.1%}  "
                      f"edge=+{r['edge']:.1%}  ({r['price']:+d} @ {r['book']})")

        if value_rows:
            value_rows.sort(key=lambda r: -r["edge"])
            print(f"{len(value_rows)} value pick(s):\n")
            for r in value_rows:
                print(f"  {r['player']:<25} vs {r['opponent']:<25} "
                      f"model={r['model_p']:.1%}  market={r['market_p']:.1%}  "
                      f"edge=+{r['edge']:.1%}  ({r['price']:+d} @ {r['book']})")
        elif not suspicious_rows:
            print("No value found.")


if __name__ == "__main__":
    main()
