"""
find_value_wc.py

Compares World Cup match predictions to live odds and flags matches
where the model beats the market's implied probability.

Requires ODDS_API_KEY. The Odds API covers World Cup under
sport key "soccer_fifa_world_cup".

Run with: python3 scripts/find_value_wc.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

ODDS_BASE      = "https://api.the-odds-api.com/v4"
WC_SPORT_KEY   = "soccer_fifa_world_cup"
WC_LEDGER      = Path(__file__).resolve().parents[1] / "data" / "ledger" / "wc_predictions_log.csv"
EDGE_THRESHOLD = 0.05
SUSPICIOUS_EDGE = 0.25
TIMEOUT = 15


def _api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Set ODDS_API_KEY before running.")
    return key


def american_to_implied(american):
    if american > 0:
        return 100 / (american + 100)
    return (-american) / (-american + 100)


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
        print(f"  [warn] Could not fetch WC events: {e}")
        return []


def get_match_odds(event_id):
    """Returns 1X2 odds for a match."""
    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/{WC_SPORT_KEY}/events/{event_id}/odds",
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

    best = {"home": None, "draw": None, "away": None}
    home = data.get("home_team", "")
    away = data.get("away_team", "")

    for bookmaker in data.get("bookmakers", []):
        bk = bookmaker.get("key")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name  = outcome.get("name", "")
                price = outcome.get("price")
                if price is None:
                    continue
                if name == home:
                    side = "home"
                elif name == away:
                    side = "away"
                elif name.lower() in ("draw", "tie"):
                    side = "draw"
                else:
                    continue

                if best[side] is None or price > best[side]["price"]:
                    best[side] = {"price": price, "book": bk}

    return best


def normalize(name):
    return name.lower().strip()


def match_team(odds_name, model_teams):
    """Match by normalized name or last word."""
    on = normalize(odds_name)
    for t in model_teams:
        if normalize(t) == on:
            return t
        if normalize(t).split()[-1] == on.split()[-1]:
            return t
    return None


def main():
    if not WC_LEDGER.exists():
        print("No WC predictions found. Run scripts/run_daily_wc.py first.")
        return

    df = pd.read_csv(WC_LEDGER, parse_dates=["date"])
    today = date.today()
    today_df = df[df["date"].dt.date == today]

    if today_df.empty:
        print("No WC predictions for today. Run scripts/run_daily_wc.py first.")
        return

    print(f"Found {len(today_df)} WC match prediction(s) for today.\n")

    events = get_wc_events()
    if not events:
        print(f"No World Cup events found on The Odds API under '{WC_SPORT_KEY}'.")
        print("The sport key may differ -- checking available soccer keys...")
        try:
            resp = requests.get(
                f"{ODDS_BASE}/sports",
                params={"apiKey": _api_key()},
                timeout=TIMEOUT,
            )
            soccer = [s for s in resp.json() if "soccer" in s.get("key", "").lower()
                      or "fifa" in s.get("key", "").lower()
                      or "world" in s.get("title", "").lower()]
            for s in soccer:
                print(f"  {s['key']} — {s['title']}")
        except Exception:
            pass
        return

    value_rows     = []
    suspicious_rows = []

    for _, pred in today_df.iterrows():
        # Find matching event
        event = None
        for e in events:
            home_match = match_team(e.get("home_team", ""), [pred["home_team"]])
            away_match = match_team(e.get("away_team", ""), [pred["away_team"]])
            if home_match and away_match:
                event = e
                break

        if not event:
            print(f"  [no match] {pred['home_team']} vs {pred['away_team']} not found in odds feed")
            continue

        odds = get_match_odds(event["id"])
        if not any(odds.values()):
            print(f"  [no odds] {pred['home_team']} vs {pred['away_team']}")
            continue

        print(f"  {pred['home_team']} vs {pred['away_team']}:")
        for side, prob_col in [("home", "p_home"), ("draw", "p_draw"), ("away", "p_away")]:
            info = odds.get(side)
            if not info:
                continue
            implied = american_to_implied(info["price"])
            model_p = float(pred[prob_col])
            edge    = model_p - implied

            label = {"home": pred["home_team"], "draw": "Draw", "away": pred["away_team"]}[side]
            print(f"    {label:<28} model={model_p:.1%}  market={implied:.1%}  "
                  f"edge={edge:+.1%}  ({info['price']:+d} @ {info['book']})")

            row = {
                "match":    f"{pred['home_team']} vs {pred['away_team']}",
                "outcome":  label,
                "side":     side,
                "model_p":  round(model_p, 4),
                "implied":  round(implied, 4),
                "edge":     round(edge, 4),
                "price":    info["price"],
                "book":     info["book"],
            }

            if edge >= SUSPICIOUS_EDGE:
                suspicious_rows.append(row)
            elif edge >= EDGE_THRESHOLD:
                value_rows.append(row)

        print()

    if suspicious_rows:
        print(f"\n⚠️  {len(suspicious_rows)} outcome(s) with edge >= {SUSPICIOUS_EDGE:.0%} -- VERIFY:\n")
        for r in sorted(suspicious_rows, key=lambda x: x["edge"], reverse=True):
            print(f"  {r['outcome']:<30} model={r['model_p']:.1%}  "
                  f"market={r['implied']:.1%}  edge=+{r['edge']:.1%}  "
                  f"({r['price']:+d} @ {r['book']})")

    if not value_rows:
        print("No clean WC value found above the edge threshold today.")
        return

    value_rows.sort(key=lambda r: r["edge"], reverse=True)
    print(f"\n{len(value_rows)} outcome(s) where model beats market by >= {EDGE_THRESHOLD:.0%}:\n")
    for r in value_rows:
        print(f"  {r['match']:<45} {r['outcome']:<15}  "
              f"model={r['model_p']:.1%}  market={r['implied']:.1%}  "
              f"edge=+{r['edge']:.1%}  ({r['price']:+d} @ {r['book']})")


if __name__ == "__main__":
    main()
