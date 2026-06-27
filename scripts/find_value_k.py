"""
find_value_k.py

Compares today's strikeout predictions to live pitcher strikeout
Over/Under odds and flags pitchers where the model beats the market.

Requires ODDS_API_KEY. The Odds API market key for pitcher strikeouts
is "pitcher_strikeouts" -- availability varies by bookmaker.

Run with: python3 scripts/find_value_k.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.name_matching import normalize_name

K_LEDGER_PATH = Path(__file__).resolve().parents[1] / "data" / "ledger" / "k_predictions_log.csv"
ODDS_BASE     = "https://api.the-odds-api.com/v4"
EDGE_THRESHOLD    = 0.05
SUSPICIOUS_EDGE   = 0.20
TIMEOUT = 15


def _api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Set ODDS_API_KEY before running.")
    return key


def american_to_implied_prob(american):
    if american > 0:
        return 100 / (american + 100)
    return (-american) / (-american + 100)


def get_mlb_events():
    resp = requests.get(
        f"{ODDS_BASE}/sports/baseball_mlb/events",
        params={"apiKey": _api_key()},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_pitcher_k_odds(event_id, regions="us"):
    """
    Fetches pitcher_strikeouts market odds for an event.
    Returns {pitcher_name: {"over": {"price": ..., "point": ..., "book": ...},
                            "under": {"price": ..., "point": ..., "book": ...}}}
    """
    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/baseball_mlb/events/{event_id}/odds",
            params={
                "apiKey": _api_key(),
                "regions": regions,
                "markets": "pitcher_strikeouts",
                "oddsFormat": "american",
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {}

    best = {}
    for bookmaker in data.get("bookmakers", []):
        bk_key = bookmaker.get("key")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "pitcher_strikeouts":
                continue
            for outcome in market.get("outcomes", []):
                pitcher = outcome.get("description") or outcome.get("name")
                side    = outcome.get("name", "").lower()  # "Over" or "Under"
                price   = outcome.get("price")
                point   = outcome.get("point")
                if not pitcher or price is None or point is None:
                    continue
                if pitcher not in best:
                    best[pitcher] = {}
                direction = "over" if "over" in side else "under"
                if direction not in best[pitcher] or price > best[pitcher][direction]["price"]:
                    best[pitcher][direction] = {"price": price, "point": point, "book": bk_key}
    return best


def match_pitcher(odds_name, model_names):
    """Match by last name after normalization."""
    odds_last = normalize_name(odds_name).split()[-1]
    for name in model_names:
        if normalize_name(name).split()[-1] == odds_last:
            return name
    return None


def main():
    import pandas as pd

    today = date.today()
    if not Path(K_LEDGER_PATH).exists():
        print("No K predictions found. Run scripts/run_daily_k.py first.")
        return

    df = pd.read_csv(K_LEDGER_PATH, parse_dates=["date"])
    today_df = df[df["date"].dt.date == today]

    if today_df.empty:
        print(f"No K predictions for today ({today.isoformat()}) -- run scripts/run_daily_k.py first.")
        return

    print(f"Found {len(today_df)} pitcher predictions today.\n")
    model_names = today_df["pitcher_name"].tolist()
    pred_by_name = {row["pitcher_name"]: row for _, row in today_df.iterrows()}

    events = get_mlb_events()
    print(f"Checking odds across {len(events)} MLB events...\n")

    value_rows     = []
    suspicious_rows = []
    matched        = 0
    unmatched      = 0
    no_market      = 0

    for event in events:
        event_id = event.get("id")
        k_odds   = get_pitcher_k_odds(event_id)

        if not k_odds:
            no_market += 1
            continue

        for odds_name, sides in k_odds.items():
            matched_name = match_pitcher(odds_name, model_names)
            if matched_name is None:
                unmatched += 1
                continue
            matched += 1

            pred = pred_by_name[matched_name]
            model_line = float(pred["line"])
            p_over_model = float(pred["p_over"])

            # Check if the odds line matches the model's line
            over_info  = sides.get("over", {})
            under_info = sides.get("under", {})

            if not over_info:
                continue

            odds_line = over_info.get("point")
            if odds_line and abs(odds_line - model_line) > 0.5:
                # Lines don't match -- skip rather than compare apples to oranges
                continue

            implied_over  = american_to_implied_prob(over_info["price"])
            edge_over     = p_over_model - implied_over

            row = {
                "pitcher":    matched_name,
                "team":       pred["team"],
                "opponent":   pred["opponent"],
                "line":       odds_line or model_line,
                "model_p_over":  round(p_over_model, 4),
                "implied_over":  round(implied_over, 4),
                "edge_over":     round(edge_over, 4),
                "over_price":    over_info["price"],
                "over_book":     over_info["book"],
                "lambda_k":      round(float(pred["lambda_k"]), 2),
            }

            if abs(edge_over) >= SUSPICIOUS_EDGE:
                suspicious_rows.append(row)
            elif edge_over >= EDGE_THRESHOLD:
                value_rows.append(row)

    print(f"Matched {matched} pitchers to predictions. "
          f"({unmatched} unmatched, {no_market} events with no K market.)\n")

    if suspicious_rows:
        print(f"⚠️  {len(suspicious_rows)} pitcher(s) with edge >= {SUSPICIOUS_EDGE:.0%} -- VERIFY LINE:\n")
        for r in sorted(suspicious_rows, key=lambda x: x["edge_over"], reverse=True):
            print(f"  {r['pitcher']:<25} λ={r['lambda_k']:.1f}  "
                  f"model={r['model_p_over']:.1%}  market={r['implied_over']:.1%}  "
                  f"edge=+{r['edge_over']:.1%}  (over {r['line']} @ {r['over_price']:+d} {r['over_book']})")

    if not value_rows:
        print("No clean K value found above the edge threshold today.")
        return

    value_rows.sort(key=lambda r: r["edge_over"], reverse=True)
    print(f"{len(value_rows)} pitcher(s) where model beats market by >= {EDGE_THRESHOLD:.0%}:\n")
    for r in value_rows:
        print(f"  {r['pitcher']:<25} vs {r['opponent']:<20}  "
              f"λ={r['lambda_k']:.1f}  model={r['model_p_over']:.1%}  "
              f"market={r['implied_over']:.1%}  edge=+{r['edge_over']:.1%}  "
              f"(over {r['line']} @ {r['over_price']:+d} {r['over_book']})")


if __name__ == "__main__":
    main()
