"""
find_value_hr.py

Home run version of find_value.py. Compares today's HR predictions
(already logged by run_daily_hr.py) to live "Over 0.5 home runs" odds and
flags batters where the model beats the market.

Same conservative, non-de-vigged comparison approach as find_value.py --
see that file's docstring for the reasoning.

Run with: python3 scripts/find_value_hr.py
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import odds_client as odds
from src.ledger import hr_columns, load_ledger
from src.name_matching import build_name_index, match_name

HR_LEDGER_PATH = Path(__file__).resolve().parents[1] / "data" / "ledger" / "hr_predictions_log.csv"
VALUE_PICKS_PATH = Path(__file__).resolve().parents[1] / "data" / "value_picks_hr.json"
EDGE_THRESHOLD = 0.03  # smaller than the hits threshold (0.05) since HR probabilities are themselves smaller


def main():
    today = date.today()
    ledger = load_ledger(HR_LEDGER_PATH, columns=hr_columns())
    today_predictions = ledger[ledger["date"].dt.date == today]

    if today_predictions.empty:
        print(f"No HR predictions logged for today ({today.isoformat()}) -- run scripts/run_daily_hr.py first.")
        return

    name_index = build_name_index(today_predictions["player_name"].tolist())
    pred_by_name = {row["player_name"]: row for _, row in today_predictions.iterrows()}

    events = odds.get_todays_mlb_events()
    print(f"Found {len(events)} MLB events with odds today.\n")

    value_rows = []
    unmatched_count = 0
    matched_count = 0
    for event in events:
        try:
            best_odds = odds.get_best_over_0_5_hr_odds(event["event_id"])
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {event['away_team']} @ {event['home_team']}: {e}")
            continue

        for odds_player_name, info in best_odds.items():
            matched_name = match_name(odds_player_name, name_index)
            if matched_name is None:
                unmatched_count += 1
                continue
            matched_count += 1

            pred = pred_by_name[matched_name]
            implied_prob = odds.american_to_implied_prob(info["price"])
            edge = pred["p_hr"] - implied_prob

            if edge >= EDGE_THRESHOLD:
                value_rows.append({
                    "player_name": matched_name,
                    "team": pred["team"],
                    "model_p_hr": round(pred["p_hr"], 4),
                    "implied_prob": round(implied_prob, 4),
                    "edge": round(edge, 4),
                    "best_price": info["price"],
                    "bookmaker": info["bookmaker"],
                })

    total = matched_count + unmatched_count
    print(f"Matched {matched_count}/{total} odds-feed batters to a logged HR prediction "
          f"({len(today_predictions)} predictions were logged today).")

    if not value_rows:
        print("\nNo HR value found above the edge threshold today.")
        return

    value_rows.sort(key=lambda r: r["edge"], reverse=True)
    print(f"\n{len(value_rows)} batter(s) where the model beats the market by >= {EDGE_THRESHOLD:.0%}:\n")
    for r in value_rows:
        print(f"  {r['player_name']:<22} {r['team']:<20} model={r['model_p_hr']:.1%}  "
              f"market={r['implied_prob']:.1%}  edge=+{r['edge']:.1%}  "
              f"({r['best_price']:+d} @ {r['bookmaker']})")

    VALUE_PICKS_PATH.write_text(json.dumps({"date": today.isoformat(), "picks": value_rows}, indent=2))
    print(f"\nValue picks written to {VALUE_PICKS_PATH.name}")


if __name__ == "__main__":
    main()
