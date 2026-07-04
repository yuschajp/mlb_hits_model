"""
find_value.py

Compares today's model predictions (already logged by run_daily.py) to
live "Over 0.5 hits" odds, and flags batters where the model's probability
beats what the best available price implies.

Requires ODDS_API_KEY to be set (see odds_client.py's docstring) and
today's predictions to already be in the ledger (run run_daily.py first).

NOTE: this uses the RAW (non-de-vigged) implied probability of the best
available price as the comparison bar, not a fair/de-vigged line -- that's
a deliberately conservative choice for v1, since de-vigging properly needs
the Over and Under price from the SAME bookmaker (the best Over price and
best Under price might come from different books, which isn't a valid pair
to de-vig together). Comparing against raw implied probability sets a
higher, safer bar since the vig is already baked into it. Roadmap: pull
matched Over/Under pairs per bookmaker and de-vig properly.

--- Bug fix: results were never saved anywhere ---

This script computed and printed value picks correctly, but never wrote
them to data/value_picks_hits.json -- meaning publish_dashboard.py's
Picks & Parlays tab always read a stale file from whenever that JSON was
last manually created (in practice, over a week old), regardless of how
often this script ran. Fixed by writing {"date": ..., "picks": [...]}
after computing value_rows, in the exact format _load_value_picks() in
publish_dashboard.py expects.

Only the vetted value_rows (edge in [EDGE_THRESHOLD, SUSPICIOUS_EDGE)) are
saved -- suspicious_rows (edge >= SUSPICIOUS_EDGE, flagged as likely a
market-type mismatch) are deliberately excluded from what gets persisted,
so a probable data error doesn't silently feed into the dashboard's
automated parlay builder. They still print to console with their warning,
same as before, for manual review.

The file is written even when value_rows is empty, with today's date and
an empty picks list -- otherwise a day with genuinely zero value picks
would leave yesterday's (or older) picks looking current on the
dashboard indefinitely, which is the exact staleness bug this fix closes.

Run with: python3 scripts/find_value.py
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import odds_client as odds
from src.ledger import load_ledger
from src.name_matching import build_name_index, match_name

LEDGER_PATH     = Path(__file__).resolve().parents[1] / "data" / "ledger" / "predictions_log.csv"
VALUE_PICKS_OUT = Path(__file__).resolve().parents[1] / "data" / "value_picks_hits.json"
EDGE_THRESHOLD = 0.05
SUSPICIOUS_EDGE = 0.20  # edges above this are almost certainly a wrong market type


def save_value_picks(value_rows, today):
    VALUE_PICKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": today.isoformat(),
        "picks": value_rows,
    }
    VALUE_PICKS_OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {len(value_rows)} value pick(s) to {VALUE_PICKS_OUT}")


def main():
    today = date.today()
    ledger = load_ledger(LEDGER_PATH)
    today_predictions = ledger[ledger["date"].dt.date == today]

    if today_predictions.empty:
        print(f"No predictions logged for today ({today.isoformat()}) -- run scripts/run_daily.py first.")
        return

    name_index = build_name_index(today_predictions["player_name"].tolist())
    pred_by_name = {row["player_name"]: row for _, row in today_predictions.iterrows()}

    events = odds.get_todays_mlb_events()
    print(f"Found {len(events)} MLB events with odds today.\n")

    value_rows = []
    suspicious_rows = []
    unmatched_count = 0
    matched_count = 0
    for event in events:
        try:
            best_odds = odds.get_best_over_0_5_hits_odds(event["event_id"])
        except Exception as e:  # noqa: BLE001 -- a single event's odds failing shouldn't kill the whole run
            print(f"  [skip] {event['away_team']} @ {event['home_team']}: {e}")
            continue

        for odds_player_name, info in best_odds.items():
            matched_name = match_name(odds_player_name, name_index)
            if matched_name is None:
                unmatched_count += 1
                print(f"  [no match] '{odds_player_name}' not found in today's predictions -- "
                      f"check name_matching.py if this happens a lot")
                continue
            matched_count += 1

            pred = pred_by_name[matched_name]
            implied_prob = odds.american_to_implied_prob(info["price"])
            edge = pred["p_hit"] - implied_prob

            row = {
                "player_name": matched_name,
                "team": pred["team"],
                "model_p_hit": round(pred["p_hit"], 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge, 4),
                "best_price": info["price"],
                "bookmaker": info["bookmaker"],
            }

            if edge >= SUSPICIOUS_EDGE:
                suspicious_rows.append(row)
            elif edge >= EDGE_THRESHOLD:
                value_rows.append(row)

    total = matched_count + unmatched_count
    print(f"\nMatched {matched_count}/{total} odds-feed batters to a logged prediction "
          f"({len(today_predictions)} predictions were logged today). If the unmatched count is "
          f"high, it's most likely games whose lineups weren't confirmed yet when run_daily.py ran -- "
          f"rerun run_daily.py closer to first pitch across the full slate, then rerun this script, "
          f"before concluding it's a name-formatting problem.")

    if suspicious_rows:
        suspicious_rows.sort(key=lambda r: r["edge"], reverse=True)
        print(f"\n⚠️  {len(suspicious_rows)} batter(s) with edge >= {SUSPICIOUS_EDGE:.0%} -- "
              f"VERIFY MARKET TYPE before acting (likely a first-inning or restricted prop, "
              f"not a full-game hit prop):\n")
        for r in suspicious_rows:
            print(f"  {r['player_name']:<22} {r['team']:<20} model={r['model_p_hit']:.1%}  "
                  f"market={r['implied_prob']:.1%}  edge=+{r['edge']:.1%}  "
                  f"({r['best_price']:+d} @ {r['bookmaker']})")

    value_rows.sort(key=lambda r: r["edge"], reverse=True)

    if not value_rows:
        print("\nNo clean value found above the edge threshold today.")
        save_value_picks(value_rows, today)
        return

    print(f"\n{len(value_rows)} batter(s) where the model beats the market by >= {EDGE_THRESHOLD:.0%}:\n")
    for r in value_rows:
        print(f"  {r['player_name']:<22} {r['team']:<20} model={r['model_p_hit']:.1%}  "
              f"market={r['implied_prob']:.1%}  edge=+{r['edge']:.1%}  "
              f"({r['best_price']:+d} @ {r['bookmaker']})")

    save_value_picks(value_rows, today)


if __name__ == "__main__":
    main()
