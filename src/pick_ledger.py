"""
pick_ledger.py

Appends every priced prediction to data/ledger/pick_history.csv.

Stores the FULL board, not just value picks. Selecting on the model's own
disagreement with the market and then measuring edge on that subset is
circular -- keeping everything lets you ask the honest question later and
re-cut thresholds without refetching odds you can no longer get.

hr_predictions_log.csv has probabilities and outcomes but no prices, so it
measures calibration and cannot measure edge. This closes that gap.
"""

import csv
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "data" / "ledger" / "pick_history.csv"

FIELDS = ["date", "player_name", "team", "market", "model_prob",
          "implied_prob", "edge", "price_taken", "bookmaker",
          "price_close", "actual", "graded"]


def _key(r):
    return (r.get("date"), r.get("player_name"), r.get("market"))


def append_picks(rows, game_date, market="HR"):
    """Add today's priced predictions, replacing same (date, player, market)."""
    if not rows:
        return 0

    date_str = str(game_date)
    existing = []
    if LEDGER.exists():
        with open(LEDGER) as f:
            existing = list(csv.DictReader(f))

    new_rows = [{
        "date": date_str,
        "player_name": r.get("player_name", ""),
        "team": r.get("team", ""),
        "market": market,
        "model_prob": r.get("model_p_hr", ""),
        "implied_prob": r.get("implied_prob", ""),
        "edge": r.get("edge", ""),
        "price_taken": r.get("best_price", ""),
        "bookmaker": r.get("bookmaker", ""),
        "price_close": "",
        "actual": "",
        "graded": "False",
    } for r in rows]

    replacing = {_key(r) for r in new_rows}
    kept = [r for r in existing if _key(r) not in replacing]
    prior = {_key(r): r for r in existing if _key(r) in replacing}

    for r in new_rows:
        p = prior.get(_key(r))
        if p:
            r["price_close"] = p.get("price_close", "")
            r["actual"] = p.get("actual", "")
            r["graded"] = p.get("graded", "False")

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept + new_rows)

    print(f"  Logged {len(new_rows)} priced predictions to {LEDGER.name} "
          f"({len(kept) + len(new_rows)} total)")
    return len(new_rows)
