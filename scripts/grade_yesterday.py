"""
grade_yesterday.py

Run this after yesterday's games are over (e.g. the next morning, before
run_daily.py's next run) to fetch actual results for any ungraded
predictions and update the ledger, then print rolling calibration stats.

Run with: python3 scripts/grade_yesterday.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import mlb_api_client as client
from src.calibration import brier_score, calibration_table
from src.ledger import load_ledger, ungraded_rows, update_outcomes

LEDGER_PATH = Path(__file__).resolve().parents[1] / "data" / "ledger" / "predictions_log.csv"


def main():
    today = date.today()
    pending = ungraded_rows(LEDGER_PATH, before_date=today)

    if pending.empty:
        print("No ungraded predictions from prior days -- nothing to do.")
    else:
        game_pks = pending["game_pk"].unique()
        print(f"Grading {len(pending)} predictions across {len(game_pks)} completed game(s)...")

        for game_pk in game_pks:
            results = client.get_game_batting_results(int(game_pk))
            actual_hits = {pid: (1 if hits >= 1 else 0) for pid, hits in results.items()}
            update_outcomes(LEDGER_PATH, actual_hits, game_pk=int(game_pk))

        print("Ledger updated.\n")

    df = load_ledger(LEDGER_PATH)
    graded = df[df["graded"] == True]  # noqa: E712
    if graded.empty:
        print("No graded predictions yet to evaluate.")
        return

    print(f"Rolling stats over {len(graded)} graded predictions "
          f"(since {graded['date'].min().date()}):")
    score = brier_score(graded)
    print(f"  Brier score: {score:.3f} (0 = perfect, 0.25 = always guessing 50%)")
    overall_hit_rate = graded["actual_hit"].mean()
    avg_predicted = graded["p_hit"].mean()
    print(f"  Average predicted P(hit): {avg_predicted:.1%}   Actual hit rate: {overall_hit_rate:.1%}")

    print("\nCalibration by predicted-probability bucket:")
    table = calibration_table(graded)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
