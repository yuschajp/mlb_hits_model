"""
grade_yesterday_hr.py

Home run version of grade_yesterday.py. Run after yesterday's games are
over to fetch actual HR results and update the HR ledger, then print
rolling calibration stats.

Run with: python3 scripts/grade_yesterday_hr.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import mlb_api_client as client
from src.calibration import brier_score, calibration_table
from src.ledger import hr_columns, load_ledger, ungraded_rows, update_outcomes

HR_LEDGER_PATH = Path(__file__).resolve().parents[1] / "data" / "ledger" / "hr_predictions_log.csv"


def main():
    today = date.today()
    pending = ungraded_rows(HR_LEDGER_PATH, before_date=today, columns=hr_columns())

    if pending.empty:
        print("No ungraded HR predictions from prior days -- nothing to do.")
    else:
        game_pks = pending["game_pk"].unique()
        print(f"Grading {len(pending)} HR predictions across {len(game_pks)} completed game(s)...")

        for game_pk in game_pks:
            results = client.get_game_hr_results(int(game_pk))
            actual_hrs = {pid: (1 if hrs >= 1 else 0) for pid, hrs in results.items()}
            update_outcomes(HR_LEDGER_PATH, actual_hrs, game_pk=int(game_pk),
                             outcome_col="actual_hr", columns=hr_columns())

        print("HR ledger updated.\n")

    df = load_ledger(HR_LEDGER_PATH, columns=hr_columns())
    graded = df[df["graded"] == True]  # noqa: E712
    if graded.empty:
        print("No graded HR predictions yet to evaluate.")
        return

    print(f"Rolling stats over {len(graded)} graded HR predictions "
          f"(since {graded['date'].min().date()}):")
    score = brier_score(graded, prob_col="p_hr", outcome_col="actual_hr")
    print(f"  Brier score: {score:.3f} (0 = perfect, 0.25 = always guessing 50%)")
    overall_rate = graded["actual_hr"].mean()
    avg_predicted = graded["p_hr"].mean()
    print(f"  Average predicted P(HR): {avg_predicted:.1%}   Actual HR rate: {overall_rate:.1%}")

    print("\nCalibration by predicted-probability bucket:")
    table = calibration_table(graded, prob_col="p_hr", outcome_col="actual_hr")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
