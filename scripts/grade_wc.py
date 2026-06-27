"""
grade_wc.py

Grades World Cup match predictions against actual results.

Run with: python3 scripts/grade_wc.py
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import wc_data_client as client
from src.calibration import brier_score, calibration_table

WC_LEDGER   = Path(__file__).resolve().parents[1] / "data" / "ledger" / "wc_predictions_log.csv"
NAIVE_BRIER = 0.2222  # always predict 33.3% for each 1X2 outcome


def main():
    if not WC_LEDGER.exists():
        print("No WC predictions found. Run scripts/run_daily_wc.py first.")
        return

    df = pd.read_csv(WC_LEDGER, parse_dates=["date"])
    pending = df[df["graded"] != True]  # noqa: E712

    if pending.empty:
        print("No ungraded WC predictions.")
    else:
        print(f"Grading {len(pending)} predictions...")
        for idx, row in pending.iterrows():
            result = client.get_match_result(int(row["match_id"]))
            if result is None:
                continue

            h = result["home_goals"]
            a = result["away_goals"]
            actual = "home" if h > a else ("draw" if h == a else "away")

            df.loc[idx, "actual_home_goals"] = h
            df.loc[idx, "actual_away_goals"] = a
            df.loc[idx, "actual_result"]     = actual
            df.loc[idx, "graded"]            = True
            print(f"  {row['home_team']} {h}-{a} {row['away_team']} → {actual.upper()}")

        df.to_csv(WC_LEDGER, index=False)
        print("WC ledger updated.\n")

    graded = df[df["graded"] == True].copy()  # noqa: E712
    graded = graded.dropna(subset=["actual_result"])
    if graded.empty:
        print("No graded WC predictions yet.")
        return

    # Build outcome columns for Brier scoring
    graded["actual_home"] = (graded["actual_result"] == "home").astype(int)
    graded["actual_draw"] = (graded["actual_result"] == "draw").astype(int)
    graded["actual_away"] = (graded["actual_result"] == "away").astype(int)
    graded["actual_over"] = (
        graded["actual_home_goals"] + graded["actual_away_goals"] > 2.5
    ).astype(int)

    print(f"Rolling stats over {len(graded)} graded WC matches:")
    score_h = brier_score(graded, prob_col="p_home", outcome_col="actual_home")
    score_d = brier_score(graded, prob_col="p_draw", outcome_col="actual_draw")
    score_a = brier_score(graded, prob_col="p_away", outcome_col="actual_away")

    print(f"  Home Brier: {score_h:.4f}  Draw Brier: {score_d:.4f}  "
          f"Away Brier: {score_a:.4f}  (naive: {NAIVE_BRIER:.4f})")

    # Result distribution
    result_counts = graded["actual_result"].value_counts()
    print(f"\n  Results: Home {result_counts.get('home',0)}  "
          f"Draw {result_counts.get('draw',0)}  Away {result_counts.get('away',0)}")

    print("\nHome win calibration:")
    print(calibration_table(graded, prob_col="p_home",
                             outcome_col="actual_home").to_string(index=False))


if __name__ == "__main__":
    main()
