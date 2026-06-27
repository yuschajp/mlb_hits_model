"""
grade_yesterday_k.py

Grades yesterday's strikeout predictions against actual results.

Run with: python3 scripts/grade_yesterday_k.py
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import mlb_api_client as client
from src.calibration import brier_score, calibration_table

K_LEDGER_PATH = Path(__file__).resolve().parents[1] / "data" / "ledger" / "k_predictions_log.csv"
NAIVE_BRIER   = 0.2500  # always predicting 50/50


def load_k_ledger(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


def main():
    today = date.today()
    df = load_k_ledger(K_LEDGER_PATH)
    if df.empty:
        print("No K predictions found. Run scripts/run_daily_k.py first.")
        return

    pending = df[(df["graded"] != True) & (df["date"].dt.date < today)]  # noqa: E712

    if pending.empty:
        print("No ungraded K predictions from prior days.")
    else:
        game_pks = pending["game_pk"].unique()
        print(f"Grading {len(pending)} K predictions across {len(game_pks)} game(s)...")

        for game_pk in game_pks:
            results = client.get_game_pitcher_k_results(int(game_pk))
            if not results:
                print(f"  [skip] game {game_pk}: no pitcher results yet")
                continue

            mask = df["game_pk"] == game_pk
            for pitcher_id, actual_ks in results.items():
                p_mask = mask & (df["pitcher_id"] == pitcher_id)
                if not p_mask.any():
                    continue
                line = float(df.loc[p_mask, "line"].iloc[0])
                actual_over = 1 if actual_ks > line else 0
                df.loc[p_mask, "actual_ks"]   = actual_ks
                df.loc[p_mask, "graded"]       = True

                pitcher_name = df.loc[p_mask, "pitcher_name"].iloc[0]
                print(f"  {pitcher_name:<25} actual={actual_ks}K  "
                      f"line={line}  result={'OVER' if actual_over else 'UNDER'}")

        df.to_csv(K_LEDGER_PATH, index=False)
        print("K ledger updated.\n")

    # Rolling calibration stats
    graded = df[df["graded"] == True].copy()  # noqa: E712
    graded = graded.dropna(subset=["actual_ks"])
    if graded.empty:
        print("No graded K predictions yet.")
        return

    graded["actual_over"] = (graded["actual_ks"] > graded["line"]).astype(int)

    print(f"Rolling stats over {len(graded)} graded K predictions "
          f"(since {graded['date'].min().date()}):")
    score = brier_score(graded, prob_col="p_over", outcome_col="actual_over")
    print(f"  Brier score: {score:.4f}  (naive baseline: {NAIVE_BRIER:.4f})")

    over_rate    = graded["actual_over"].mean()
    avg_pred     = graded["p_over"].mean()
    avg_actual_k = graded["actual_ks"].mean()
    avg_lambda   = graded["lambda_k"].mean()

    print(f"  Avg predicted P(over): {avg_pred:.1%}   Actual over rate: {over_rate:.1%}")
    print(f"  Avg expected Ks (λ): {avg_lambda:.1f}   Avg actual Ks: {avg_actual_k:.1f}")

    print("\nCalibration by predicted-probability bucket:")
    cal = calibration_table(graded, prob_col="p_over", outcome_col="actual_over")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # Best and worst predictions
    graded["error"] = abs(graded["lambda_k"] - graded["actual_ks"])
    print(f"\nMost accurate predictions (λ closest to actual):")
    best = graded.nsmallest(5, "error")[["pitcher_name", "lambda_k", "actual_ks", "p_over", "line"]]
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
