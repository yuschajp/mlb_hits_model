"""
publish_dashboard.py

Reads the local hit and HR ledger CSVs and writes docs/dashboard_data.json,
which the GitHub Pages dashboard (docs/index.html) reads at load time.

Run this after find_value.py each day, then run push_dashboard.sh to
commit and publish the updated data to GitHub Pages.

No API keys or network access needed -- this only reads local files.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.calibration import brier_score, calibration_table
from src.ledger import hr_columns, load_ledger

ROOT = Path(__file__).resolve().parents[1]
HITS_LEDGER = ROOT / "data" / "ledger" / "predictions_log.csv"
HR_LEDGER = ROOT / "data" / "ledger" / "hr_predictions_log.csv"
OUT = ROOT / "docs" / "dashboard_data.json"

NAIVE_BRIER = 0.235  # always-predict-league-average baseline


def summarize_hits(ledger_path):
    df = load_ledger(ledger_path)
    if df.empty:
        return {}

    today = date.today()
    today_rows = df[df["date"].dt.date == today]
    graded = df[df["graded"] == True]  # noqa: E712

    top10 = (
        today_rows.sort_values("p_hit", ascending=False)
        .head(10)[["player_name", "team", "opponent_pitcher", "venue", "lineup_spot", "p_hit"]]
        .to_dict(orient="records")
    )

    score = brier_score(graded) if not graded.empty else None
    cal = calibration_table(graded) if not graded.empty else None
    cal_rows = cal.to_dict(orient="records") if cal is not None else []

    # Rolling 7-day accuracy
    week_ago = today - timedelta(days=7)
    recent = graded[graded["date"].dt.date >= week_ago]
    recent_score = brier_score(recent) if not recent.empty else None

    return {
        "today_count": len(today_rows),
        "total_graded": len(graded),
        "since_date": str(graded["date"].min().date()) if not graded.empty else None,
        "brier_score": round(score, 4) if score else None,
        "recent_brier_score": round(recent_score, 4) if recent_score else None,
        "naive_brier": NAIVE_BRIER,
        "avg_predicted": round(float(graded["p_hit"].mean()), 4) if not graded.empty else None,
        "actual_hit_rate": round(float(graded["actual_hit"].mean()), 4) if not graded.empty else None,
        "top10_today": [
            {
                "player_name": r["player_name"],
                "team": r["team"],
                "opponent_pitcher": r["opponent_pitcher"],
                "venue": r["venue"],
                "lineup_spot": int(r["lineup_spot"]),
                "p_hit": round(float(r["p_hit"]), 4),
            }
            for r in top10
        ],
        "calibration": [
            {
                "range": r["PredRange"],
                "n": int(r["N"]),
                "predicted": round(float(r["AvgPredicted"]), 3),
                "actual": round(float(r["ActualFrequency"]), 3),
            }
            for r in cal_rows
        ],
    }


def summarize_hr(ledger_path):
    cols = hr_columns()
    df = load_ledger(ledger_path, columns=cols)
    if df.empty:
        return {}

    today = date.today()
    today_rows = df[df["date"].dt.date == today]
    graded = df[df["graded"] == True]  # noqa: E712

    top10 = (
        today_rows.sort_values("p_hr", ascending=False)
        .head(10)[["player_name", "team", "opponent_pitcher", "venue", "lineup_spot", "p_hr"]]
        .to_dict(orient="records")
    )

    score = brier_score(graded, prob_col="p_hr", outcome_col="actual_hr") if not graded.empty else None
    cal = calibration_table(graded, prob_col="p_hr", outcome_col="actual_hr") if not graded.empty else None
    cal_rows = cal.to_dict(orient="records") if cal is not None else []

    return {
        "today_count": len(today_rows),
        "total_graded": len(graded),
        "since_date": str(graded["date"].min().date()) if not graded.empty else None,
        "brier_score": round(score, 4) if score else None,
        "naive_brier": NAIVE_BRIER,
        "avg_predicted": round(float(graded["p_hr"].mean()), 4) if not graded.empty else None,
        "actual_hr_rate": round(float(graded["actual_hr"].mean()), 4) if not graded.empty else None,
        "top10_today": [
            {
                "player_name": r["player_name"],
                "team": r["team"],
                "opponent_pitcher": r["opponent_pitcher"],
                "venue": r["venue"],
                "lineup_spot": int(r["lineup_spot"]),
                "p_hr": round(float(r["p_hr"]), 4),
            }
            for r in top10
        ],
        "calibration": [
            {
                "range": r["PredRange"],
                "n": int(r["N"]),
                "predicted": round(float(r["AvgPredicted"]), 3),
                "actual": round(float(r["ActualFrequency"]), 3),
            }
            for r in cal_rows
        ],
    }


def main():
    payload = {
        "generated_at": date.today().isoformat(),
        "hits": summarize_hits(HITS_LEDGER),
        "hr": summarize_hr(HR_LEDGER),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Dashboard data written to {OUT}")
    print(f"  Hits: {payload['hits'].get('today_count', 0)} predictions today, "
          f"{payload['hits'].get('total_graded', 0)} graded total")
    print(f"  HR:   {payload['hr'].get('today_count', 0)} predictions today, "
          f"{payload['hr'].get('total_graded', 0)} graded total")
    print(f"\nNext: run push_dashboard.sh to commit and publish to GitHub Pages.")


if __name__ == "__main__":
    main()
