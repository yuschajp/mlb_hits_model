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
from itertools import combinations
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.calibration import brier_score, calibration_table
from src.ledger import hr_columns, load_ledger

ROOT = Path(__file__).resolve().parents[1]
HITS_LEDGER = ROOT / "data" / "ledger" / "predictions_log.csv"
HR_LEDGER = ROOT / "data" / "ledger" / "hr_predictions_log.csv"
VALUE_PICKS_HITS = ROOT / "data" / "value_picks_hits.json"
VALUE_PICKS_HR = ROOT / "data" / "value_picks_hr.json"
OUT = ROOT / "docs" / "dashboard_data.json"

NAIVE_BRIER = 0.235  # always-predict-league-average baseline


# ── Parlay helpers ────────────────────────────────────────────────────────────

def _to_decimal(american):
    return american / 100 + 1 if american > 0 else 100 / (-american) + 1

def _to_american(decimal):
    if decimal >= 2.0:
        return int(round((decimal - 1) * 100))
    return int(round(-100 / (decimal - 1)))

def _load_value_picks(path, bet_type):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    today = date.today().isoformat()
    if data.get("date") != today:
        return []
    for p in data["picks"]:
        p["type"] = bet_type
    return data["picks"]

def _build_picks_payload(today):
    hits = _load_value_picks(VALUE_PICKS_HITS, "hit")
    hr = _load_value_picks(VALUE_PICKS_HR, "hr")
    all_picks = hits + hr

    # Singles: all value picks sorted by edge descending
    singles = sorted(all_picks, key=lambda p: p["edge"], reverse=True)

    # Parlays: top combos of 2 and 3 legs from the best 6 picks
    pool = singles[:6]
    parlay_rows = []
    for n in (2, 3):
        for combo in combinations(pool, n):
            model_prob = 1.0
            decimal_odds = 1.0
            for pick in combo:
                prob = pick.get("model_p_hit") or pick.get("model_p_hr")
                model_prob *= prob
                decimal_odds *= _to_decimal(pick["best_price"])
            implied_prob = 1.0 / decimal_odds
            parlay_rows.append({
                "legs": [
                    {
                        "player_name": p["player_name"],
                        "team": p["team"],
                        "type": p["type"],
                        "best_price": p["best_price"],
                        "bookmaker": p["bookmaker"],
                    }
                    for p in combo
                ],
                "n_legs": n,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(model_prob - implied_prob, 4),
                "payout": _to_american(decimal_odds),
            })

    parlay_rows.sort(key=lambda r: r["edge"], reverse=True)

    # Round robins: top 3-pick pool → all C(3,2) 2-leg parlays
    rr_pools = []
    rr_pool = singles[:4]
    for size in (3, 4):
        if len(rr_pool) >= size:
            pool_picks = rr_pool[:size]
            legs_out = []
            for combo in combinations(pool_picks, 2):
                model_prob = 1.0
                decimal_odds = 1.0
                for pick in combo:
                    prob = pick.get("model_p_hit") or pick.get("model_p_hr")
                    model_prob *= prob
                    decimal_odds *= _to_decimal(pick["best_price"])
                legs_out.append({
                    "players": [p["player_name"] for p in combo],
                    "payout": _to_american(decimal_odds),
                    "model_prob": round(model_prob, 4),
                })
            rr_pools.append({
                "size": size,
                "picks": [p["player_name"] for p in pool_picks],
                "parlays": legs_out,
            })

    return {
        "singles": singles,
        "parlays": parlay_rows[:8],
        "round_robins": rr_pools,
    }


def compute_returns(df, prob_col, outcome_col, threshold, american_odds):
    """Daily + cumulative P&L assuming 1-unit flat bets on picks above threshold."""
    graded = df[(df["graded"] == True) & (df[prob_col] >= threshold)].copy().sort_values("date")
    if graded.empty:
        return []
    win_payout = american_odds / 100 if american_odds > 0 else 100 / (-american_odds)
    result = []
    cumulative = 0.0
    for day, group in graded.groupby(graded["date"].dt.date):
        wins = float(group[outcome_col].sum())
        losses = len(group) - wins
        daily_pnl = wins * win_payout - losses
        cumulative += daily_pnl
        result.append({
            "date": str(day),
            "n_bets": len(group),
            "n_wins": int(wins),
            "daily_pnl": round(daily_pnl, 3),
            "cumulative_pnl": round(cumulative, 3),
        })
    return result


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
        "returns": compute_returns(df, "p_hit", "actual_hit", threshold=0.60, american_odds=-110),
        "returns_label": "p_hit ≥ 60% · flat -110",
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
        "returns": compute_returns(df, "p_hr", "actual_hr", threshold=0.15, american_odds=350),
        "returns_label": "p_hr ≥ 15% · flat +350",
    }


def main():
    today = date.today()
    payload = {
        "generated_at": today.isoformat(),
        "hits": summarize_hits(HITS_LEDGER),
        "hr": summarize_hr(HR_LEDGER),
        "picks": _build_picks_payload(today),
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
