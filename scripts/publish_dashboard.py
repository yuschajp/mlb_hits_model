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

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.calibration import brier_score, calibration_table
from src.ledger import hr_columns, load_ledger

ROOT             = Path(__file__).resolve().parents[1]
HITS_LEDGER      = ROOT / "data" / "ledger" / "predictions_log.csv"
HR_LEDGER        = ROOT / "data" / "ledger" / "hr_predictions_log.csv"
K_LEDGER         = ROOT / "data" / "ledger" / "k_predictions_log.csv"
WC_LEDGER        = ROOT / "data" / "ledger" / "wc_predictions_log.csv"
WC_GS_LEDGER     = ROOT / "data" / "ledger" / "wc_gs_predictions_log.csv"
TENNIS_LEDGER    = ROOT / "data" / "ledger" / "tennis_predictions_log.csv"
VALUE_PICKS_HITS = ROOT / "data" / "value_picks_hits.json"
VALUE_PICKS_HR   = ROOT / "data" / "value_picks_hr.json"
OUT              = ROOT / "docs" / "dashboard_data.json"

# F1 model lives as a sibling repo on the same Desktop
F1_QUALI_LEDGER  = ROOT.parent / "f1_model" / "data" / "ledger" / "quali_predictions_log.csv"
F1_RACE_LEDGER   = ROOT.parent / "f1_model" / "data" / "ledger" / "predictions_log.csv"

NAIVE_BRIER = 0.235


# ── P&L simulation ────────────────────────────────────────────────────────────

def compute_returns(df, prob_col, outcome_col, threshold, american_odds):
    """
    Simulates flat-betting every graded prediction above threshold at the
    given american_odds price. Returns a list of daily cumulative P&L rows.
    """
    graded = df[(df["graded"] == True) & (df[prob_col] >= threshold)].copy()  # noqa: E712
    graded = graded.dropna(subset=[outcome_col])
    if graded.empty:
        return []
    graded["date"] = pd.to_datetime(graded["date"])
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
    hr   = _load_value_picks(VALUE_PICKS_HR, "hr")
    all_picks = hits + hr

    singles = sorted(all_picks, key=lambda p: p["edge"], reverse=True)

    pool = singles[:6]
    parlay_rows = []
    for n in (2, 3):
        for combo in combinations(pool, n):
            model_prob   = 1.0
            decimal_odds = 1.0
            for pick in combo:
                prob = pick.get("model_p_hit") or pick.get("model_p_hr")
                model_prob   *= prob
                decimal_odds *= _to_decimal(pick["best_price"])
            implied_prob = 1.0 / decimal_odds
            edge = model_prob - implied_prob
            parlay_rows.append({
                "legs": [
                    {
                        "player_name": p["player_name"],
                        "team":        p["team"],
                        "type":        p["type"],
                        "price":       p["best_price"],
                        "bookmaker":   p["bookmaker"],
                    }
                    for p in combo
                ],
                "n_legs":       n,
                "model_prob":   round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge":         round(edge, 4),
                "payout":       _to_american(decimal_odds),
            })

    parlay_rows.sort(key=lambda r: r["edge"], reverse=True)

    # Round robins: 3-pick and 4-pick pools
    rr_pools = []
    for size in (3, 4):
        pool_rr = singles[:size]
        if len(pool_rr) < size:
            continue
        combos = []
        for leg_a, leg_b in combinations(pool_rr, 2):
            dec = _to_decimal(leg_a["best_price"]) * _to_decimal(leg_b["best_price"])
            prob_a = leg_a.get("model_p_hit") or leg_a.get("model_p_hr")
            prob_b = leg_b.get("model_p_hit") or leg_b.get("model_p_hr")
            combos.append({
                "players": [leg_a["player_name"], leg_b["player_name"]],
                "payout":  _to_american(dec),
                "model_prob": round(prob_a * prob_b, 4),
            })
        rr_pools.append({
            "size":    size,
            "picks":   [p["player_name"] for p in pool_rr],
            "parlays": combos,
        })

    return {
        "singles":      singles,
        "parlays":      parlay_rows[:8],
        "round_robins": rr_pools,
    }


# ── Summarizers ───────────────────────────────────────────────────────────────

def summarize_hits(ledger_path):
    df = load_ledger(ledger_path)
    if df.empty:
        return {}

    today     = date.today()
    today_rows = df[df["date"].dt.date == today]
    graded    = df[df["graded"] == True]  # noqa: E712

    top10 = (
        today_rows.sort_values("p_hit", ascending=False)
        .head(10)[["player_name", "team", "opponent_pitcher", "venue", "lineup_spot", "p_hit"]]
        .to_dict(orient="records")
    )

    score    = brier_score(graded) if not graded.empty else None
    cal      = calibration_table(graded) if not graded.empty else None
    cal_rows = cal.to_dict(orient="records") if cal is not None else []

    week_ago     = today - timedelta(days=7)
    recent       = graded[graded["date"].dt.date >= week_ago]
    recent_score = brier_score(recent) if not recent.empty else None

    return {
        "today_count":    len(today_rows),
        "total_graded":   len(graded),
        "since_date":     str(graded["date"].min().date()) if not graded.empty else None,
        "brier_score":    round(score, 4) if score else None,
        "recent_brier_score": round(recent_score, 4) if recent_score else None,
        "naive_brier":    NAIVE_BRIER,
        "avg_predicted":  round(float(graded["p_hit"].mean()), 4) if not graded.empty else None,
        "actual_hit_rate": round(float(graded["actual_hit"].mean()), 4) if not graded.empty else None,
        "top10_today": [
            {
                "player_name":     r["player_name"],
                "team":            r["team"],
                "opponent_pitcher": r["opponent_pitcher"],
                "venue":           r["venue"],
                "lineup_spot":     int(r["lineup_spot"]),
                "p_hit":           round(float(r["p_hit"]), 4),
            }
            for r in top10
        ],
        "calibration": [
            {
                "range":     r["PredRange"],
                "n":         int(r["N"]),
                "predicted": round(float(r["AvgPredicted"]), 3),
                "actual":    round(float(r["ActualFrequency"]), 3),
            }
            for r in cal_rows
        ],
        "returns":       compute_returns(df, "p_hit", "actual_hit", threshold=0.60, american_odds=-110),
        "returns_label": "p_hit ≥ 60% · flat -110",
    }


def summarize_hr(ledger_path):
    cols = hr_columns()
    df   = load_ledger(ledger_path, columns=cols)
    if df.empty:
        return {}

    today      = date.today()
    today_rows = df[df["date"].dt.date == today]
    graded     = df[df["graded"] == True]  # noqa: E712

    top10 = (
        today_rows.sort_values("p_hr", ascending=False)
        .head(10)[["player_name", "team", "opponent_pitcher", "venue", "lineup_spot", "p_hr"]]
        .to_dict(orient="records")
    )

    score    = brier_score(graded, prob_col="p_hr", outcome_col="actual_hr") if not graded.empty else None
    cal      = calibration_table(graded, prob_col="p_hr", outcome_col="actual_hr") if not graded.empty else None
    cal_rows = cal.to_dict(orient="records") if cal is not None else []

    return {
        "today_count":   len(today_rows),
        "total_graded":  len(graded),
        "since_date":    str(graded["date"].min().date()) if not graded.empty else None,
        "brier_score":   round(score, 4) if score else None,
        "naive_brier":   NAIVE_BRIER,
        "avg_predicted": round(float(graded["p_hr"].mean()), 4) if not graded.empty else None,
        "actual_hr_rate": round(float(graded["actual_hr"].mean()), 4) if not graded.empty else None,
        "top10_today": [
            {
                "player_name":     r["player_name"],
                "team":            r["team"],
                "opponent_pitcher": r["opponent_pitcher"],
                "venue":           r["venue"],
                "lineup_spot":     int(r["lineup_spot"]),
                "p_hr":            round(float(r["p_hr"]), 4),
            }
            for r in top10
        ],
        "calibration": [
            {
                "range":     r["PredRange"],
                "n":         int(r["N"]),
                "predicted": round(float(r["AvgPredicted"]), 3),
                "actual":    round(float(r["ActualFrequency"]), 3),
            }
            for r in cal_rows
        ],
        "returns":       compute_returns(df, "p_hr", "actual_hr", threshold=0.15, american_odds=350),
        "returns_label": "p_hr ≥ 15% · flat +350",
    }


def summarize_k():
    """Reads the K prediction ledger and returns summary for the dashboard."""
    if not K_LEDGER.exists():
        return {}

    try:
        df = pd.read_csv(K_LEDGER, parse_dates=["date"])
        if df.empty:
            return {}

        today = date.today()
        today_rows = df[df["date"].dt.date == today]
        graded = df[df["graded"] == True].copy()  # noqa: E712
        graded = graded.dropna(subset=["actual_ks"])

        # Add actual_over column for calibration
        if not graded.empty:
            graded["actual_over"] = (graded["actual_ks"] > graded["line"]).astype(int)

        top_today = (
            today_rows.sort_values("p_over", ascending=False)
            .head(10)[["pitcher_name", "team", "opponent", "venue", "line", "lambda_k", "p_over", "p_under"]]
            .to_dict(orient="records")
        )

        score = brier_score(graded, prob_col="p_over", outcome_col="actual_over") if not graded.empty else None
        cal   = calibration_table(graded, prob_col="p_over", outcome_col="actual_over") if not graded.empty else None
        cal_rows = cal.to_dict(orient="records") if cal is not None else []

        return {
            "today_count":  len(today_rows),
            "total_graded": len(graded),
            "since_date":   str(graded["date"].min().date()) if not graded.empty else None,
            "brier_score":  round(score, 4) if score else None,
            "naive_brier":  0.25,
            "avg_lambda":   round(float(today_rows["lambda_k"].mean()), 2) if not today_rows.empty else None,
            "over_rate":    round(float(graded["actual_over"].mean()), 3) if not graded.empty else None,
            "top10_today": [
                {
                    "pitcher_name": r["pitcher_name"],
                    "team":         r["team"],
                    "opponent":     r["opponent"],
                    "venue":        r["venue"],
                    "line":         float(r["line"]),
                    "lambda_k":     round(float(r["lambda_k"]), 2),
                    "p_over":       round(float(r["p_over"]), 4),
                    "p_under":      round(float(r["p_under"]), 4),
                }
                for r in top_today
            ],
            "calibration": [
                {
                    "range":     r["PredRange"],
                    "n":         int(r["N"]),
                    "predicted": round(float(r["AvgPredicted"]), 3),
                    "actual":    round(float(r["ActualFrequency"]), 3),
                }
                for r in cal_rows
            ],
        }
    except Exception as e:
        print(f"  [warn] Could not read K ledger: {e}")
        return {}


def summarize_wc():
    """Reads the WC prediction ledger and returns summary for the dashboard."""
    if not WC_LEDGER.exists():
        return {}
    try:
        df = pd.read_csv(WC_LEDGER, parse_dates=["date"])
        if df.empty:
            return {}

        today      = date.today()
        today_rows = df[df["date"].dt.date == today]
        graded     = df[df["graded"] == True].copy()  # noqa: E712
        graded     = graded.dropna(subset=["actual_result"])

        if not graded.empty:
            graded["actual_home"] = (graded["actual_result"] == "home").astype(int)
            graded["actual_draw"] = (graded["actual_result"] == "draw").astype(int)
            graded["actual_away"] = (graded["actual_result"] == "away").astype(int)

        matches_today = [
            {
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "stage":     r.get("stage", ""),
                "xg_home":   round(float(r["xg_home"]), 2),
                "xg_away":   round(float(r["xg_away"]), 2),
                "p_home":    round(float(r["p_home"]), 4),
                "p_draw":    round(float(r["p_draw"]), 4),
                "p_away":    round(float(r["p_away"]), 4),
                "over_2_5":  round(float(r["over_2_5"]), 4),
                "btts":      round(float(r["btts"]), 4),
                "graded":    bool(r.get("graded", False)),
                "result":    r.get("actual_result", None),
            }
            for _, r in today_rows.iterrows()
        ]

        score_h = brier_score(graded, prob_col="p_home", outcome_col="actual_home") if not graded.empty else None

        return {
            "today_count":  len(today_rows),
            "total_graded": len(graded),
            "brier_home":   round(score_h, 4) if score_h else None,
            "naive_brier":  0.222,
            "matches_today": matches_today,
        }
    except Exception as e:
        print(f"  [warn] Could not read WC ledger: {e}")
        return {}


def summarize_wc_gs():
    """
    Reads the WC anytime-goalscorer prediction ledger and returns a
    summary for the dashboard. Mirrors summarize_wc() in structure, but
    was previously missing entirely -- goalscorer predictions were being
    generated and logged by run_daily_wc_gs.py, but nothing ever read
    wc_gs_predictions_log.csv into the dashboard payload, so they never
    reached docs/dashboard_data.json regardless of how often the pipeline
    ran.
    """
    if not WC_GS_LEDGER.exists():
        return {}
    try:
        df = pd.read_csv(WC_GS_LEDGER, parse_dates=["date"])
        if df.empty:
            return {}

        today      = date.today()
        today_rows = df[df["date"].dt.date == today]
        graded     = df[df["graded"] == True].copy()  # noqa: E712
        graded     = graded.dropna(subset=["actual_scored"])

        top_today = (
            today_rows.sort_values("p_scores", ascending=False)
            .head(15)[["player", "team", "opponent", "home_team", "away_team",
                       "tournament_goals", "lambda", "p_scores"]]
            .to_dict(orient="records")
        )

        score = brier_score(graded, prob_col="p_scores", outcome_col="actual_scored") if not graded.empty else None
        cal   = calibration_table(graded, prob_col="p_scores", outcome_col="actual_scored") if not graded.empty else None
        cal_rows = cal.to_dict(orient="records") if cal is not None else []

        return {
            "today_count":  len(today_rows),
            "total_graded": len(graded),
            "since_date":   str(graded["date"].min().date()) if not graded.empty else None,
            "brier_score":  round(score, 4) if score else None,
            "naive_brier":  0.20,
            "top15_today": [
                {
                    "player":            r["player"],
                    "team":              r["team"],
                    "opponent":          r["opponent"],
                    "match":             f"{r['home_team']} vs {r['away_team']}",
                    "tournament_goals":  int(r["tournament_goals"]),
                    "lambda":            round(float(r["lambda"]), 3),
                    "p_scores":          round(float(r["p_scores"]), 4),
                }
                for r in top_today
            ],
            "calibration": [
                {
                    "range":     r["PredRange"],
                    "n":         int(r["N"]),
                    "predicted": round(float(r["AvgPredicted"]), 3),
                    "actual":    round(float(r["ActualFrequency"]), 3),
                }
                for r in cal_rows
            ],
        }
    except Exception as e:
        print(f"  [warn] Could not read WC goalscorer ledger: {e}")
        return {}


def summarize_tennis():
    """Reads the tennis prediction ledger and returns dashboard summary."""
    if not TENNIS_LEDGER.exists():
        return {}
    try:
        df = pd.read_csv(TENNIS_LEDGER, parse_dates=["date"])
        if df.empty:
            return {}

        today      = date.today()
        today_rows = df[df["date"].dt.date == today]
        graded     = df[df["graded"] == True].copy()  # noqa: E712

        if not graded.empty:
            graded["correct"] = (graded["p_a_wins"] > 0.5).astype(int)

        matches = [
            {
                "tour":      r["tour"],
                "round":     r["round"],
                "player_a":  r["player_a"],
                "player_b":  r["player_b"],
                "elo_a":     round(float(r["elo_a"]), 1),
                "elo_b":     round(float(r["elo_b"]), 1),
                "p_a_wins":  round(float(r["p_a_wins"]), 4),
                "winner":    r.get("winner", None),
            }
            for _, r in today_rows.iterrows()
        ]

        return {
            "today_count":  len(today_rows),
            "total_graded": len(graded),
            "accuracy":     round(float(graded["correct"].mean()), 3) if not graded.empty else None,
            "matches": matches,
        }
    except Exception as e:
        print(f"  [warn] Could not read tennis ledger: {e}")
        return {}


def summarize_f1():
    """
    Reads qualifying and race prediction ledgers from the sibling f1_model
    repo and returns a summary dict for the dashboard F1 tab.
    """
    result = {"quali": {}, "race": {}, "history": []}

    # ── Qualifying predictions ────────────────────────────────────────────────
    if F1_QUALI_LEDGER.exists():
        try:
            df = pd.read_csv(F1_QUALI_LEDGER)
            if not df.empty:
                # Most recent race weekend
                latest_meeting = df["meeting_key"].max()
                latest = df[df["meeting_key"] == latest_meeting].copy()
                latest = latest.sort_values("predicted_grid")

                graded = latest[latest["graded"] == True]  # noqa: E712

                result["quali"] = {
                    "circuit":     latest.iloc[0]["circuit"] if not latest.empty else None,
                    "meeting_key": int(latest_meeting),
                    "graded":      not graded.empty,
                    "grid": [
                        {
                            "predicted_grid": int(r["predicted_grid"]),
                            "actual_grid":    int(r["actual_grid"]) if pd.notna(r.get("actual_grid")) else None,
                            "driver_id":      r["driver_id"],
                            "driver_name":    r["driver_name"],
                            "team":           r["team"],
                            "p_pole":         round(float(r["p_pole"]), 4),
                            "p_q3":           round(float(r["p_q3"]), 4),
                            "gap_predicted":  round(float(r["predicted_gap"]) - float(latest.iloc[0]["predicted_gap"]), 4),
                        }
                        for _, r in latest.iterrows()
                    ],
                }

                # Qualifying calibration history
                all_graded = df[df["graded"] == True].copy()  # noqa: E712
                if not all_graded.empty:
                    all_graded["actual_pole"] = (all_graded["actual_grid"] == 1).astype(int)
                    all_graded["actual_q3"]   = (all_graded["actual_grid"] <= 10).astype(int)
                    pole_brier = brier_score(all_graded, prob_col="p_pole", outcome_col="actual_pole")
                    q3_brier   = brier_score(all_graded, prob_col="p_q3",   outcome_col="actual_q3")
                    all_graded["pos_error"] = abs(all_graded["predicted_grid"] - all_graded["actual_grid"])
                    result["quali"]["stats"] = {
                        "total_graded":    len(all_graded),
                        "pole_brier":      round(pole_brier, 4) if pole_brier else None,
                        "q3_brier":        round(q3_brier, 4) if q3_brier else None,
                        "avg_pos_error":   round(float(all_graded["pos_error"].mean()), 2),
                        "within_3_spots":  round(float((all_graded["pos_error"] <= 3).mean()), 3),
                    }
        except Exception as e:
            print(f"  [warn] Could not read F1 quali ledger: {e}")

    # ── Race predictions ──────────────────────────────────────────────────────
    if F1_RACE_LEDGER.exists():
        try:
            df = pd.read_csv(F1_RACE_LEDGER)
            if not df.empty:
                latest_round = df["round"].max()
                latest = df[df["round"] == latest_round].sort_values("p_podium", ascending=False)

                result["race"] = {
                    "race_name":   latest.iloc[0]["race_name"] if not latest.empty else None,
                    "race_date":   str(latest.iloc[0]["race_date"])[:10] if not latest.empty else None,
                    "round":       int(latest_round),
                    "top10": [
                        {
                            "driver_name": r["driver_name"],
                            "team":        r["team"],
                            "grid":        int(r["grid_position"]),
                            "p_podium":    round(float(r["p_podium"]), 4),
                            "p_points":    round(float(r["p_points"]), 4),
                        }
                        for _, r in latest.head(10).iterrows()
                    ],
                }

                # Race calibration history
                all_graded = df[df["graded"] == True].dropna(subset=["actual_podium"])  # noqa: E712
                if not all_graded.empty:
                    pod_brier = brier_score(all_graded, prob_col="p_podium", outcome_col="actual_podium")
                    pts_brier = brier_score(all_graded, prob_col="p_points", outcome_col="actual_points")
                    result["race"]["stats"] = {
                        "total_graded":  len(all_graded),
                        "podium_brier":  round(pod_brier, 4) if pod_brier else None,
                        "points_brier":  round(pts_brier, 4) if pts_brier else None,
                    }
        except Exception as e:
            print(f"  [warn] Could not read F1 race ledger: {e}")

    return result


def main():
    today   = date.today()
    f1_data = summarize_f1()
    k_data  = summarize_k()
    wc_data = summarize_wc()
    wc_gs_data = summarize_wc_gs()
    tennis_data = summarize_tennis()
    payload = {
        "generated_at": today.isoformat(),
        "hits":  summarize_hits(HITS_LEDGER),
        "hr":    summarize_hr(HR_LEDGER),
        "k":     k_data,
        "wc":    wc_data,
        "wc_gs": wc_gs_data,
        "tennis": tennis_data,
        "picks": _build_picks_payload(today),
        "f1":    f1_data,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Sanitize NaN/Inf values which are invalid JSON -- replace with null
    clean = json.loads(
        json.dumps(payload, indent=2)
        .replace(': NaN', ': null')
        .replace(': Infinity', ': null')
        .replace(': -Infinity', ': null')
    )
    OUT.write_text(json.dumps(clean, indent=2))
    print(f"Dashboard data written to {OUT}")
    print(f"  Hits: {payload['hits'].get('today_count', 0)} predictions today, "
          f"{payload['hits'].get('total_graded', 0)} graded total")
    print(f"  HR:   {payload['hr'].get('today_count', 0)} predictions today, "
          f"{payload['hr'].get('total_graded', 0)} graded total")
    print(f"  K:    {k_data.get('today_count', 0)} predictions today, "
          f"{k_data.get('total_graded', 0)} graded total")
    print(f"  WC:   {wc_data.get('today_count', 0)} match predictions today, "
          f"{wc_data.get('total_graded', 0)} graded total")
    print(f"  WC-GS: {wc_gs_data.get('today_count', 0)} goalscorer predictions today, "
          f"{wc_gs_data.get('total_graded', 0)} graded total")
    quali_graded = f1_data.get("quali", {}).get("stats", {}).get("total_graded", 0)
    print(f"  F1:   {len(f1_data.get('quali', {}).get('grid', []))} quali predictions, "
          f"{quali_graded} graded total")
    print(f"\nNext: run push_dashboard.sh to commit and publish to GitHub Pages.")


if __name__ == "__main__":
    main()
