"""
ledger.py

Append-only CSV ledger for daily predictions. Each day's run appends new
rows (one per batter/game); a separate grading step fills in the outcome
column once the game is over. Plain CSV rather than a database, consistent
with how the rest of this portfolio's projects are built -- no extra
infrastructure needed to run this daily on a laptop or a small cron box.

Generalized to support multiple prop types (hits, home runs, ...) sharing
this same module: every function takes optional columns/outcome_col
parameters that default to the original hits schema, so existing callers
(run_daily.py, grade_yesterday.py, find_value.py) work completely
unchanged. New prop types pass their own columns list and outcome column
name explicitly -- see hr_columns() below for the home run example.
"""

from pathlib import Path

import pandas as pd

COLUMNS = [
    "date", "game_pk", "player_id", "player_name", "team", "opponent",
    "lineup_spot", "venue", "park_factor", "opponent_pitcher",
    "p_hit", "adjusted_ba", "expected_ab", "actual_hit", "graded",
]


def hr_columns():
    """Column schema for the home run ledger -- pass this as `columns=` for HR-related calls."""
    return [
        "date", "game_pk", "player_id", "player_name", "team", "opponent",
        "lineup_spot", "venue", "park_factor", "opponent_pitcher",
        "p_hr", "adjusted_hr_rate", "expected_ab", "actual_hr", "graded",
    ]


def append_predictions(rows, ledger_path, columns=COLUMNS):
    """
    Upserts rows into the ledger, keyed on (date, game_pk, player_id):
    any existing row matching that key is replaced by the new one, rather
    than duplicated. This matters because the daily scripts are meant to
    be rerunnable during the day as more lineups get confirmed -- without
    this, running it twice in one day would log every already-seen player
    a second time with slightly refreshed stats, double-counting them in
    later calibration analysis.

    Safe to do even for in-progress days: run_daily.py already skips games
    with status "Final" before building predictions, so a row that's
    already been graded (from a genuinely separate, earlier day) should
    never collide with a same-day rerun's keys.
    """
    ledger_path = Path(ledger_path)
    df_new = pd.DataFrame(rows)
    for col in columns:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[columns]
    df_new["graded"] = df_new["graded"].fillna(False)
    df_new["date"] = pd.to_datetime(df_new["date"])

    key_cols = ["date", "game_pk", "player_id"]

    if ledger_path.exists():
        df_existing = pd.read_csv(ledger_path, parse_dates=["date"])
        new_keys = set(map(tuple, df_new[key_cols].astype(str).to_numpy()))
        existing_keys = df_existing[key_cols].astype(str).apply(tuple, axis=1)
        df_existing = df_existing[~existing_keys.isin(new_keys)]
        combined = pd.concat([df_existing, df_new], ignore_index=True)
        combined.to_csv(ledger_path, mode="w", header=True, index=False)
    else:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(ledger_path, mode="w", header=True, index=False)


def load_ledger(ledger_path, columns=COLUMNS):
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(ledger_path, parse_dates=["date"])


def ungraded_rows(ledger_path, before_date=None, columns=COLUMNS):
    """Rows with no recorded outcome yet, optionally only for games before before_date."""
    df = load_ledger(ledger_path, columns=columns)
    if df.empty:
        return df
    mask = df["graded"] != True  # noqa: E712 (explicit comparison reads clearer here)
    if before_date is not None:
        mask &= df["date"].dt.date < before_date
    return df[mask]


def update_outcomes(ledger_path, player_id_to_actual, game_pk, outcome_col="actual_hit", columns=COLUMNS):
    """
    Fill in the outcome column for a specific game's rows once results are
    known. player_id_to_actual: {player_id: 0 or 1}. outcome_col is
    "actual_hit" for the hits ledger (default, unchanged from before) or
    "actual_hr" for the home run ledger.
    """
    df = load_ledger(ledger_path, columns=columns)
    if df.empty:
        return df
    mask = (df["game_pk"] == game_pk) & (df["player_id"].isin(player_id_to_actual.keys()))
    df.loc[mask, outcome_col] = df.loc[mask, "player_id"].map(player_id_to_actual)
    df.loc[mask, "graded"] = True
    df.to_csv(ledger_path, index=False)
    return df
