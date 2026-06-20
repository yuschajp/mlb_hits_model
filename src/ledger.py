"""
ledger.py

Append-only CSV ledger for daily predictions. Each day's run appends new
rows (one per batter/game); a separate grading step fills in actual_hit
once the game is over. Plain CSV rather than a database, consistent with
how the rest of this portfolio's projects are built -- no extra
infrastructure needed to run this daily on a laptop or a small cron box.
"""

from pathlib import Path

import pandas as pd

COLUMNS = [
    "date", "game_pk", "player_id", "player_name", "team", "opponent",
    "lineup_spot", "venue", "park_factor", "opponent_pitcher",
    "p_hit", "adjusted_ba", "expected_ab", "actual_hit", "graded",
]


def append_predictions(rows, ledger_path):
    """rows: list of dicts matching COLUMNS (actual_hit/graded default to blank/False)."""
    ledger_path = Path(ledger_path)
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]
    df["graded"] = df["graded"].fillna(False)

    if ledger_path.exists():
        df.to_csv(ledger_path, mode="a", header=False, index=False)
    else:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ledger_path, mode="w", header=True, index=False)


def load_ledger(ledger_path):
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(ledger_path, parse_dates=["date"])


def ungraded_rows(ledger_path, before_date=None):
    """Rows with no recorded outcome yet, optionally only for games before before_date."""
    df = load_ledger(ledger_path)
    if df.empty:
        return df
    mask = df["graded"] != True  # noqa: E712 (explicit comparison reads clearer here)
    if before_date is not None:
        mask &= df["date"].dt.date < before_date
    return df[mask]


def update_outcomes(ledger_path, player_id_to_actual_hit, game_pk):
    """
    Fill in actual_hit for a specific game's rows once results are known.
    player_id_to_actual_hit: {player_id: 0 or 1}.
    """
    df = load_ledger(ledger_path)
    if df.empty:
        return df
    mask = (df["game_pk"] == game_pk) & (df["player_id"].isin(player_id_to_actual_hit.keys()))
    df.loc[mask, "actual_hit"] = df.loc[mask, "player_id"].map(player_id_to_actual_hit)
    df.loc[mask, "graded"] = True
    df.to_csv(ledger_path, index=False)
    return df
