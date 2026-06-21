"""
park_factors.py

Loads the park factors lookup table. These values are illustrative
starting points based on each park's well-known general characteristics
(dimensions, altitude, marine air, roof/no roof) -- NOT pulled from a live,
current-season authoritative source. Refresh annually from FanGraphs'
published park factors guide (https://www.fangraphs.com/guts.aspx?type=pf)
before relying on this for anything real; park factors shift over time
(see Camden Yards' 2022 fence changes for an example of how much a single
off-season alteration can move these numbers).
"""

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "park_factors.csv"


def load_park_factors(path=DEFAULT_PATH, value_col="park_factor"):
    """
    value_col defaults to "park_factor" (the hits table's column name) for
    backward compatibility. Pass value_col="hr_park_factor" when loading
    data/hr_park_factors.csv, which uses a different column name since it's
    a genuinely different metric -- see that file's notes for examples of
    where the two diverge (Fenway being hits-friendly but HR-suppressing).
    """
    df = pd.read_csv(path)
    return dict(zip(df["team"], df[value_col]))


def get_park_factor(team_name, factors=None, default=1.0):
    """Look up a team's park factor by name, falling back to league-neutral (1.0) if unknown."""
    if factors is None:
        factors = load_park_factors()
    return factors.get(team_name, default)
