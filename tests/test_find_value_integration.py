"""
Integration test for find_value.py -- mocks odds_client and uses a real
temp ledger file to exercise the full matching + edge-calculation flow
without any network access.
"""

import os
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

os.environ.setdefault("ODDS_API_KEY", "test-key-not-real")

from src import odds_client as odds  # noqa: E402
from src.ledger import append_predictions  # noqa: E402

import find_value  # noqa: E402


FAKE_EVENTS = [{"event_id": "evt1", "home_team": "Houston Astros", "away_team": "Texas Rangers"}]

FAKE_BEST_ODDS = {
    # Model will say 75% -- a -150 price implies 60%, so this should clear
    # the default 5-point edge threshold (75% - 60% = 15 points).
    "Yordan Alvarez": {"price": -150, "bookmaker": "fanduel"},
    # Model will say 50% -- a -200 price implies ~67%, the model is WORSE
    # than the market here, so this should NOT be flagged.
    "Josh Jung": {"price": -200, "bookmaker": "betmgm"},
}


def _sample_prediction_rows():
    today = date.today().isoformat()
    return [
        {"date": today, "game_pk": 1, "player_id": 1, "player_name": "Yordan Alvarez",
         "team": "Houston Astros", "opponent": "Texas Rangers", "lineup_spot": 3,
         "venue": "Daikin Park", "park_factor": 1.0, "opponent_pitcher": "Some Pitcher",
         "p_hit": 0.75, "adjusted_ba": 0.30, "expected_ab": 4.1},
        {"date": today, "game_pk": 1, "player_id": 2, "player_name": "Josh Jung",
         "team": "Texas Rangers", "opponent": "Houston Astros", "lineup_spot": 6,
         "venue": "Daikin Park", "park_factor": 1.0, "opponent_pitcher": "Some Other Pitcher",
         "p_hit": 0.50, "adjusted_ba": 0.22, "expected_ab": 3.8},
    ]


def test_find_value_flags_only_real_edges(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as d:
        ledger_path = Path(d) / "predictions_log.csv"
        append_predictions(_sample_prediction_rows(), ledger_path)

        monkeypatch.setattr(find_value, "LEDGER_PATH", ledger_path)
        monkeypatch.setattr(odds, "get_todays_mlb_events", lambda: FAKE_EVENTS)
        monkeypatch.setattr(odds, "get_best_over_0_5_hits_odds", lambda event_id: FAKE_BEST_ODDS)

        find_value.main()
        output = capsys.readouterr().out

        assert "Yordan Alvarez" in output
        assert "Josh Jung" not in output  # model is worse than market here, shouldn't be flagged


if __name__ == "__main__":
    import io
    import contextlib

    class _FakeCapsys:
        def readouterr(self):
            return _Result(self._buf.getvalue())

        def __init__(self, buf):
            self._buf = buf

    class _Result:
        def __init__(self, out):
            self.out = out

    class _MonkeyPatch:
        def __init__(self):
            self._originals = []

        def setattr(self, obj, name, value):
            self._originals.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, val in self._originals:
                setattr(obj, name, val)

    mp = _MonkeyPatch()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        test_find_value_flags_only_real_edges(mp, _FakeCapsys(buf))
    mp.undo()
    print("  ok  test_find_value_flags_only_real_edges")
    print("\nAll 1 tests passed.")
