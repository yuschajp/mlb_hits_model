"""Tests for name_matching.py -- pure string logic, no network required."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.name_matching import build_name_index, match_name, normalize_name


def test_normalize_strips_accents():
    assert normalize_name("Andrés Giménez") == "andres gimenez"


def test_normalize_handles_suffixes():
    assert normalize_name("Bobby Witt Jr.") == "bobby witt"
    assert normalize_name("Bobby Witt Jr") == "bobby witt"


def test_normalize_handles_periods_and_case():
    assert normalize_name("C.J. Abrams") == normalize_name("CJ Abrams")


def test_normalize_collapses_whitespace():
    assert normalize_name("Shohei   Ohtani") == "shohei ohtani"


def test_normalize_handles_none():
    assert normalize_name(None) == ""


def test_match_name_finds_cross_vendor_match():
    index = build_name_index(["Andrés Giménez", "Bobby Witt Jr.", "Yordan Alvarez"])
    assert match_name("Andres Gimenez", index) == "Andrés Giménez"
    assert match_name("Bobby Witt", index) == "Bobby Witt Jr."
    assert match_name("Someone Else", index) is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
