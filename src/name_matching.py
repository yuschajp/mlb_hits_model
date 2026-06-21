"""
name_matching.py

MLB Stats API and The Odds API are two independent vendors, so the same
player can show up as "Andrés Giménez" in one and "Andres Gimenez" in the
other, or with/without a period after a suffix. This normalizes names so
exact-after-normalization matching works for the common cases: accents,
case, periods, and Jr./Sr./II/III suffixes.

This is NOT true fuzzy matching -- it won't catch genuine misspellings,
nicknames, or "Mike Trout" vs "Michael Trout" type differences. If a
meaningful number of matches fail in practice, the next step up would be
a library like rapidfuzz rather than expanding this by hand.
"""

import unicodedata

_SUFFIXES = (" jr.", " jr", " sr.", " sr", " ii", " iii", " iv")


def normalize_name(name):
    if name is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = no_accents.lower().replace(".", "")
    lowered = " ".join(lowered.split())  # collapse whitespace
    for suffix in _SUFFIXES:
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)].strip()
            break
    return lowered


def build_name_index(names):
    """Returns {normalized_name: original_name}. Later duplicates overwrite earlier ones."""
    return {normalize_name(n): n for n in names if n}


def match_name(name, index):
    """Looks up a name in a normalized index; returns the original matching name or None."""
    return index.get(normalize_name(name))
