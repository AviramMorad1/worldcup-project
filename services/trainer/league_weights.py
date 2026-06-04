"""
league_weights.py — load league strength weights from CSV with fallback aliases.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LEAGUE_STRENGTHS_CSV = Path("/app/datasets/league_strengths.csv")
LEAGUE_STRENGTHS_LOCAL = Path("datasets/league_strengths.csv")

DEFAULT_WEIGHT = 0.45
OTHER_KNOWN_WEIGHT = 0.50

# Inline fallback if CSV missing (mirrors build_league_strengths.py)
FALLBACK_BY_CANONICAL: dict[str, float] = {
    "premier league": 1.00,
    "la liga": 0.95,
    "serie a": 0.92,
    "bundesliga": 0.92,
    "ligue 1": 0.88,
    "primeira liga": 0.78,
    "eredivisie": 0.75,
    "belgian pro league": 0.70,
    "süper lig": 0.70,
    "super lig": 0.70,
    "championship": 0.68,
    "brazil série a": 0.68,
    "brazil serie a": 0.68,
    "mls": 0.65,
    "liga mx": 0.65,
    "saudi pro league": 0.62,
    "argentine primera división": 0.62,
    "argentine primera division": 0.62,
    "scottish premiership": 0.60,
    "j1 league": 0.58,
    "k league 1": 0.58,
    "danish superliga": 0.55,
    "austrian bundesliga": 0.55,
    "swiss super league": 0.55,
    "greek super league": 0.55,
    "czech first league": 0.52,
    "south african premiership": 0.50,
    "persian gulf pro league": 0.50,
    "a-league men": 0.50,
}

ALIASES: dict[str, str] = {
    "eng premier league": "premier league",
    "es la liga": "la liga",
    "it serie a": "serie a",
    "de bundesliga": "bundesliga",
    "fr ligue 1": "ligue 1",
    "pt primeira liga": "primeira liga",
    "nl eredivisie": "eredivisie",
    "sa saudi pro league": "saudi pro league",
    "br brazilian série a": "brazil série a",
    "br brazilian serie a": "brazil série a",
    "tr türkiye süper lig": "süper lig",
    "tr turkiye super lig": "süper lig",
    "eng championship": "championship",
    "be belgian pro league": "belgian pro league",
    "cz czech first league": "czech first league",
    "sco scottish premiership": "scottish premiership",
    "ar argentine primera división": "argentine primera división",
    "za south african premiership": "south african premiership",
    "ir persian gulf pro league": "persian gulf pro league",
    "dk danish superliga": "danish superliga",
    "gr super league greece": "greek super league",
    "ch swiss super league": "swiss super league",
    "au a-league men": "a-league men",
    "mx liga mx": "liga mx",
    "major league soccer": "mls",
    "pl ekstraklasa": "ekstraklasa",
    "no eliteserien": "eliteserien",
    "se allsvenskan": "allsvenskan",
    "hr croatian hnl": "croatian hnl",
    "ua ukrainian premier league": "ukrainian premier league",
    "us major league soccer": "mls",
    "jp j1 league": "j1 league",
    "kr k league 1": "k league 1",
    "eg egyptian premier league": "egyptian premier league",
    "saudi pro league": "saudi pro league",
    "sa saudi professional league": "saudi pro league",
    "co categoría primera a": "categoría primera a",
    "co categoria primera a": "categoría primera a",
    "egyptian premier league": "egyptian premier league",
    "morocco botola pro": "botola pro",
    "ma botola pro": "botola pro",
    "j1 league": "j1 league",
    "k league 1": "k league 1",
    "ligue 1": "ligue 1",
    "fr ligue 1": "ligue 1",
    "championship": "championship",
    "eng championship": "championship",
}


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_s).strip().lower()


_weight_cache: dict[str, float] | None = None


def load_league_weights() -> dict[str, float]:
    """Return map normalized_league_name -> strength_weight."""
    global _weight_cache
    if _weight_cache is not None:
        return _weight_cache

    path = LEAGUE_STRENGTHS_CSV if LEAGUE_STRENGTHS_CSV.exists() else LEAGUE_STRENGTHS_LOCAL
    weights: dict[str, float] = dict(FALLBACK_BY_CANONICAL)

    if path.exists():
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                norm = str(row.get("normalized_league_name", "")).strip().lower()
                if not norm:
                    norm = _norm(str(row.get("league_name", "")))
                try:
                    w = float(row["strength_weight"])
                except (KeyError, TypeError, ValueError):
                    continue
                weights[norm] = max(0.45, min(1.0, w))
            logger.info("Loaded %d league weights from %s.", len(df), path.name)
        except Exception as exc:
            logger.warning("Could not read league_strengths.csv: %s — using fallback.", exc)
    else:
        logger.warning(
            "league_strengths.csv not found — using inline fallback league weights."
        )

    for alias, canonical in ALIASES.items():
        if canonical in weights:
            weights[alias] = weights[canonical]

    _weight_cache = weights
    return weights


def lookup_league_weight(league: str) -> tuple[str, float]:
    """
    Return (normalized_league_name, strength_weight) for a player stats league string.
    """
    weights = load_league_weights()
    n = _norm(league)
    if not n:
        return ("unknown", DEFAULT_WEIGHT)

    if n in weights:
        return (n, weights[n])
    if n in ALIASES and ALIASES[n] in weights:
        return (ALIASES[n], weights[ALIASES[n]])

    # Partial match on canonical keys
    for key, w in weights.items():
        if key in n or n in key:
            return (key, w)

    return (n, OTHER_KNOWN_WEIGHT)
