#!/usr/bin/env python3
"""
build_league_strengths.py
-------------------------
Build league strength weights from a global national-league ranking CSV
(primary), with Hebrew Wikipedia UEFA scrape or static fallback as backup.

Primary file (user-provided):
  datasets/global_national_league_rankings.csv
  Columns: Place, country, Confederation, Points

Output: datasets/league_strengths.csv
"""

from __future__ import annotations

import csv
import logging
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "datasets" / "league_strengths.csv"

GLOBAL_RANKING_CSV_PATHS = [
    REPO_ROOT / "datasets" / "global_national_league_rankings.csv",
    REPO_ROOT / "datasets" / "The Strongest National League in the World.csv",
    Path(r"c:\Users\nitib\OneDrive\שולחן העבודה\The Strongest National League in the World.csv"),
]

WIKI_SOURCE_URL = (
    "https://he.wikipedia.org/wiki/"
    "%D7%A9%D7%99%D7%98%D7%AA_%D7%94%D7%93%D7%99%D7%A8%D7%95%D7%92_%D7%A9%D7%9C_%D7%90%D7%95%D7%A4%22%D7%90"
)

FIELDNAMES = [
    "league_name",
    "normalized_league_name",
    "country",
    "rank",
    "points",
    "continent_rank",
    "class_mark",
    "strength_weight",
    "source",
    "source_url",
    "collected_at",
]

# Country name (English, normalized) → canonical domestic league for player-stats matching
COUNTRY_TO_LEAGUE: dict[str, str] = {
    "england": "Premier League",
    "brazil": "Brazil Série A",
    "italy": "Serie A",
    "spain": "La Liga",
    "germany": "Bundesliga",
    "france": "Ligue 1",
    "portugal": "Primeira Liga",
    "netherlands": "Eredivisie",
    "argentina": "Argentine Primera División",
    "belgium": "Belgian Pro League",
    "colombia": "Categoría Primera A",
    "turkey": "Süper Lig",
    "paraguay": "Primera División Paraguay",
    "czech republic": "Czech First League",
    "egypt": "Egyptian Premier League",
    "greece": "Greek Super League",
    "ecuador": "LigaPro",
    "scotland": "Scottish Premiership",
    "uruguay": "Uruguayan Primera División",
    "romania": "Liga I",
    "roumania": "Liga I",
    "croatia": "Croatian HNL",
    "serbia": "Serbian SuperLiga",
    "austria": "Austrian Bundesliga",
    "denmark": "Danish Superliga",
    "mexico": "Liga MX",
    "israel": "Israeli Premier League",
    "saudi arabia": "Saudi Pro League",
    "japan": "J1 League",
    "korea republic": "K League 1",
    "south korea": "K League 1",
    "cyprus": "Cypriot First Division",
    "morocco": "Botola Pro",
    "poland": "Ekstraklasa",
    "switzerland": "Swiss Super League",
    "ukraine": "Ukrainian Premier League",
    "norway": "Eliteserien",
    "chile": "Chilean Primera División",
    "algeria": "Algerian Ligue 1",
    "costa rica": "Costa Rican Primera División",
    "bulgaria": "First Professional League",
    "peru": "Liga 1",
    "azerbaijan": "Azerbaijan Premier League",
    "bolivia": "Bolivian Primera División",
    "northern ireland": "NIFL Premiership",
    "sweden": "Allsvenskan",
    "hungary": "NB I",
    "slovenia": "PrvaLiga",
    "south africa": "South African Premiership",
    "usa": "MLS",
    "united states": "MLS",
    "republic of ireland": "League of Ireland",
    "ireland": "League of Ireland",
    "latvia": "Virslīga",
    "qatar": "Qatar Stars League",
    "iran": "Persian Gulf Pro League",
    "australia": "A-League Men",
    "canada": "Canadian Premier League",
    "tunisia": "Tunisian Ligue 1",
    "senegal": "Ligue 1 Senegal",
    "ghana": "Ghana Premier League",
    "nigeria": "Nigeria Premier League",
    "cameroon": "Elite One",
    "ivory coast": "Ligue 1 Ivory Coast",
    "cote d'ivoire": "Ligue 1 Ivory Coast",
    "iraq": "Iraqi Premier League",
    "uzbekistan": "Uzbek Super League",
    "jordan": "Jordan Pro League",
    "haiti": "Ligue Haïtienne",
    "panama": "Liga Panameña",
    "curacao": "Curaçao League",
    "curaçao": "Curaçao League",
    "bosnia and herzegovina": "Premier League of Bosnia",
    "new zealand": "New Zealand National League",
}

HEBREW_COUNTRY_TO_LEAGUE: dict[str, tuple[str, str]] = {
    "אנגליה": ("England", "Premier League"),
    "איטליה": ("Italy", "Serie A"),
    "ספרד": ("Spain", "La Liga"),
    "גרמניה": ("Germany", "Bundesliga"),
    "צרפת": ("France", "Ligue 1"),
    "פורטוגל": ("Portugal", "Primeira Liga"),
    "הולנד": ("Netherlands", "Eredivisie"),
    "בלגיה": ("Belgium", "Belgian Pro League"),
    "טורקיה": ("Turkey", "Süper Lig"),
    "צ'כיה": ("Czech Republic", "Czech First League"),
    "יוון": ("Greece", "Greek Super League"),
    "פולין": ("Poland", "Ekstraklasa"),
    "דנמרק": ("Denmark", "Danish Superliga"),
    "נורווגיה": ("Norway", "Eliteserien"),
    "שווייץ": ("Switzerland", "Swiss Super League"),
    "אוסטריה": ("Austria", "Austrian Bundesliga"),
    "סקוטלנד": ("Scotland", "Scottish Premiership"),
    "שוודיה": ("Sweden", "Allsvenskan"),
    "קרואטיה": ("Croatia", "Croatian HNL"),
    "אוקראינה": ("Ukraine", "Ukrainian Premier League"),
    "סרביה": ("Serbia", "Serbian SuperLiga"),
    "רומניה": ("Romania", "Liga I"),
    "מקסיקו": ("Mexico", "Liga MX"),
    "ארגנטינה": ("Argentina", "Argentine Primera División"),
    "ברזיל": ("Brazil", "Brazil Série A"),
    "סעודיה": ("Saudi Arabia", "Saudi Pro League"),
    "יפן": ("Japan", "J1 League"),
    "דרום אפריקה": ("South Africa", "South African Premiership"),
    "ארצות הברית": ("USA", "MLS"),
}

FALLBACK_WEIGHTS: dict[str, float] = {
    "Premier League": 1.00,
    "La Liga": 0.95,
    "Serie A": 0.92,
    "Bundesliga": 0.92,
    "Ligue 1": 0.88,
    "Primeira Liga": 0.78,
    "Eredivisie": 0.75,
    "Belgian Pro League": 0.70,
    "Süper Lig": 0.70,
    "Championship": 0.68,
    "Brazil Série A": 0.68,
    "MLS": 0.65,
    "Liga MX": 0.65,
    "Saudi Pro League": 0.62,
    "Argentine Primera División": 0.62,
    "Scottish Premiership": 0.60,
    "J1 League": 0.58,
    "K League 1": 0.58,
    "Danish Superliga": 0.55,
    "Persian Gulf Pro League": 0.50,
    "A-League Men": 0.50,
    "Other known league": 0.50,
    "Unknown": 0.45,
}


def _norm(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_s).strip().lower()


def _clean_country(raw: str) -> str:
    """Strip whitespace / odd chars from CSV country field."""
    s = unicodedata.normalize("NFKC", str(raw))
    s = re.sub(r"[\s\xa0\u200b]+", " ", s).strip()
    return s


def _parse_points(val) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace(",", ".")
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _league_for_country(country: str) -> tuple[str, str]:
    """Return (display_country, league_name)."""
    c = _clean_country(country)
    key = _norm(c)
    league = COUNTRY_TO_LEAGUE.get(key)
    if league:
        return c, league
    return c, f"{c} Top Division"


def _find_global_csv() -> Path | None:
    for path in GLOBAL_RANKING_CSV_PATHS:
        if path.is_file():
            return path
    return None


def _load_global_rankings_csv() -> list[dict] | None:
    """Load user global ranking CSV (all confederations)."""
    path = _find_global_csv()
    if path is None:
        logger.info("Global ranking CSV not found in datasets/ or OneDrive path.")
        return None

    now = datetime.now(timezone.utc).isoformat()
    parsed: list[dict] = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None

        # Detect columns: Place, country, Confederation, Points
        for row in reader:
            if len(row) < 3:
                continue
            try:
                rank_i = int(str(row[0]).strip())
            except (TypeError, ValueError):
                continue

            country_raw = row[1] if len(row) > 1 else ""
            conf = str(row[2]).strip() if len(row) > 2 else ""
            pts_raw = row[3] if len(row) > 3 else row[-1]
            pts_f = _parse_points(pts_raw)
            if pts_f is None:
                continue

            country_en, league = _league_for_country(country_raw)
            parsed.append({
                "league_name": league,
                "normalized_league_name": _norm(league),
                "country": country_en,
                "rank": rank_i,
                "points": round(pts_f, 3),
                "confederation": conf.upper() if conf else "",
                "points_float": pts_f,
            })

    if len(parsed) < 5:
        logger.warning("Global CSV parsed fewer than 5 rows from %s", path)
        return None

    # Continent rank within confederation
    conf_ranks: dict[str, int] = {}
    for p in sorted(parsed, key=lambda x: x["rank"]):
        conf = p.get("confederation", "") or "OTHER"
        conf_ranks[conf] = conf_ranks.get(conf, 0) + 1
        p["continent_rank"] = conf_ranks[conf]
        p["class_mark"] = conf

    pts_list = [p["points_float"] for p in parsed]
    mn, mx = min(pts_list), max(pts_list)
    span = mx - mn if mx > mn else 1.0
    for p in parsed:
        w = 0.45 + 0.55 * (p["points_float"] - mn) / span
        p["strength_weight"] = round(max(0.45, min(1.0, w)), 4)
        p.pop("points_float", None)
        p["source"] = "global_rankings_csv"
        try:
            p["source_url"] = str(path.relative_to(REPO_ROOT))
        except ValueError:
            p["source_url"] = path.name
        p["collected_at"] = now

    logger.info("Loaded %d leagues from global rankings CSV: %s", len(parsed), path.name)
    return parsed


def _flatten_columns(df) -> None:
    df.columns = [
        "_".join(str(x).strip() for x in col if str(x) not in ("nan", "None"))
        for col in df.columns
    ]


def _try_scrape_uefa_wikipedia() -> list[dict] | None:
    """Backup: Hebrew Wikipedia UEFA table."""
    try:
        import pandas as pd
        import urllib.request
    except ImportError:
        return None

    try:
        req = urllib.request.Request(
            WIKI_SOURCE_URL,
            headers={"User-Agent": "worldcup-analytics/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        tables = pd.read_html(html)
    except Exception as exc:
        logger.info("Wikipedia backup failed: %s", exc)
        return None

    if not tables:
        return None

    df = tables[0].copy()
    _flatten_columns(df)
    rank_col = next((c for c in df.columns if "2026" in c and "מיקום" in c), None)
    country_col = next((c for c in df.columns if "מדינה" in c), None)
    total_col = next((c for c in df.columns if "סך" in c), None)
    if not all((rank_col, country_col, total_col)):
        return None

    now = datetime.now(timezone.utc).isoformat()
    parsed = []
    for i, row in df.iterrows():
        hebrew = str(row.get(country_col, "")).strip()
        pts_f = _parse_points(row.get(total_col))
        if not hebrew or pts_f is None:
            continue
        mapping = HEBREW_COUNTRY_TO_LEAGUE.get(hebrew)
        if mapping:
            country_en, league = mapping
        else:
            country_en, league = hebrew, f"{hebrew} League"
        try:
            rank_i = int(float(row.get(rank_col)))
        except (TypeError, ValueError):
            rank_i = i + 1
        parsed.append({
            "league_name": league,
            "normalized_league_name": _norm(league),
            "country": country_en,
            "rank": rank_i,
            "points": round(pts_f, 3),
            "continent_rank": rank_i,
            "class_mark": "UEFA",
            "points_float": pts_f,
        })

    if len(parsed) < 10:
        return None

    pts_list = [p["points_float"] for p in parsed]
    mn, mx = min(pts_list), max(pts_list)
    span = mx - mn if mx > mn else 1.0
    for p in parsed:
        w = 0.45 + 0.55 * (p["points_float"] - mn) / span
        p["strength_weight"] = round(max(0.45, min(1.0, w)), 4)
        p.pop("points_float", None)
        p["source"] = "uefa_wikipedia_he"
        p["source_url"] = WIKI_SOURCE_URL
        p["collected_at"] = now

    logger.info("Wikipedia backup: %d UEFA rows.", len(parsed))
    return parsed


def _rows_from_fallback() -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, (league, weight) in enumerate(
        sorted(FALLBACK_WEIGHTS.items(), key=lambda x: -x[1]), start=1
    ):
        if league in ("Other known league", "Unknown"):
            continue
        rows.append({
            "league_name": league,
            "normalized_league_name": _norm(league),
            "country": "",
            "rank": i,
            "points": "",
            "continent_rank": "",
            "class_mark": "",
            "strength_weight": round(weight, 4),
            "source": "fallback",
            "source_url": "",
            "collected_at": now,
        })
    return rows


def main() -> int:
    rows = _load_global_rankings_csv()
    source_note = "global_rankings_csv"

    if not rows:
        rows = _try_scrape_uefa_wikipedia()
        source_note = "uefa_wikipedia_he"

    if not rows:
        logger.warning(
            "League ranking source unavailable; using fallback league weights."
        )
        rows = _rows_from_fallback()
        source_note = "fallback"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    top = sorted(rows, key=lambda r: -float(r["strength_weight"]))[:10]
    print(f"\nSource: {source_note}")
    print(f"Leagues written: {len(rows)}")
    print(f"File: {OUTPUT.resolve()}")
    print("\nTop 10 league weights:")
    for r in top:
        pts = r.get("points", "")
        pts_s = f"  pts={pts}" if pts != "" else ""
        print(
            f"  {r['strength_weight']:.3f}  {r['league_name']} "
            f"({r.get('country', '')}) [{r.get('class_mark', '')}]{pts_s}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
