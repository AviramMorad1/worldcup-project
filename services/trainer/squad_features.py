"""
squad_features.py
-----------------
Optional squad/player-strength features for 2026 World Cup predictions.

Data flow:
  1. Collector loads players_data_2025_2026.csv → raw_player_stats (DB)
  2. This module reads raw_player_stats from DB (or CSV fallback)
  3. Nation codes are mapped to national team names
  4. Optional: national_team_squads.csv or football-data.org API → raw_national_squads
  5. Optional: squad_player_matches built from raw_national_squads ↔ raw_player_stats
  6. team_squad_strength computed per team (0–100 score)
  7. get_squad_strength_features(team_a, team_b) → (strength_a, strength_b)
  8. predictions.py applies conservative probability adjustment

If any data is missing, functions return neutral defaults (50.0) and the
existing prediction system works unchanged.

Squad data is ONLY used for 2026 prediction adjustment, NOT for historical
training, to avoid anachronistic contamination.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = "/app/data/worldcup.db"


def _datasets_dir() -> Path:
    """Docker mounts datasets at /app/datasets; locally use repo datasets/."""
    docker = Path("/app/datasets")
    if docker.is_dir():
        return docker
    here = Path(__file__).resolve().parent
    for candidate in (here.parents[2] / "datasets", here.parents[1] / "datasets"):
        if candidate.is_dir():
            return candidate
    return here.parent / "datasets"


DATASETS_DIR = _datasets_dir()

PLAYERS_CSV       = DATASETS_DIR / "players_data_2025_2026.csv"
MANUAL_SQUADS_CSV = DATASETS_DIR / "national_team_squads.csv"

# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

_SQUAD_ENABLED        = os.environ.get("SQUAD_FEATURES_ENABLED", "true").lower() == "true"
_SQUAD_ADJ_ENABLED    = os.environ.get("SQUAD_ADJUSTMENT_ENABLED", "true").lower() == "true"
_MAX_SQUAD_ADJ        = float(os.environ.get("MAX_SQUAD_PROBA_ADJUSTMENT", "0.08"))
EXPECTED_SQUAD_SIZE   = 26
TOP_LEAGUE_THRESHOLD  = 0.85

# Squad CSV used for 2026 strength (not historical training)
WC_PLAYERS_CSV_PATHS = [
    DATASETS_DIR / "squads" / "wc_players_with_stats.csv",
    DATASETS_DIR / "wc_players_with_stats.csv",
    Path("/app/datasets/squads/wc_players_with_stats.csv"),
    Path("/app/datasets/wc_players_with_stats.csv"),
]

ALL_SQUADS_CSV = DATASETS_DIR / "squads" / "all_squads.csv"

# Map squad CSV team names → prediction model team names
TEAM_CANONICAL: dict[str, str] = {
    "Czech Republic": "Czechia",
    "South Korea": "Korea Republic",
    "Korea Republic": "Korea Republic",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Turkey": "Türkiye",
    "DR Congo": "Congo DR",
    "Iran": "IR Iran",
    "United States": "United States",
    "USA": "United States",
}
_SQUAD_API_ENABLED    = os.environ.get("SQUAD_API_ENABLED", "true").lower() == "true"
_SQUAD_SOURCE         = os.environ.get("SQUAD_SOURCE", "football-data")
_SQUAD_SEASON         = os.environ.get("SQUAD_SEASON", "2026")
_API_TOKEN            = os.environ.get("FOOTBALL_DATA_API_TOKEN", "")

# ---------------------------------------------------------------------------
# FBref Nation code → national team name
# ---------------------------------------------------------------------------

NATION_CODE_MAP: dict[str, str] = {
    # WC 2026 participants
    "MEX": "Mexico",
    "KOR": "Korea Republic",
    "RSA": "South Africa",
    "CZE": "Czechia",
    "CAN": "Canada",
    "BIH": "Bosnia and Herzegovina",
    "QAT": "Qatar",
    "SUI": "Switzerland",
    "HAI": "Haiti",
    "BRA": "Brazil",
    "SCO": "Scotland",
    "MAR": "Morocco",
    "USA": "United States",
    "AUS": "Australia",
    "PAR": "Paraguay",
    "TUR": "Türkiye",
    "CIV": "Côte d'Ivoire",
    "GER": "Germany",
    "ECU": "Ecuador",
    "CUW": "Curaçao",
    "NED": "Netherlands",
    "SWE": "Sweden",
    "JPN": "Japan",
    "TUN": "Tunisia",
    "IRN": "IR Iran",
    "BEL": "Belgium",
    "NZL": "New Zealand",
    "EGY": "Egypt",
    "KSA": "Saudi Arabia",
    "ESP": "Spain",
    "URU": "Uruguay",
    "CPV": "Cabo Verde",
    "FRA": "France",
    "IRQ": "Iraq",
    "SEN": "Senegal",
    "NOR": "Norway",
    "ARG": "Argentina",
    "AUT": "Austria",
    "ALG": "Algeria",
    "JOR": "Jordan",
    "POR": "Portugal",
    "UZB": "Uzbekistan",
    "COD": "Congo DR",
    "COL": "Colombia",
    "GHA": "Ghana",
    "ENG": "England",
    "PAN": "Panama",
    "CRO": "Croatia",
    # Additional nations (not in WC 2026 but appear in dataset)
    "ITA": "Italy",
    "RUS": "Russia",
    "CHN": "China",
    "SRB": "Serbia",
    "DEN": "Denmark",
    "POL": "Poland",
    "GRE": "Greece",
    "WAL": "Wales",
    "NIR": "Northern Ireland",
    "ISR": "Israel",
    "GEO": "Georgia",
    "CMR": "Cameroon",
    "NGA": "Nigeria",
    "CHI": "Chile",
    "VEN": "Venezuela",
    "PER": "Peru",
    "CRC": "Costa Rica",
    "HND": "Honduras",
    "JAM": "Jamaica",
    "TRI": "Trinidad and Tobago",
    "BOL": "Bolivia",
    "ECU": "Ecuador",
    "PER": "Peru",
    "HUN": "Hungary",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "MDA": "Moldova",
    "ALB": "Albania",
    "AZE": "Azerbaijan",
    "ARM": "Armenia",
    "KAZ": "Kazakhstan",
    "UKR": "Ukraine",
    "FIN": "Finland",
    "ISL": "Iceland",
    "LUX": "Luxembourg",
    "BUL": "Bulgaria",
    "ROU": "Romania",
    "TUR": "Türkiye",
    "BLR": "Belarus",
    "LTU": "Lithuania",
    "LVA": "Latvia",
    "EST": "Estonia",
    "AND": "Andorra",
    "SMR": "San Marino",
    "MLT": "Malta",
    "CYP": "Cyprus",
    "MKD": "North Macedonia",
    "MNE": "Montenegro",
    "KOS": "Kosovo",
    "BOS": "Bosnia and Herzegovina",
    "IRL": "Republic of Ireland",
    "BOH": "Bohemia",
    "CGO": "Congo",
    "TOG": "Togo",
    "MLI": "Mali",
    "GUI": "Guinea",
    "BFA": "Burkina Faso",
    "MAD": "Madagascar",
    "MOZ": "Mozambique",
    "ZIM": "Zimbabwe",
    "ZAM": "Zambia",
    "TAN": "Tanzania",
    "ANG": "Angola",
    "BEN": "Benin",
    "LBR": "Liberia",
    "SLE": "Sierra Leone",
    "GAM": "Gambia",
    "GNB": "Guinea-Bissau",
    "CAP": "Cape Verde",
    "COM": "Comoros",
    "DJI": "Djibouti",
    "SOM": "Somalia",
    "ETH": "Ethiopia",
    "ERI": "Eritrea",
    "SUD": "Sudan",
    "LBA": "Libya",
    "MTN": "Mauritania",
    "NIG": "Niger",
    "TCD": "Chad",
    "CMR": "Cameroon",
    "GAB": "Gabon",
    "GNE": "Equatorial Guinea",
    "RWA": "Rwanda",
    "BUR": "Burundi",
    "UGA": "Uganda",
    "KEN": "Kenya",
    "SOM": "Somalia",
    "DRC": "Congo DR",
    "ZAF": "South Africa",
    "LES": "Lesotho",
    "SWZ": "Eswatini",
    "NAM": "Namibia",
    "BOT": "Botswana",
    "MWI": "Malawi",
    "SYR": "Syria",
    "LBN": "Lebanon",
    "JOR": "Jordan",
    "PSE": "Palestine",
    "YEM": "Yemen",
    "SAU": "Saudi Arabia",
    "KUW": "Kuwait",
    "UAE": "United Arab Emirates",
    "OMA": "Oman",
    "QAT": "Qatar",
    "BHR": "Bahrain",
    "IDN": "Indonesia",
    "THA": "Thailand",
    "VIE": "Vietnam",
    "PHI": "Philippines",
    "MYS": "Malaysia",
    "SGP": "Singapore",
    "IND": "India",
    "PAK": "Pakistan",
    "BAN": "Bangladesh",
    "SRI": "Sri Lanka",
    "NEP": "Nepal",
    "AFG": "Afghanistan",
    "IRN": "IR Iran",
    "TJK": "Tajikistan",
    "TKM": "Turkmenistan",
    "KGZ": "Kyrgyzstan",
    "MNG": "Mongolia",
    "CHN": "China",
    "HKG": "Hong Kong",
    "MAC": "Macau",
    "TWN": "Chinese Taipei",
    "PRK": "Korea DPR",
    "NKO": "Korea DPR",
    "MAS": "Malaysia",
    "NZL": "New Zealand",
    "FIJ": "Fiji",
    "PNG": "Papua New Guinea",
    "SOL": "Solomon Islands",
    "TAH": "Tahiti",
    "CUB": "Cuba",
    "HAI": "Haiti",
    "DOM": "Dominican Republic",
    "PUR": "Puerto Rico",
    "GUY": "Guyana",
    "SUR": "Suriname",
    "GUF": "French Guiana",
}

# League weights loaded from datasets/league_strengths.csv (see league_weights.py)
from league_weights import lookup_league_weight, load_league_weights  # noqa: E402

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_PLAYER_STATS = """
    CREATE TABLE IF NOT EXISTS raw_player_stats (
        id                     TEXT PRIMARY KEY,
        player_name            TEXT,
        normalized_player_name TEXT,
        national_team          TEXT,
        nation_code            TEXT,
        position               TEXT,
        club                   TEXT,
        league                 TEXT,
        normalized_league_name TEXT,
        league_weight          REAL,
        age                    REAL,
        caps                   REAL,
        goals_intl             REAL,
        is_captain             INTEGER,
        appearances            REAL,
        starts                 REAL,
        minutes                REAL,
        nineties               REAL,
        goals                  REAL,
        assists                REAL,
        goals_assists          REAL,
        shots                  REAL,
        shots_on_target        REAL,
        saves                  REAL,
        clean_sheets           REAL,
        yellow_cards           REAL,
        red_cards              REAL,
        crosses                REAL,
        interceptions          REAL,
        tackles_won            REAL,
        plus_minus             REAL,
        points_per_match       REAL,
        source_file            TEXT,
        collected_at           TEXT
    )
"""

_PLAYER_STATS_COLS = [
    "id", "player_name", "normalized_player_name", "national_team", "nation_code",
    "position", "club", "league", "normalized_league_name", "league_weight",
    "age", "caps", "goals_intl", "is_captain",
    "appearances", "starts", "minutes", "nineties",
    "goals", "assists", "goals_assists",
    "shots", "shots_on_target", "saves", "clean_sheets",
    "yellow_cards", "red_cards", "crosses", "interceptions", "tackles_won",
    "plus_minus", "points_per_match", "source_file", "collected_at",
]

_RAW_PLAYER_STATS_MIGRATIONS = [
    ("national_team", "TEXT"),
    ("normalized_league_name", "TEXT"),
    ("league_weight", "REAL"),
    ("caps", "REAL"),
    ("goals_intl", "REAL"),
    ("is_captain", "INTEGER"),
    ("goals_assists", "REAL"),
]

_TEAM_TO_CODE: dict[str, str] = {}
for _code, _team in NATION_CODE_MAP.items():
    _TEAM_TO_CODE[_team] = _code
    canon = TEAM_CANONICAL.get(_team, _team)
    _TEAM_TO_CODE[canon] = _code

_DDL_NATIONAL_SQUADS = """
    CREATE TABLE IF NOT EXISTS raw_national_squads (
        id                     TEXT PRIMARY KEY,
        team                   TEXT,
        normalized_team        TEXT,
        player_name            TEXT,
        normalized_player_name TEXT,
        position               TEXT,
        age                    REAL,
        club                   TEXT,
        league                 TEXT,
        source                 TEXT,
        is_key_player          INTEGER DEFAULT 0,
        is_injured             INTEGER DEFAULT 0,
        collected_at           TEXT
    )
"""

_DDL_SQUAD_PLAYER_MATCHES = """
    CREATE TABLE IF NOT EXISTS squad_player_matches (
        id                     TEXT PRIMARY KEY,
        team                   TEXT,
        player_name            TEXT,
        normalized_player_name TEXT,
        stats_player_name      TEXT,
        match_score            REAL,
        matched_by             TEXT,
        has_stats              INTEGER,
        position               TEXT,
        club                   TEXT,
        league                 TEXT,
        minutes                REAL,
        goals                  REAL,
        assists                REAL,
        source                 TEXT,
        created_at             TEXT
    )
"""

_DDL_SQUAD_STRENGTH = """
    CREATE TABLE IF NOT EXISTS team_squad_strength (
        team                      TEXT PRIMARY KEY,
        players_with_stats        INTEGER,
        expected_squad_size       INTEGER,
        missing_players           INTEGER,
        coverage_ratio            REAL,
        coverage_tier             TEXT,
        data_quality_score        REAL,
        avg_age                     REAL,
        total_minutes             REAL,
        total_starts              REAL,
        regular_starters          INTEGER,
        total_goals               REAL,
        total_assists             REAL,
        total_goals_assists       REAL,
        total_shots               REAL,
        total_sot                 REAL,
        avg_league_weight         REAL,
        top_league_players        INTEGER,
        weighted_minutes_score    REAL,
        league_quality_score      REAL,
        production_score          REAL,
        international_experience_score REAL,
        depth_score               REAL,
        attack_strength           REAL,
        midfield_strength         REAL,
        defense_strength          REAL,
        goalkeeper_strength       REAL,
        computed_squad_strength   REAL,
        fifa_fallback_strength    REAL,
        final_squad_strength      REAL,
        updated_at                TEXT
    )
"""

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalise_name(name: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    lower = ascii_str.lower()
    clean = re.sub(r"[^a-z0-9\s]", " ", lower)
    return re.sub(r"\s+", " ", clean).strip()


def parse_nation_code(nation_str: str) -> tuple[str, str]:
    """
    Parse FBref Nation field like 'us USA' → ('USA', 'United States').
    Returns (code, team_name).
    """
    if not isinstance(nation_str, str) or not nation_str.strip():
        return ("", "")
    parts = nation_str.strip().split()
    if len(parts) >= 2:
        code = parts[-1].upper()
    else:
        code = parts[0].upper()
    team = NATION_CODE_MAP.get(code, "")
    return (code, team)


def canonical_team(name: str) -> str:
    """Map squad CSV / stats team name to prediction team name."""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    return TEAM_CANONICAL.get(s, s)


def league_weight(comp: str) -> float:
    """Return quality weight for a league/competition string."""
    _, w = lookup_league_weight(comp or "")
    return w


def _coverage_tier(n_with_stats: int) -> tuple[str, float, float]:
    """Return (tier_label, coverage_weight, data_quality_score)."""
    if n_with_stats >= 20:
        return ("High", 1.00, 1.00)
    if n_with_stats >= 13:
        return ("Medium", 0.70, 0.70)
    if n_with_stats >= 6:
        return ("Low", 0.40, 0.40)
    return ("Very Low", 0.20, 0.20)


def _fifa_fallback_from_rank(rank: float) -> float:
    """FIFA rank → fallback strength 0–100 (missing rank → 45)."""
    if rank is None or rank >= 150:
        return 45.0
    r = int(rank)
    if r <= 10:
        return 78.0
    if r <= 25:
        return 70.0
    if r <= 50:
        return 62.0
    if r <= 75:
        return 54.0
    if r <= 100:
        return 47.0
    return 40.0


def _load_fifa_ranks() -> dict[str, float]:
    """Latest FIFA rank per team (prefers year 2026)."""
    ranks: dict[str, float] = {}
    try:
        conn = _open_db()
        try:
            df = pd.read_sql_query(
                "SELECT team, year, rank FROM raw_rankings ORDER BY year DESC",
                conn,
            )
        finally:
            conn.close()
        if df.empty:
            return ranks
        for team in df["team"].unique():
            sub = df[df["team"] == team]
            row = sub[sub["year"] == 2026]
            if row.empty:
                row = sub.iloc[[0]]
            ranks[team] = float(row.iloc[0]["rank"])
            canon = canonical_team(team)
            if canon:
                ranks[canon] = ranks[team]
    except Exception as exc:
        logger.warning("Could not load FIFA ranks for fallback: %s", exc)
    return ranks


def _expected_squad_sizes() -> dict[str, int]:
    """Expected roster size per team from all_squads.csv, default 26."""
    sizes: dict[str, int] = {}
    for path in (ALL_SQUADS_CSV, Path("/app/datasets/squads/all_squads.csv")):
        if path.exists():
            try:
                df = pd.read_csv(path)
                for team, cnt in df.groupby("team").size().items():
                    canon = canonical_team(str(team))
                    sizes[canon] = int(cnt)
            except Exception:
                pass
            break
    return sizes


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    if not _table_exists(conn, table):
        return False
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)


def _safe_add_col(conn: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
    if not _col_exists(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        conn.commit()
        logger.info("squad_features: added column '%s' to %s.", col, table)


_TEAM_SQUAD_MIGRATIONS = [
    ("players_with_stats", "INTEGER"),
    ("expected_squad_size", "INTEGER"),
    ("missing_players", "INTEGER"),
    ("coverage_ratio", "REAL"),
    ("coverage_tier", "TEXT"),
    ("total_starts", "REAL"),
    ("total_goals_assists", "REAL"),
    ("avg_league_weight", "REAL"),
    ("weighted_minutes_score", "REAL"),
    ("league_quality_score", "REAL"),
    ("production_score", "REAL"),
    ("international_experience_score", "REAL"),
    ("depth_score", "REAL"),
    ("computed_squad_strength", "REAL"),
    ("fifa_fallback_strength", "REAL"),
    ("final_squad_strength", "REAL"),
]


def _migrate_team_squad_strength(conn: sqlite3.Connection) -> None:
    """Upgrade legacy team_squad_strength schema without dropping rows."""
    if not _table_exists(conn, "team_squad_strength"):
        return
    existing = {r[1] for r in conn.execute("PRAGMA table_info(team_squad_strength)")}
    for col, ctype in _TEAM_SQUAD_MIGRATIONS:
        if col not in existing:
            _safe_add_col(conn, "team_squad_strength", col, ctype)
    if "overall_squad_strength" in existing and "final_squad_strength" in existing:
        conn.execute(
            """
            UPDATE team_squad_strength
            SET final_squad_strength = COALESCE(final_squad_strength, overall_squad_strength),
                computed_squad_strength = COALESCE(
                    computed_squad_strength, overall_squad_strength
                ),
                players_with_stats = COALESCE(players_with_stats, matched_players, squad_size)
            WHERE final_squad_strength IS NULL OR players_with_stats IS NULL
            """
        )
        conn.commit()


def _migrate_raw_player_stats(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "raw_player_stats"):
        return
    for col, ctype in _RAW_PLAYER_STATS_MIGRATIONS:
        _safe_add_col(conn, "raw_player_stats", col, ctype)


def create_squad_tables() -> None:
    """Create all squad tables if they do not already exist."""
    conn = _open_db()
    try:
        conn.execute(_DDL_PLAYER_STATS)
        conn.execute(_DDL_NATIONAL_SQUADS)
        conn.execute(_DDL_SQUAD_PLAYER_MATCHES)
        conn.execute(_DDL_SQUAD_STRENGTH)
        conn.commit()
        _migrate_raw_player_stats(conn)
        _migrate_team_squad_strength(conn)
        _migrate_match_predictions(conn)
    finally:
        conn.close()
    logger.info("Squad tables created / verified.")


def _migrate_match_predictions(conn: sqlite3.Connection) -> None:
    """Add squad-adjustment columns to match_predictions if missing."""
    if not _table_exists(conn, "match_predictions"):
        return
    new_cols = [
        ("base_team_a_proba", "REAL"),
        ("base_draw_proba", "REAL"),
        ("base_team_b_proba", "REAL"),
        ("adjusted_team_a_proba", "REAL"),
        ("adjusted_draw_proba", "REAL"),
        ("adjusted_team_b_proba", "REAL"),
        ("squad_strength_a", "REAL"),
        ("squad_strength_b", "REAL"),
        ("squad_strength_diff", "REAL"),
        ("squad_coverage_a", "REAL"),
        ("squad_coverage_b", "REAL"),
        ("squad_coverage_tier_a", "TEXT"),
        ("squad_coverage_tier_b", "TEXT"),
        ("squad_adjustment_amount", "REAL"),
        ("squad_adjustment_applied", "INTEGER"),
        ("raw_model_confidence", "REAL"),
        ("adjusted_confidence", "REAL"),
        ("squad_adjusted_confidence", "REAL"),
        ("confidence_band", "TEXT"),
        ("key_factors", "TEXT"),
        ("explanation", "TEXT"),
    ]
    for col, ctype in new_cols:
        _safe_add_col(conn, "match_predictions", col, ctype)


# ---------------------------------------------------------------------------
# Player stats loader (DB → DataFrame)
# ---------------------------------------------------------------------------


def load_player_stats_from_db() -> pd.DataFrame:
    """Load raw_player_stats from DB into a DataFrame."""
    try:
        conn = _open_db()
        try:
            if not _table_exists(conn, "raw_player_stats"):
                return pd.DataFrame()
            df = pd.read_sql_query("SELECT * FROM raw_player_stats", conn)
        finally:
            conn.close()
        return df
    except Exception as exc:
        logger.warning("Could not load raw_player_stats from DB: %s", exc)
        return pd.DataFrame()


def _float_val(row, *cols: str) -> float | None:
    for col in cols:
        v = row.get(col)
        if v is None or str(v).strip() in ("", "nan", "None"):
            continue
        try:
            return float(v)
        except (ValueError, TypeError):
            continue
    return None


def load_wc_player_stats_dataframe() -> pd.DataFrame:
    """Load datasets/squads/wc_players_with_stats.csv (same format as collector)."""
    load_league_weights()
    path = None
    for candidate in WC_PLAYERS_CSV_PATHS:
        if candidate.exists():
            path = candidate
            break
    if path is None:
        return pd.DataFrame()

    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read WC player stats %s: %s", path, exc)
        return pd.DataFrame()

    raw.columns = [c.strip() for c in raw.columns]
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []

    for idx, row in raw.iterrows():
        player = str(row.get("player_name", row.get("Player", ""))).strip()
        if not player:
            continue
        team = canonical_team(str(row.get("team", row.get("national_team", ""))).strip())
        if not team:
            _, team = parse_nation_code(str(row.get("Nation", "")))
            team = canonical_team(team)
        code = _TEAM_TO_CODE.get(team, "")
        league = str(row.get("stats_league", row.get("league", row.get("Comp", "")))).strip()
        norm_league, lw = lookup_league_weight(league)
        goals = _float_val(row, "goals_stats", "goals", "Gls") or 0.0
        ast = _float_val(row, "assists", "Ast") or 0.0
        ga = _float_val(row, "G+A", "goals_assists")
        if ga is None:
            ga = goals + ast
        cap = row.get("is_captain", 0)
        try:
            is_cap = int(float(cap)) if cap not in ("", None) else 0
        except (TypeError, ValueError):
            is_cap = 0

        records.append({
            "id": f"wc_{normalise_name(team)}_{normalise_name(player)}_{idx}",
            "player_name": player,
            "normalized_player_name": normalise_name(player),
            "national_team": team,
            "nation_code": code,
            "position": str(row.get("stats_position", row.get("position_wc", ""))).strip(),
            "club": str(row.get("club_wc", row.get("club", ""))).strip(),
            "league": league,
            "normalized_league_name": norm_league,
            "league_weight": lw,
            "age": _float_val(row, "age_wc", "age", "Age"),
            "caps": _float_val(row, "caps"),
            "goals_intl": _float_val(row, "goals_intl"),
            "is_captain": is_cap,
            "appearances": _float_val(row, "matches_played", "appearances", "MP"),
            "starts": _float_val(row, "starts", "Starts"),
            "minutes": _float_val(row, "minutes", "Min"),
            "nineties": _float_val(row, "90s", "nineties"),
            "goals": goals,
            "assists": ast,
            "goals_assists": ga,
            "shots": _float_val(row, "shots", "Sh"),
            "shots_on_target": _float_val(row, "shots_on_target", "SoT"),
            "saves": _float_val(row, "saves", "Saves"),
            "clean_sheets": _float_val(row, "clean_sheets", "CS"),
            "yellow_cards": _float_val(row, "yellow_cards", "CrdY"),
            "red_cards": _float_val(row, "red_cards", "CrdR"),
            "crosses": _float_val(row, "crosses", "Crs"),
            "interceptions": _float_val(row, "interceptions", "Int"),
            "tackles_won": _float_val(row, "tackles_won", "TklW"),
            "plus_minus": _float_val(row, "plus_minus", "+/-"),
            "points_per_match": _float_val(row, "points_per_match", "PPM"),
            "source_file": path.name,
            "collected_at": now,
        })

    if not records:
        return pd.DataFrame()
    logger.info("WC player stats CSV: %d players from %s.", len(records), path.name)
    return pd.DataFrame(records)


def load_player_stats_csv(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and normalise the players_data CSV directly from file.
    Returns a normalised DataFrame with columns matching raw_player_stats schema.
    """
    path = csv_path or PLAYERS_CSV
    if not path.exists():
        logger.info("Players stats CSV not found at %s — skipping.", path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read players CSV %s: %s", path, exc)
        return pd.DataFrame()

    logger.info("Players CSV: %d rows, %d columns from %s.", len(df), len(df.columns), path.name)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    rows = []
    now = datetime.now(timezone.utc).isoformat()

    for idx, row in df.iterrows():
        player = row.get("Player", "")
        if not isinstance(player, str) or not player.strip():
            continue

        nation_str = str(row.get("Nation", ""))
        code, team = parse_nation_code(nation_str)

        pos   = str(row.get("Pos", "")).strip()
        club  = str(row.get("Squad", "")).strip()
        comp  = str(row.get("Comp",  "")).strip()

        def _n(col: str, fallback: float = 0.0) -> float:
            v = row.get(col)
            try:
                return float(v) if v is not None and str(v) not in ("", "nan") else fallback
            except (ValueError, TypeError):
                return fallback

        rows.append({
            "id":                    f"ps_{idx}",
            "player_name":           player.strip(),
            "normalized_player_name": normalise_name(player),
            "nation_code":           code,
            "national_team":         team,
            "position":              pos,
            "club":                  club,
            "league":                comp,
            "age":                   _n("Age"),
            "appearances":           _n("MP"),
            "starts":                _n("Starts"),
            "minutes":               _n("Min"),
            "nineties":              _n("90s"),
            "goals":                 _n("Gls"),
            "assists":               _n("Ast"),
            "shots":                 _n("Sh"),
            "shots_on_target":       _n("SoT"),
            "saves":                 _n("Saves"),
            "clean_sheets":          _n("CS"),
            "yellow_cards":          _n("CrdY"),
            "red_cards":             _n("CrdR"),
            "crosses":               _n("Crs"),
            "interceptions":         _n("Int"),
            "tackles_won":           _n("TklW"),
            "plus_minus":            _n("+/-"),
            "points_per_match":      _n("PPM"),
            "league_weight":         league_weight(comp),
            "source_file":           path.name,
            "collected_at":          now,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Manual squad loader
# ---------------------------------------------------------------------------


def load_manual_squads_csv(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load datasets/national_team_squads.csv if it exists.
    Expected columns: team, player_name, position, age, club, league,
                      is_key_player, is_injured.
    """
    path = csv_path or MANUAL_SQUADS_CSV
    if not path.exists():
        logger.info("Manual squads CSV not found at %s — skipping.", path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read manual squads CSV: %s", exc)
        return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]
    required = {"team", "player_name"}
    if not required.issubset(df.columns):
        logger.warning(
            "Manual squads CSV missing required columns %s — found: %s",
            required, list(df.columns),
        )
        return pd.DataFrame()

    logger.info(
        "Manual squads CSV: %d rows, %d team(s).",
        len(df),
        df["team"].nunique() if "team" in df.columns else 0,
    )
    return df


# ---------------------------------------------------------------------------
# API squad fetcher
# ---------------------------------------------------------------------------


def fetch_squads_from_api() -> pd.DataFrame:
    """
    Attempt to fetch national team squads from football-data.org.

    NOTE: The football-data.org free tier (Plan: Free/v4) does NOT expose
    national team squad endpoints (these are available on paid tiers only).
    This function logs a clear warning and returns an empty DataFrame.
    """
    if not _SQUAD_API_ENABLED:
        logger.info("Squad API disabled (SQUAD_API_ENABLED=false).")
        return pd.DataFrame()

    if not _API_TOKEN:
        logger.info(
            "FOOTBALL_DATA_API_TOKEN not set — skipping squad API fetch. "
            "To enable, set FOOTBALL_DATA_API_TOKEN in .env."
        )
        return pd.DataFrame()

    try:
        import urllib.request  # stdlib only

        url = "https://api.football-data.org/v4/competitions/WC/teams"
        req = urllib.request.Request(url)
        req.add_header("X-Auth-Token", _API_TOKEN)
        req.add_header("User-Agent", "worldcup-analytics/1.0")

        with urllib.request.urlopen(req, timeout=10) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning(
            "football-data.org squad API fetch failed: %s. "
            "This is expected on the free tier which does not provide squad endpoints. "
            "Using fallback data sources.",
            exc,
        )
        return pd.DataFrame()

    # football-data.org v4 free tier returns competition teams but not squad rosters
    teams = data.get("teams", [])
    if not teams:
        logger.warning(
            "football-data.org API: no squad data returned. "
            "The free tier does not expose national team squad rosters. "
            "Place datasets/national_team_squads.csv manually for squad filtering."
        )
        return pd.DataFrame()

    logger.info("football-data.org API: %d team(s) returned.", len(teams))
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Save player stats to DB
# ---------------------------------------------------------------------------


def save_player_stats_to_db(df: pd.DataFrame, replace_all: bool = False) -> int:
    """Insert player stats into raw_player_stats."""
    if df.empty:
        return 0

    for col in _PLAYER_STATS_COLS:
        if col not in df.columns:
            df[col] = None

    conn = _open_db()
    try:
        _migrate_raw_player_stats(conn)
        if replace_all:
            conn.execute("DELETE FROM raw_player_stats")
            conn.commit()
            to_write = df
        else:
            existing_ids = {
                r[0] for r in conn.execute("SELECT id FROM raw_player_stats").fetchall()
            }
            to_write = df[~df["id"].isin(existing_ids)]
            if to_write.empty:
                logger.info("raw_player_stats: all rows already present.")
                return 0

        rows = [
            tuple(
                None if (str(r[c]) in ("nan", "None", "")) else r[c]
                for c in _PLAYER_STATS_COLS
            )
            for _, r in to_write[_PLAYER_STATS_COLS].iterrows()
        ]
        conn.executemany(
            f"INSERT OR REPLACE INTO raw_player_stats ({', '.join(_PLAYER_STATS_COLS)}) "
            f"VALUES ({', '.join('?' * len(_PLAYER_STATS_COLS))})",
            rows,
        )
        conn.commit()
        logger.info("raw_player_stats: saved %d rows (replace_all=%s).", len(rows), replace_all)
        return len(rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Save national squads to DB
# ---------------------------------------------------------------------------


def save_national_squads_to_db(df: pd.DataFrame, source: str = "manual") -> int:
    """Insert national squad entries into raw_national_squads."""
    if df.empty:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    conn = _open_db()
    try:
        rows = []
        for i, r in df.iterrows():
            team        = str(r.get("team", "")).strip()
            player_name = str(r.get("player_name", "")).strip()
            if not team or not player_name:
                continue
            rows.append((
                f"sq_{source}_{i}",
                team,
                normalise_name(team),
                player_name,
                normalise_name(player_name),
                str(r.get("position", "")).strip(),
                float(r["age"]) if str(r.get("age", "")) not in ("", "nan") else None,
                str(r.get("club", "")).strip(),
                str(r.get("league", "")).strip(),
                source,
                int(r["is_key_player"]) if "is_key_player" in r else 0,
                int(r["is_injured"])    if "is_injured"    in r else 0,
                now,
            ))

        conn.executemany(
            "INSERT OR REPLACE INTO raw_national_squads "
            "(id, team, normalized_team, player_name, normalized_player_name, "
            " position, age, club, league, source, is_key_player, is_injured, collected_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        logger.info(
            "raw_national_squads: inserted/updated %d rows from source '%s'.", len(rows), source
        )
        return len(rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Player matching
# ---------------------------------------------------------------------------


def match_squad_players_to_stats() -> int:
    """
    Match raw_national_squads players to raw_player_stats entries.
    Stores results in squad_player_matches.
    Returns number of matched rows.
    """
    conn = _open_db()
    try:
        if not _table_exists(conn, "raw_national_squads"):
            return 0
        squad_count = conn.execute(
            "SELECT COUNT(*) FROM raw_national_squads"
        ).fetchone()[0]
        if squad_count == 0:
            return 0

        stats_df = pd.read_sql_query(
            "SELECT normalized_player_name, player_name, national_team, "
            "nation_code, position, club, league, minutes, goals, assists "
            "FROM raw_player_stats",
            conn,
        )

        squad_rows = conn.execute(
            "SELECT team, player_name, normalized_player_name, position, "
            "is_key_player, is_injured FROM raw_national_squads"
        ).fetchall()
    finally:
        conn.close()

    if stats_df.empty or not squad_rows:
        return 0

    # Build lookup: normalized_name → list of stats rows
    stats_lookup: dict[str, list[dict]] = {}
    for _, r in stats_df.iterrows():
        key = str(r["normalized_player_name"])
        stats_lookup.setdefault(key, []).append(r.to_dict())

    now = datetime.now(timezone.utc).isoformat()
    match_rows = []

    for sq in squad_rows:
        team        = sq[0]
        pname       = sq[1]
        norm_pname  = sq[2] or normalise_name(pname)
        pos         = sq[3]

        matched_by  = "unmatched"
        stats_entry = None
        match_score = 0.0

        # Try exact normalized name
        candidates = stats_lookup.get(norm_pname, [])
        if candidates:
            # Prefer same national team
            same_team = [c for c in candidates if c.get("national_team") == team]
            best = same_team[0] if same_team else sorted(
                candidates, key=lambda c: c.get("minutes", 0) or 0, reverse=True
            )[0]
            stats_entry = best
            matched_by  = "normalized_name"
            match_score = 1.0

        if stats_entry is None:
            matched_by = "unmatched"

        row_id = f"spm_{normalise_name(team)}_{normalise_name(pname)}"
        match_rows.append((
            row_id,
            team, pname, norm_pname,
            stats_entry["player_name"] if stats_entry else None,
            match_score, matched_by,
            1 if stats_entry else 0,
            pos or (stats_entry["position"] if stats_entry else None),
            stats_entry["club"]    if stats_entry else None,
            stats_entry["league"]  if stats_entry else None,
            float(stats_entry["minutes"])  if stats_entry and stats_entry.get("minutes")  else None,
            float(stats_entry["goals"])    if stats_entry and stats_entry.get("goals")    else None,
            float(stats_entry["assists"])  if stats_entry and stats_entry.get("assists")  else None,
            "squad_match", now,
        ))

    conn = _open_db()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO squad_player_matches "
            "(id, team, player_name, normalized_player_name, stats_player_name, "
            " match_score, matched_by, has_stats, position, club, league, "
            " minutes, goals, assists, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            match_rows,
        )
        conn.commit()
    finally:
        conn.close()

    matched = sum(1 for r in match_rows if r[7] == 1)
    logger.info(
        "squad_player_matches: %d/%d squad players matched to stats.",
        matched, len(match_rows),
    )
    return matched


# ---------------------------------------------------------------------------
# Squad strength computation
# ---------------------------------------------------------------------------

# Scoring component weights (must sum to 1.0)
_W_PLAYING_TIME = 0.35
_W_LEAGUE       = 0.25
_W_POSITIONAL   = 0.20
_W_INTL         = 0.10
_W_DEPTH        = 0.10

_BENCH_MINUTES = EXPECTED_SQUAD_SIZE * 900  # 26 × 900 min


def _clamp100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def _positional_score(players_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return (attack, midfield, defense, goalkeeper) strength scores 0–100."""
    def _pos_subset(df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
        pos = df["position"].fillna("").astype(str).str.upper()
        mask = pos.str.contains("|".join(patterns), na=False)
        return df[mask]

    att = _pos_subset(players_df, ["FW", "AM", "W"])
    mid = _pos_subset(players_df, ["MF", "CM", "DM"])
    dfe = _pos_subset(players_df, ["DF", "CB", "FB"])
    gkp = _pos_subset(players_df, ["GK"])

    def _attack_score(df: pd.DataFrame) -> float:
        if df.empty:
            return 50.0
        g_a = float(df["goals"].sum() + df["assists"].sum())
        sot = float(df["shots_on_target"].sum())
        sh = float(df["shots"].sum())
        mins = max(float(df["minutes"].sum()), 1.0)
        raw = (g_a * 5.0 + sot * 0.8 + sh * 0.2) / (mins / 90.0) * 8.0
        return _clamp100(raw)

    def _mid_score(df: pd.DataFrame) -> float:
        if df.empty:
            return 50.0
        ast = float(df["assists"].sum())
        crs = float(df.get("crosses", pd.Series(0)).sum())
        tkl = float(df["tackles_won"].sum())
        intr = float(df["interceptions"].sum())
        mins = max(float(df["minutes"].sum()), 1.0)
        raw = (ast * 4.0 + crs * 0.3 + tkl * 0.6 + intr * 0.6) / (mins / 90.0) * 8.0
        return _clamp100(raw)

    def _def_score(df: pd.DataFrame) -> float:
        if df.empty:
            return 50.0
        tkl = float(df["tackles_won"].sum())
        intr = float(df["interceptions"].sum())
        pm = float(df["plus_minus"].sum())
        cs = float(df["clean_sheets"].sum())
        mins = max(float(df["minutes"].sum()), 1.0)
        raw = (tkl * 0.6 + intr * 0.6 + max(0.0, pm) * 0.3 + cs * 4.0) / (mins / 90.0) * 8.0
        return _clamp100(raw)

    def _gk_score(df: pd.DataFrame) -> float:
        if df.empty:
            return 50.0
        sv = float(df["saves"].sum())
        cs = float(df["clean_sheets"].sum())
        mins = max(float(df["minutes"].sum()), 1.0)
        raw = (sv * 0.4 + cs * 6.0) / (mins / 90.0) * 8.0
        return _clamp100(raw)

    return (
        _attack_score(att),
        _mid_score(mid),
        _def_score(dfe),
        _gk_score(gkp),
    )


def _international_experience_score(df: pd.DataFrame) -> float:
    """Caps, international goals, captaincy → 0–100 (neutral 50 if missing)."""
    if "caps" not in df.columns and "goals_intl" not in df.columns:
        logger.debug("International columns missing — neutral intl score 50.")
        return 50.0
    caps = pd.to_numeric(df.get("caps", 0), errors="coerce").fillna(0)
    gintl = pd.to_numeric(df.get("goals_intl", 0), errors="coerce").fillna(0)
    captain = pd.to_numeric(df.get("is_captain", 0), errors="coerce").fillna(0)
    cap_score = min(100.0, float(caps.mean()) * 1.2) if caps.sum() > 0 else 50.0
    goal_score = min(100.0, float(gintl.sum()) * 3.0)
    cap_bonus = min(15.0, float(captain.sum()) * 5.0)
    return _clamp100(0.6 * cap_score + 0.3 * goal_score + 0.1 * (50.0 + cap_bonus))


def compute_team_squad_strength() -> int:
    """
    Compute team_squad_strength from raw_player_stats (2026 adjustment only).

    Missing squad members are uncertainty, not zero strength.
    Returns number of teams processed.
    """
    load_league_weights()
    stats_df = load_player_stats_from_db()
    if stats_df.empty:
        logger.warning(
            "raw_player_stats is empty — cannot compute squad strength. "
            "Run collector with wc_players_with_stats.csv."
        )
        return 0

    stats_df = stats_df[
        stats_df["national_team"].notna() & (stats_df["national_team"].astype(str).str.strip() != "")
    ].copy()
    stats_df["team_key"] = stats_df["national_team"].map(
        lambda t: canonical_team(str(t))
    )

    expected_sizes = _expected_squad_sizes()
    fifa_ranks = _load_fifa_ranks()
    now = datetime.now(timezone.utc).isoformat()
    results: list[tuple] = []
    tier_counts: dict[str, int] = {}

    for team in stats_df["team_key"].unique():
        if not team:
            continue
        tp = stats_df[stats_df["team_key"] == team].copy()
        num_cols = [
            "goals", "assists", "goals_assists", "shots", "shots_on_target",
            "saves", "clean_sheets", "minutes", "starts", "appearances",
            "tackles_won", "interceptions", "plus_minus", "league_weight", "age",
            "crosses", "caps", "goals_intl", "is_captain",
        ]
        for col in num_cols:
            if col in tp.columns:
                tp[col] = pd.to_numeric(tp[col], errors="coerce").fillna(0)
            else:
                tp[col] = 0.0
        if "goals_assists" in tp.columns and tp["goals_assists"].sum() == 0:
            tp["goals_assists"] = tp["goals"] + tp["assists"]

        players_with_stats = len(tp)
        expected = expected_sizes.get(team, EXPECTED_SQUAD_SIZE)
        missing_players = max(0, expected - players_with_stats)
        coverage_ratio = min(players_with_stats / float(expected), 1.0)
        tier, cov_weight, dq_score = _coverage_tier(players_with_stats)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        avg_age = float(tp["age"][tp["age"] > 0].mean() or 0)
        total_minutes = float(tp["minutes"].sum())
        total_starts = float(tp["starts"].sum())
        regular_starters = int((tp["starts"] >= 10).sum())
        total_goals = float(tp["goals"].sum())
        total_assists = float(tp["assists"].sum())
        total_ga = float(tp["goals_assists"].sum())
        total_shots = float(tp["shots"].sum())
        total_sot = float(tp["shots_on_target"].sum())

        total_min = max(total_minutes, 1.0)
        avg_lw = float((tp["league_weight"] * tp["minutes"]).sum() / total_min)
        top_league_players = int((tp["league_weight"] >= TOP_LEAGUE_THRESHOLD).sum())
        weighted_minutes_score = _clamp100(total_minutes / _BENCH_MINUTES * 100.0)
        league_quality_score = _clamp100(avg_lw * 100.0 + min(20.0, top_league_players * 2.0))

        att_s, mid_s, def_s, gk_s = _positional_score(tp)
        production_score = _clamp100((att_s + mid_s + def_s + gk_s) / 4.0)
        position_balance = production_score
        intl_score = _international_experience_score(tp)
        depth_score = _clamp100(coverage_ratio * 100.0)
        playing_time_score = _clamp100(
            0.7 * weighted_minutes_score + 0.3 * min(100.0, regular_starters / 15.0 * 100.0)
        )

        computed = (
            playing_time_score * _W_PLAYING_TIME
            + league_quality_score * _W_LEAGUE
            + position_balance * _W_POSITIONAL
            + intl_score * _W_INTL
            + depth_score * _W_DEPTH
        )
        computed = _clamp100(computed)

        rank = fifa_ranks.get(team)
        fifa_fb = 45.0 if rank is None else _fifa_fallback_from_rank(rank)
        final_strength = _clamp100(
            cov_weight * computed + (1.0 - cov_weight) * fifa_fb
        )

        results.append((
            team,
            players_with_stats,
            expected,
            missing_players,
            round(coverage_ratio, 4),
            tier,
            round(dq_score, 4),
            round(avg_age, 1),
            round(total_minutes, 0),
            round(total_starts, 0),
            regular_starters,
            round(total_goals, 0),
            round(total_assists, 0),
            round(total_ga, 0),
            round(total_shots, 0),
            round(total_sot, 0),
            round(avg_lw, 4),
            top_league_players,
            round(weighted_minutes_score, 2),
            round(league_quality_score, 2),
            round(production_score, 2),
            round(intl_score, 2),
            round(depth_score, 2),
            round(att_s, 2),
            round(mid_s, 2),
            round(def_s, 2),
            round(gk_s, 2),
            round(computed, 2),
            round(fifa_fb, 2),
            round(final_strength, 2),
            now,
        ))

    conn = _open_db()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO team_squad_strength ("
            "team, players_with_stats, expected_squad_size, missing_players, "
            "coverage_ratio, coverage_tier, data_quality_score, avg_age, "
            "total_minutes, total_starts, regular_starters, total_goals, "
            "total_assists, total_goals_assists, total_shots, total_sot, "
            "avg_league_weight, top_league_players, weighted_minutes_score, "
            "league_quality_score, production_score, international_experience_score, "
            "depth_score, attack_strength, midfield_strength, defense_strength, "
            "goalkeeper_strength, computed_squad_strength, fifa_fallback_strength, "
            "final_squad_strength, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            results,
        )
        conn.commit()
    finally:
        conn.close()

    top10 = sorted(results, key=lambda r: r[29], reverse=True)[:10]
    bottom10 = sorted(results, key=lambda r: r[29])[:10]
    very_low = [r[0] for r in results if r[5] == "Very Low"]

    logger.info(
        "team_squad_strength: %d teams. Coverage tiers: %s",
        len(results), tier_counts,
    )
    logger.info("Top 10 final_squad_strength: %s", [(t[0], t[29]) for t in top10])
    logger.info("Bottom 10 final_squad_strength: %s", [(t[0], t[29]) for t in bottom10])
    if very_low:
        logger.info("Very Low coverage teams (%d): %s", len(very_low), very_low)

    return len(results)


# ---------------------------------------------------------------------------
# Feature output
# ---------------------------------------------------------------------------

_NEUTRAL_STRENGTH = 50.0


# Extra names used in DB / CSV vs 2026 prediction team labels
TEAM_LOOKUP_ALIASES: dict[str, list[str]] = {
    "IR Iran": ["IR Iran", "Iran"],
    "Korea Republic": ["Korea Republic", "South Korea"],
    "Congo DR": ["Congo DR", "DR Congo"],
    "Cabo Verde": ["Cabo Verde", "Cape Verde"],
    "Côte d'Ivoire": ["Côte d'Ivoire", "Ivory Coast"],
    "Czechia": ["Czechia", "Czech Republic"],
    "Türkiye": ["Türkiye", "Turkey"],
    "United States": ["United States", "USA"],
}


def _fetch_squad_row(conn: sqlite3.Connection, team: str) -> dict | None:
    names = [team] + TEAM_LOOKUP_ALIASES.get(team, [])
    sql = (
        "SELECT final_squad_strength, coverage_ratio, coverage_tier, "
        "data_quality_score, players_with_stats, computed_squad_strength, "
        "fifa_fallback_strength, avg_league_weight "
        "FROM team_squad_strength WHERE team = ?"
    )
    for name in names:
        row = conn.execute(sql, (name,)).fetchone()
        if row:
            return dict(row)
    if _col_exists(conn, "team_squad_strength", "overall_squad_strength"):
        cov_col = (
            "match_coverage"
            if _col_exists(conn, "team_squad_strength", "match_coverage")
            else "NULL"
        )
        for name in names:
            row = conn.execute(
                f"SELECT overall_squad_strength, {cov_col}, data_quality_score, "
                "matched_players, squad_size FROM team_squad_strength WHERE team = ?",
                (name,),
            ).fetchone()
            if row:
                overall = row[0]
                cov = row[1] if row[1] is not None else 0.0
                return {
                    "final_squad_strength": overall,
                    "coverage_ratio": cov,
                    "coverage_tier": "Unknown",
                    "data_quality_score": row[2],
                    "players_with_stats": row[3] or row[4],
                    "computed_squad_strength": overall,
                    "fifa_fallback_strength": overall,
                    "avg_league_weight": None,
                }
    return None


def get_squad_strength_features(
    team_a: str,
    team_b: str,
) -> tuple[float, float, float, bool, str, str, float, float, float, float]:
    """
    Return squad features for prediction adjustment.

    (strength_a, strength_b, diff, available, tier_a, tier_b,
     coverage_a, coverage_b, quality_a, quality_b)
    """
    empty = (
        _NEUTRAL_STRENGTH, _NEUTRAL_STRENGTH, 0.0, False,
        "Unknown", "Unknown", 0.0, 0.0, 0.0, 0.0,
    )
    if not _SQUAD_ENABLED:
        return empty

    try:
        conn = _open_db()
        try:
            if not _table_exists(conn, "team_squad_strength"):
                return empty
            ra = _fetch_squad_row(conn, team_a)
            rb = _fetch_squad_row(conn, team_b)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Could not fetch squad strength: %s", exc)
        return empty

    def _strength(row: dict | None) -> tuple[float, str, float, float]:
        if row is None:
            return _NEUTRAL_STRENGTH, "Unknown", 0.0, 0.0
        fs = row.get("final_squad_strength")
        if fs is None:
            fs = row.get("overall_squad_strength", _NEUTRAL_STRENGTH)
        return (
            float(fs),
            str(row.get("coverage_tier") or "Unknown"),
            float(row.get("coverage_ratio") or 0),
            float(row.get("data_quality_score") or 0),
        )

    sa, tier_a, cov_a, dq_a = _strength(ra)
    sb, tier_b, cov_b, dq_b = _strength(rb)
    available = ra is not None and rb is not None
    if not available:
        if ra is None:
            logger.debug("No squad strength for %s.", team_a)
        if rb is None:
            logger.debug("No squad strength for %s.", team_b)
    return (sa, sb, sa - sb, available, tier_a, tier_b, cov_a, cov_b, dq_a, dq_b)


def _max_adjustment_for_tiers(tier_a: str, tier_b: str) -> float:
    tiers = {tier_a, tier_b}
    if "Very Low" in tiers:
        return 0.02
    if "Low" in tiers:
        return 0.03
    if "Medium" in tiers:
        return 0.05
    return min(_MAX_SQUAD_ADJ, 0.08)


def apply_squad_adjustment(
    proba: list[float],
    strength_a: float,
    strength_b: float,
    available: bool,
    tier_a: str = "High",
    tier_b: str = "High",
) -> tuple[list[float], bool, float]:
    """
    Conservative squad adjustment on baseline probabilities.

    proba: [p_team_b, p_draw, p_team_a]
    Returns (adjusted_proba, applied, adjustment_amount signed).
    """
    if not _SQUAD_ENABLED or not _SQUAD_ADJ_ENABLED or not available:
        return proba, False, 0.0

    if len(proba) != 3:
        return proba, False, 0.0

    squad_diff = strength_a - strength_b
    norm_diff = max(-1.0, min(1.0, squad_diff / 30.0))
    max_adj = _max_adjustment_for_tiers(tier_a, tier_b)
    adjustment = norm_diff * max_adj

    if abs(adjustment) < 1e-6:
        return proba, False, 0.0

    p0, p1, p2 = float(proba[0]), float(proba[1]), float(proba[2])
    floor = 0.01
    take = abs(adjustment)

    if adjustment > 0:
        p2 += take
        side = p0 + p1
        if side > 0:
            p0 -= take * (p0 / side)
            p1 -= take * (p1 / side)
    else:
        p0 += take
        side = p2 + p1
        if side > 0:
            p2 -= take * (p2 / side)
            p1 -= take * (p1 / side)

    p0, p1, p2 = max(floor, p0), max(floor, p1), max(floor, p2)
    total = p0 + p1 + p2
    return [p0 / total, p1 / total, p2 / total], True, adjustment


# ---------------------------------------------------------------------------
# Explanation helper
# ---------------------------------------------------------------------------


def explain_squad_strength(
    team_a: str,
    team_b: str,
    adjustment_amount: float = 0.0,
) -> str:
    """Readable squad comparison note for predictions."""
    try:
        conn = _open_db()
        try:
            ra = _fetch_squad_row(conn, team_a)
            rb = _fetch_squad_row(conn, team_b)
        finally:
            conn.close()
    except Exception:
        return ""

    if ra is None and rb is None:
        return (
            "Player-stat coverage unavailable for both teams; "
            "prediction relies on FIFA ranking, ELO, and historical model features."
        )

    parts: list[str] = []
    for team, row in ((team_a, ra), (team_b, rb)):
        if row is None:
            parts.append(f"{team}: no squad strength row.")
            continue
        tier = row.get("coverage_tier", "Unknown")
        n = row.get("players_with_stats", 0)
        if tier in ("Low", "Very Low"):
            parts.append(
                f"{team} has {tier} player-stat coverage ({n} players with stats); "
                "missing players treated as unknown and squad influence reduced."
            )
        else:
            parts.append(
                f"{team} squad score {row['final_squad_strength']:.0f}/100 "
                f"({n} players, avg league weight {row.get('avg_league_weight', 0):.2f})."
            )

    if ra and rb:
        sa = float(ra["final_squad_strength"])
        sb = float(rb["final_squad_strength"])
        if sa > sb + 3:
            parts.append(f"{team_a} has stronger current squad profile and league quality.")
        elif sb > sa + 3:
            parts.append(f"{team_b} has stronger current squad profile and league quality.")

    if abs(adjustment_amount) >= 0.005:
        favored = team_a if adjustment_amount > 0 else team_b
        parts.append(
            f"Squad adjustment shifted probability toward {favored} by "
            f"{abs(adjustment_amount) * 100:.1f} percentage points (conservative cap)."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Setup entrypoint
# ---------------------------------------------------------------------------


def setup_squad_features() -> bool:
    """
    Main setup: creates tables, loads data, computes squad strength.

    Safe to call on every trainer cycle — skips re-loading when already populated.
    Returns True if squad strength data is available after setup.
    """
    if not _SQUAD_ENABLED:
        logger.info("Squad features disabled (SQUAD_FEATURES_ENABLED=false).")
        return False

    try:
        create_squad_tables()

        stats_df = load_player_stats_from_db()
        if len(stats_df) < 100:
            wc_df = load_wc_player_stats_dataframe()
            if not wc_df.empty:
                n = save_player_stats_to_db(wc_df, replace_all=True)
                logger.info("Loaded %d player stats from WC CSV into DB.", n)
                stats_df = load_player_stats_from_db()
            elif stats_df.empty:
                raw_df = load_player_stats_csv(PLAYERS_CSV)
                if not raw_df.empty:
                    save_player_stats_to_db(raw_df)
                    stats_df = load_player_stats_from_db()

        if stats_df.empty or len(stats_df) < 50:
            logger.warning(
                "No player stats in DB (need wc_players_with_stats.csv). "
                "Found %d rows.", len(stats_df),
            )
            return False

        # Try manual squads CSV
        manual_df = load_manual_squads_csv()
        if not manual_df.empty:
            save_national_squads_to_db(manual_df, source="manual")
            match_squad_players_to_stats()

        # Try API (expected to return empty on free tier)
        if _SQUAD_API_ENABLED and _API_TOKEN:
            api_df = fetch_squads_from_api()
            if not api_df.empty:
                save_national_squads_to_db(api_df, source="api")
                match_squad_players_to_stats()

        # Compute (or recompute) squad strength
        n_teams = compute_team_squad_strength()
        if n_teams == 0:
            logger.warning("No team squad strength computed.")
            return False

        logger.info(
            "Squad features ready: %d players, %d teams with strength scores.",
            len(stats_df), n_teams,
        )
        return True

    except Exception as exc:
        logger.warning(
            "squad_features.setup_squad_features() failed: %s. "
            "Predictions will proceed without squad adjustment.",
            exc,
            exc_info=True,
        )
        return False
