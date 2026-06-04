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

DB_PATH      = "/app/data/worldcup.db"
DATASETS_DIR = Path("/app/datasets")

PLAYERS_CSV       = DATASETS_DIR / "players_data_2025_2026.csv"
MANUAL_SQUADS_CSV = DATASETS_DIR / "national_team_squads.csv"

# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

_SQUAD_ENABLED        = os.environ.get("SQUAD_FEATURES_ENABLED", "true").lower() == "true"
_SQUAD_ADJ_WEIGHT     = float(os.environ.get("SQUAD_ADJUSTMENT_WEIGHT",    "0.20"))
_MAX_SQUAD_ADJ        = float(os.environ.get("MAX_SQUAD_PROBA_ADJUSTMENT", "0.10"))
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

# ---------------------------------------------------------------------------
# League quality weights
# ---------------------------------------------------------------------------

LEAGUE_WEIGHTS: dict[str, float] = {
    "eng premier league": 1.00,
    "es la liga":         0.95,
    "de bundesliga":      0.90,
    "it serie a":         0.90,
    "fr ligue 1":         0.85,
}
DEFAULT_LEAGUE_WEIGHT = 0.50

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_PLAYER_STATS = """
    CREATE TABLE IF NOT EXISTS raw_player_stats (
        id                   TEXT PRIMARY KEY,
        player_name          TEXT,
        normalized_player_name TEXT,
        nation_code          TEXT,
        national_team        TEXT,
        position             TEXT,
        club                 TEXT,
        league               TEXT,
        age                  REAL,
        appearances          REAL,
        starts               REAL,
        minutes              REAL,
        nineties             REAL,
        goals                REAL,
        assists              REAL,
        shots                REAL,
        shots_on_target      REAL,
        saves                REAL,
        clean_sheets         REAL,
        yellow_cards         REAL,
        red_cards            REAL,
        crosses              REAL,
        interceptions        REAL,
        tackles_won          REAL,
        plus_minus           REAL,
        points_per_match     REAL,
        league_weight        REAL,
        source_file          TEXT,
        collected_at         TEXT
    )
"""

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
        team                   TEXT PRIMARY KEY,
        squad_size             INTEGER,
        matched_players        INTEGER,
        unmatched_players      INTEGER,
        match_coverage         REAL,
        avg_age                REAL,
        total_minutes          REAL,
        total_goals            REAL,
        total_assists          REAL,
        total_g_a              REAL,
        total_shots            REAL,
        total_sot              REAL,
        top_league_players     INTEGER,
        regular_starters       INTEGER,
        key_player_count       INTEGER,
        injured_key_players    INTEGER,
        attack_strength        REAL,
        midfield_strength      REAL,
        defense_strength       REAL,
        goalkeeper_strength    REAL,
        overall_squad_strength REAL,
        data_quality_score     REAL,
        updated_at             TEXT
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


def league_weight(comp: str) -> float:
    """Return quality weight for a competition string."""
    if not isinstance(comp, str):
        return DEFAULT_LEAGUE_WEIGHT
    return LEAGUE_WEIGHTS.get(comp.strip().lower(), DEFAULT_LEAGUE_WEIGHT)


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


def create_squad_tables() -> None:
    """Create all squad tables if they do not already exist."""
    conn = _open_db()
    try:
        conn.execute(_DDL_PLAYER_STATS)
        conn.execute(_DDL_NATIONAL_SQUADS)
        conn.execute(_DDL_SQUAD_PLAYER_MATCHES)
        conn.execute(_DDL_SQUAD_STRENGTH)
        conn.commit()

        # Migrate match_predictions to include squad columns
        _migrate_match_predictions(conn)
    finally:
        conn.close()
    logger.info("Squad tables created / verified.")


def _migrate_match_predictions(conn: sqlite3.Connection) -> None:
    """Add squad-adjustment columns to match_predictions if missing."""
    if not _table_exists(conn, "match_predictions"):
        return
    new_cols = [
        ("squad_strength_a",        "REAL"),
        ("squad_strength_b",        "REAL"),
        ("squad_strength_diff",     "REAL"),
        ("squad_adjustment_applied","INTEGER"),
        ("raw_model_confidence",    "REAL"),
        ("squad_adjusted_confidence","REAL"),
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


def save_player_stats_to_db(df: pd.DataFrame) -> int:
    """Insert player stats rows into raw_player_stats, skipping existing IDs."""
    if df.empty:
        return 0

    conn = _open_db()
    try:
        existing_ids = {
            r[0] for r in conn.execute("SELECT id FROM raw_player_stats").fetchall()
        }
        new_rows = df[~df["id"].isin(existing_ids)]
        if new_rows.empty:
            logger.info("raw_player_stats: all rows already present.")
            return 0

        cols = [
            "id", "player_name", "normalized_player_name",
            "nation_code", "national_team", "position",
            "club", "league", "age", "appearances", "starts",
            "minutes", "nineties", "goals", "assists",
            "shots", "shots_on_target", "saves", "clean_sheets",
            "yellow_cards", "red_cards", "crosses",
            "interceptions", "tackles_won", "plus_minus",
            "points_per_match", "league_weight",
            "source_file", "collected_at",
        ]
        rows = [
            tuple(None if (str(r[c]) in ("nan", "None", "")) else r[c] for c in cols)
            for _, r in new_rows[cols].iterrows()
        ]
        conn.executemany(
            f"INSERT OR IGNORE INTO raw_player_stats ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})",
            rows,
        )
        conn.commit()
        logger.info("raw_player_stats: inserted %d new rows.", len(rows))
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
_W_PLAYING_TIME  = 0.30
_W_POSITIONAL    = 0.25
_W_LEAGUE        = 0.20
_W_PRODUCTION    = 0.15
_W_DEPTH         = 0.10


def _positional_score(players_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return (attack, midfield, defense, goalkeeper) strength scores 0–100."""
    def _pos_subset(df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
        mask = df["position"].str.upper().str.contains("|".join(patterns), na=False)
        return df[mask]

    att = _pos_subset(players_df, ["FW"])
    mid = _pos_subset(players_df, ["MF"])
    dfe = _pos_subset(players_df, ["DF"])
    gkp = _pos_subset(players_df, ["GK"])

    def _attack_score(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        g_a = (df["goals"].sum() + df["assists"].sum())
        sot = df["shots_on_target"].sum()
        mins = max(df["minutes"].sum(), 1)
        raw = (g_a * 4 + sot * 0.5) / (mins / 90) * 10
        return min(100.0, raw)

    def _mid_score(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        ast  = df["assists"].sum()
        tkl  = df["tackles_won"].sum()
        intr = df["interceptions"].sum()
        mins = max(df["minutes"].sum(), 1)
        raw = (ast * 3 + tkl * 0.5 + intr * 0.5) / (mins / 90) * 10
        return min(100.0, raw)

    def _def_score(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        tkl  = df["tackles_won"].sum()
        intr = df["interceptions"].sum()
        pm   = df["plus_minus"].sum()
        mins = max(df["minutes"].sum(), 1)
        raw = (tkl * 0.5 + intr * 0.5 + max(0, pm) * 0.2) / (mins / 90) * 10
        return min(100.0, raw)

    def _gk_score(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        sv   = df["saves"].sum()
        cs   = df["clean_sheets"].sum()
        mins = max(df["minutes"].sum(), 1)
        raw = (sv * 0.5 + cs * 5) / (mins / 90) * 10
        return min(100.0, raw)

    return (
        _attack_score(att),
        _mid_score(mid),
        _def_score(dfe),
        _gk_score(gkp),
    )


def compute_team_squad_strength() -> int:
    """
    Compute team_squad_strength for every national team represented in raw_player_stats.

    Uses the nation_code ↔ national_team mapping populated during CSV loading.
    Also uses raw_national_squads (if populated) to filter to confirmed squad members.

    Returns number of teams processed.
    """
    stats_df = load_player_stats_from_db()
    if stats_df.empty:
        logger.warning(
            "raw_player_stats is empty — cannot compute squad strength. "
            "Ensure players_data_2025_2026.csv is in datasets/."
        )
        return 0

    # Filter to players with a known national team
    stats_df = stats_df[stats_df["national_team"].notna() & (stats_df["national_team"] != "")]

    # Check if we have squad filtering from raw_national_squads
    conn = _open_db()
    try:
        squad_teams: set[str] = set()
        if _table_exists(conn, "raw_national_squads"):
            squad_count = conn.execute(
                "SELECT COUNT(*) FROM raw_national_squads"
            ).fetchone()[0]
            if squad_count > 0:
                rows = conn.execute(
                    "SELECT DISTINCT team FROM raw_national_squads"
                ).fetchall()
                squad_teams = {r[0] for r in rows}
    finally:
        conn.close()

    now     = datetime.now(timezone.utc).isoformat()
    teams   = stats_df["national_team"].unique()
    results = []

    for team in teams:
        team_players = stats_df[stats_df["national_team"] == team].copy()
        if team_players.empty:
            continue

        # Coerce numeric columns
        for col in ["goals", "assists", "shots", "shots_on_target", "saves",
                    "clean_sheets", "minutes", "starts", "tackles_won",
                    "interceptions", "plus_minus", "league_weight", "age"]:
            team_players[col] = pd.to_numeric(team_players.get(col, 0), errors="coerce").fillna(0)

        squad_size     = len(team_players)
        matched        = squad_size    # all players from CSV are "matched" (nation code)
        unmatched      = 0
        match_coverage = 1.0

        avg_age         = float(team_players["age"][team_players["age"] > 0].mean() or 0)
        total_minutes   = float(team_players["minutes"].sum())
        total_goals     = float(team_players["goals"].sum())
        total_assists   = float(team_players["assists"].sum())
        total_g_a       = total_goals + total_assists
        total_shots     = float(team_players["shots"].sum())
        total_sot       = float(team_players["shots_on_target"].sum())
        regular_starters = int((team_players["starts"] >= 10).sum())

        top_league_players = int(
            (team_players["league_weight"] >= 0.90).sum()
        )
        key_player_count = 0
        injured_key_players = 0

        # Positional strength
        att_s, mid_s, def_s, gk_s = _positional_score(team_players)
        positional_avg = (att_s + mid_s + def_s + gk_s) / 4.0

        # League strength (weighted average by minutes)
        total_min = max(total_minutes, 1)
        league_str = float(
            (team_players["league_weight"] * team_players["minutes"]).sum() / total_min
        ) * 100

        # Playing time component (normalize by 26-player × 90-min × 30-game benchmark)
        benchmark_minutes = 26 * 90 * 30  # ~70,200
        pt_score = min(100.0, total_minutes / benchmark_minutes * 100)

        # Production component
        benchmark_ga = 30.0
        production_score = min(100.0, total_g_a / benchmark_ga * 100)

        # Depth: squad_size / 26 typical squad
        depth_score = min(100.0, squad_size / 26.0 * 100)

        # Overall weighted score
        overall = (
            pt_score          * _W_PLAYING_TIME
            + positional_avg  * _W_POSITIONAL
            + league_str      * _W_LEAGUE
            + production_score * _W_PRODUCTION
            + depth_score     * _W_DEPTH
        )
        overall = min(100.0, max(0.0, overall))

        # Data quality: penalise if very few players found
        dq = min(1.0, squad_size / 15.0)   # 15+ players = full quality
        data_quality_score = float(dq * 100)

        results.append((
            team, squad_size, matched, unmatched, match_coverage,
            round(avg_age, 1), round(total_minutes, 0),
            round(total_goals, 0), round(total_assists, 0), round(total_g_a, 0),
            round(total_shots, 0), round(total_sot, 0),
            top_league_players, regular_starters,
            key_player_count, injured_key_players,
            round(att_s, 2), round(mid_s, 2), round(def_s, 2), round(gk_s, 2),
            round(overall, 2), round(data_quality_score, 2), now,
        ))

    conn = _open_db()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO team_squad_strength "
            "(team, squad_size, matched_players, unmatched_players, match_coverage, "
            " avg_age, total_minutes, total_goals, total_assists, total_g_a, "
            " total_shots, total_sot, top_league_players, regular_starters, "
            " key_player_count, injured_key_players, attack_strength, "
            " midfield_strength, defense_strength, goalkeeper_strength, "
            " overall_squad_strength, data_quality_score, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            results,
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "team_squad_strength: computed for %d national teams. "
        "Top 5: %s",
        len(results),
        sorted(
            [(r[0], r[20]) for r in results],
            key=lambda x: x[1], reverse=True,
        )[:5],
    )
    return len(results)


# ---------------------------------------------------------------------------
# Feature output
# ---------------------------------------------------------------------------

_NEUTRAL_STRENGTH = 50.0   # used when data is unavailable


def get_squad_strength_features(
    team_a: str,
    team_b: str,
) -> tuple[float, float, float, bool]:
    """
    Return (strength_a, strength_b, strength_diff, available).

    If squad strength data is unavailable for either team, returns
    (50.0, 50.0, 0.0, False) so the caller can skip adjustment.
    """
    if not _SQUAD_ENABLED:
        return _NEUTRAL_STRENGTH, _NEUTRAL_STRENGTH, 0.0, False

    try:
        conn = _open_db()
        try:
            if not _table_exists(conn, "team_squad_strength"):
                return _NEUTRAL_STRENGTH, _NEUTRAL_STRENGTH, 0.0, False

            def _get(team: str) -> float | None:
                row = conn.execute(
                    "SELECT overall_squad_strength FROM team_squad_strength WHERE team = ?",
                    (team,),
                ).fetchone()
                return float(row[0]) if row else None

            sa = _get(team_a)
            sb = _get(team_b)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Could not fetch squad strength: %s", exc)
        return _NEUTRAL_STRENGTH, _NEUTRAL_STRENGTH, 0.0, False

    if sa is None or sb is None:
        # Log missing teams only for WC 2026 teams
        if sa is None:
            logger.debug("No squad strength data for %s.", team_a)
        if sb is None:
            logger.debug("No squad strength data for %s.", team_b)
        sa = sa or _NEUTRAL_STRENGTH
        sb = sb or _NEUTRAL_STRENGTH
        return sa, sb, sa - sb, False

    return sa, sb, sa - sb, True


def apply_squad_adjustment(
    proba: list[float],   # [p_team_b_wins, p_draw, p_team_a_wins]
    strength_a: float,
    strength_b: float,
    available: bool,
) -> tuple[list[float], bool]:
    """
    Apply conservative squad-strength adjustment to model probabilities.

    proba: [prob_class0 (team_b wins), prob_class1 (draw), prob_class2 (team_a wins)]
    Returns (adjusted_proba, adjustment_applied).
    """
    if not _SQUAD_ENABLED or not available:
        return proba, False

    if len(proba) != 3:
        return proba, False

    diff = (strength_a - strength_b) / 100.0         # normalise to [-1, 1]
    diff = max(-1.0, min(1.0, diff))
    adjustment = diff * _SQUAD_ADJ_WEIGHT * _MAX_SQUAD_ADJ  # max ±2% by default

    p0, p1, p2 = proba
    if diff > 0:          # team_a stronger
        p2 = min(1.0, p2 + abs(adjustment))
        p0 = max(0.0, p0 - abs(adjustment) * 0.7)
        p1 = max(0.0, p1 - abs(adjustment) * 0.3)
    elif diff < 0:        # team_b stronger
        p0 = min(1.0, p0 + abs(adjustment))
        p2 = max(0.0, p2 - abs(adjustment) * 0.7)
        p1 = max(0.0, p1 - abs(adjustment) * 0.3)

    total = p0 + p1 + p2
    if total > 0:
        p0, p1, p2 = p0 / total, p1 / total, p2 / total

    return [p0, p1, p2], True


# ---------------------------------------------------------------------------
# Explanation helper
# ---------------------------------------------------------------------------


def explain_squad_strength(
    team_a: str,
    team_b: str,
) -> str:
    """Return a brief explanation of squad strength comparison for dashboard/predictions."""
    try:
        conn = _open_db()
        try:
            def _row(team: str) -> dict | None:
                r = conn.execute(
                    "SELECT overall_squad_strength, squad_size, matched_players, "
                    "top_league_players, data_quality_score "
                    "FROM team_squad_strength WHERE team = ?",
                    (team,),
                ).fetchone()
                return dict(r) if r else None
            ra = _row(team_a)
            rb = _row(team_b)
        finally:
            conn.close()
    except Exception:
        return ""

    if ra is None and rb is None:
        return (
            "Squad data was unavailable for both teams, "
            "so this prediction relies on FIFA ranking, ELO, and historical patterns."
        )
    if ra is None:
        return (
            f"Squad data unavailable for {team_a}. "
            f"{team_b} squad strength: {rb['overall_squad_strength']:.0f}/100."
        )
    if rb is None:
        return (
            f"Squad data unavailable for {team_b}. "
            f"{team_a} squad strength: {ra['overall_squad_strength']:.0f}/100."
        )

    stronger = team_a if ra["overall_squad_strength"] > rb["overall_squad_strength"] else team_b
    diff     = abs(ra["overall_squad_strength"] - rb["overall_squad_strength"])

    dq_a = ra.get("data_quality_score", 100)
    dq_b = rb.get("data_quality_score", 100)

    coverage_note = ""
    if dq_a < 50 or dq_b < 50:
        low = team_a if dq_a < dq_b else team_b
        coverage_note = (
            f" ⚠️ Low squad data coverage for {low} "
            f"(few players found in top 5 European leagues)."
        )

    return (
        f"{stronger} has a stronger squad profile ({stronger} {ra['overall_squad_strength']:.0f}/100"
        f" vs {team_b if stronger == team_a else team_a} "
        f"{rb['overall_squad_strength']:.0f}/100; diff={diff:.1f})."
        f" Based on {ra['squad_size']} and {rb['squad_size']} players in top-5 European leagues."
        f"{coverage_note}"
    )


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

        # Check if player stats already loaded
        stats_df = load_player_stats_from_db()
        if stats_df.empty:
            logger.info("Loading player stats from CSV ...")
            raw_df = load_player_stats_csv()
            if not raw_df.empty:
                n = save_player_stats_to_db(raw_df)
                logger.info("Loaded %d player stats rows.", n)
            else:
                logger.warning(
                    "No player stats data available. "
                    "Place players_data_2025_2026.csv in datasets/ for squad features."
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

        return True

    except Exception as exc:
        logger.warning(
            "squad_features.setup_squad_features() failed: %s. "
            "Predictions will proceed without squad adjustment.",
            exc,
        )
        return False
