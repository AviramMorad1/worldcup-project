"""
football_loader.py
------------------
Loads World Cup CSV datasets into the worldcup SQLite database.

Supports multiple historical CSV formats and merges them intelligently.

Tables populated:
  raw_matches   — one row per match from all available World Cup CSVs
  raw_rankings  — one row per country/year ranking snapshot

Historical CSV sources loaded (if present in /app/datasets/):
  1. worldcup_historical.csv  — covers 1930–2014 (World Cup historical data)
  2. worldcup_2018.csv        — 2018 World Cup
  3. matches.csv              — 2022 World Cup (original project file)

Optional:
  current_rankings.csv  — current FIFA rankings (stored with year=2026)
"""

import logging
import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH           = "/app/data/worldcup.db"
DATASETS_DIR      = Path("/app/datasets")

# CSV paths — each is optional; the loader handles missing files gracefully
HISTORICAL_CSV      = DATASETS_DIR / "worldcup_historical.csv"   # 1930–2014
WORLDCUP_2018_CSV   = DATASETS_DIR / "worldcup_2018.csv"          # 2018
MATCHES_CSV         = DATASETS_DIR / "matches.csv"                # 2022
RANKINGS_CSV        = DATASETS_DIR / "rankings.csv"               # historical rankings
CURRENT_RANKINGS_CSV = DATASETS_DIR / "current_rankings.csv"      # latest FIFA rankings

# ---------------------------------------------------------------------------
# Team name normalisation map
# ---------------------------------------------------------------------------

TEAM_NAME_MAP: dict[str, str] = {
    # United States
    "usa":                        "United States",
    "united states of america":   "United States",
    "us":                         "United States",
    # Korea
    "south korea":                "Korea Republic",
    "korea":                      "Korea Republic",
    # Iran
    "iran":                       "IR Iran",
    "islamic republic of iran":   "IR Iran",
    # Ivory Coast
    "ivory coast":                "Côte d'Ivoire",
    "cote d'ivoire":              "Côte d'Ivoire",
    "cote divoire":               "Côte d'Ivoire",
    # Czech
    "czech republic":             "Czechia",
    # Turkey
    "turkey":                     "Türkiye",
    # Netherlands
    "holland":                    "Netherlands",
    # Russia
    "russian federation":         "Russia",
    # Serbia
    "serbia and montenegro":      "Serbia",
    # Congo
    "democratic republic of the congo": "Congo DR",
    "dr congo":                   "Congo DR",
    "congo, democratic republic of the": "Congo DR",
    # Cape Verde
    "cape verde":                 "Cabo Verde",
    # North Macedonia
    "macedonia":                  "North Macedonia",
    "republic of north macedonia": "North Macedonia",
    # Eswatini
    "swaziland":                  "Eswatini",
    # China
    "china pr":                   "China",
    "people's republic of china": "China",
    # Bosnia
    "bosnia-herzegovina":         "Bosnia and Herzegovina",
    # Togo
    "togolaise":                  "Togo",
    # Trinidad
    "trinidad and tobago":        "Trinidad and Tobago",
}


def normalise_team(name: str) -> str:
    """Return the canonical team name for a given input string."""
    if not isinstance(name, str):
        return name
    stripped = name.strip()
    lower = stripped.lower()
    return TEAM_NAME_MAP.get(lower, stripped)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_MATCHES = """
    CREATE TABLE IF NOT EXISTS raw_matches (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        date                  TEXT,
        year                  INTEGER,
        stage                 TEXT,
        team_a                TEXT,
        team_b                TEXT,
        score_a               INTEGER,
        score_b               INTEGER,
        winner                TEXT,
        possession_a          REAL,
        possession_b          REAL,
        attempts_a            INTEGER,
        attempts_b            INTEGER,
        on_target_attempts_a  INTEGER,
        on_target_attempts_b  INTEGER,
        corners_a             INTEGER,
        corners_b             INTEGER,
        yellow_cards_a        INTEGER,
        yellow_cards_b        INTEGER,
        red_cards_a           INTEGER,
        red_cards_b           INTEGER,
        fouls_against_a       INTEGER,
        fouls_against_b       INTEGER,
        passes_a              INTEGER,
        passes_b              INTEGER,
        passes_completed_a    INTEGER,
        passes_completed_b    INTEGER
    )
"""

_DDL_RANKINGS = """
    CREATE TABLE IF NOT EXISTS raw_rankings (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        team   TEXT,
        year   INTEGER,
        rank   INTEGER,
        points REAL
    )
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL_MATCHES)
    conn.execute(_DDL_RANKINGS)
    conn.commit()
    logger.info("football_loader: tables created / verified.")


def _table_has_rows(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
    return row[0] > 0


def _years_in_db(conn: sqlite3.Connection) -> set[int]:
    """Return the set of years currently in raw_matches."""
    rows = conn.execute("SELECT DISTINCT year FROM raw_matches WHERE year IS NOT NULL").fetchall()
    return {int(r[0]) for r in rows}


def _existing_match_keys(conn: sqlite3.Connection) -> set[tuple]:
    """Return a set of (year, team_a, team_b) already in raw_matches."""
    rows = conn.execute(
        "SELECT year, team_a, team_b FROM raw_matches WHERE year IS NOT NULL"
    ).fetchall()
    return {(int(r[0]), str(r[1]), str(r[2])) for r in rows}


def _compute_winner(row: pd.Series) -> str:
    try:
        a, b = int(row["score_a"]), int(row["score_b"])
    except (ValueError, TypeError):
        return "Unknown"
    if a > b:
        return str(row["team_a"])
    if b > a:
        return str(row["team_b"])
    return "Draw"


def _parse_year_from_col(val: object) -> int | None:
    """Extract a 4-digit year from an integer, float, or string."""
    if val is None:
        return None
    try:
        year = int(float(str(val)))
        if 1900 <= year <= 2100:
            return year
    except (ValueError, TypeError):
        pass
    # Try date-string parsing
    if isinstance(val, str):
        try:
            parsed = pd.to_datetime(val, dayfirst=True, errors="coerce")
            if not pd.isna(parsed):
                return int(parsed.year)
        except Exception:
            pass
        m = re.search(r"\b(19|20)\d{2}\b", val)
        if m:
            return int(m.group())
    return None


def _resolve_col(col_lower: dict[str, str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in col_lower:
            return col_lower[cand]
    return None


# ---------------------------------------------------------------------------
# Universal CSV reader — handles both CSV formats
# ---------------------------------------------------------------------------

def _read_wc_csv(path: Path) -> pd.DataFrame | None:
    """
    Read a World Cup CSV file and normalise to the internal schema.

    Handles two main formats:
    - Classic Kaggle WC format: Year, Stage, Home/Away Team Name, Goals
    - Custom format:  team_a, team_b, score_a, score_b, date, stage, year
    Both formats plus many column-name variants are supported.

    Returns a DataFrame with columns:
      date, year, stage, team_a, team_b, score_a, score_b
    """
    if not path.exists():
        logger.info("CSV not found at %s — skipping.", path)
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", path, exc)
        return None

    df.columns = [c.strip() for c in df.columns]
    col_lower: dict[str, str] = {c.lower(): c for c in df.columns}

    # --- year ---
    year_col = _resolve_col(col_lower, ["year", "tournament_year"])
    if year_col:
        df["_year"] = df[year_col].apply(_parse_year_from_col)
    else:
        date_candidates = ["date", "datetime", "match date"]
        date_col = _resolve_col(col_lower, date_candidates)
        if date_col:
            df["_year"] = df[date_col].apply(lambda v: _parse_year_from_col(str(v)))
        else:
            logger.error("CSV %s has no recognisable year/date column.", path)
            return None

    # --- date ---
    date_col = _resolve_col(col_lower, ["date", "datetime", "match date"])
    if date_col:
        df["_date"] = df[date_col].apply(lambda v: str(v).strip() if pd.notna(v) else None)
    else:
        df["_date"] = None

    # --- stage ---
    stage_col = _resolve_col(col_lower, ["stage", "round", "category"])
    df["_stage"] = df[stage_col].apply(lambda v: str(v).strip() if pd.notna(v) else None) \
        if stage_col else None

    # --- team_a ---
    team_a_col = _resolve_col(
        col_lower,
        ["team_a", "team1", "home team name", "home_team_name", "home team", "team a"],
    )
    if not team_a_col:
        logger.error("CSV %s has no recognisable team_a column.", path)
        return None
    df["_team_a"] = df[team_a_col].apply(normalise_team)

    # --- team_b ---
    team_b_col = _resolve_col(
        col_lower,
        ["team_b", "team2", "away team name", "away_team_name", "away team", "team b"],
    )
    if not team_b_col:
        logger.error("CSV %s has no recognisable team_b column.", path)
        return None
    df["_team_b"] = df[team_b_col].apply(normalise_team)

    # --- score_a ---
    score_a_col = _resolve_col(
        col_lower,
        ["score_a", "home team goals", "home_team_goals", "goals team1",
         "number of goals team1", "home_score", "home goals"],
    )
    if not score_a_col:
        logger.error("CSV %s has no recognisable score_a column.", path)
        return None
    df["_score_a"] = pd.to_numeric(df[score_a_col], errors="coerce")

    # --- score_b ---
    score_b_col = _resolve_col(
        col_lower,
        ["score_b", "away team goals", "away_team_goals", "goals team2",
         "number of goals team2", "away_score", "away goals"],
    )
    if not score_b_col:
        logger.error("CSV %s has no recognisable score_b column.", path)
        return None
    df["_score_b"] = pd.to_numeric(df[score_b_col], errors="coerce")

    # --- optional stat columns ---
    stat_map = {
        "possession_a":         ["possession team1"],
        "possession_b":         ["possession team2"],
        "attempts_a":           ["attempts team1"],
        "attempts_b":           ["attempts team2"],
        "on_target_attempts_a": ["on target attempts team1"],
        "on_target_attempts_b": ["on target attempts team2"],
        "corners_a":            ["corners team1"],
        "corners_b":            ["corners team2"],
        "yellow_cards_a":       ["yellow cards team1"],
        "yellow_cards_b":       ["yellow cards team2"],
        "red_cards_a":          ["red cards team1"],
        "red_cards_b":          ["red cards team2"],
        "fouls_against_a":      ["fouls against team1"],
        "fouls_against_b":      ["fouls against team2"],
        "passes_a":             ["passes team1"],
        "passes_b":             ["passes team2"],
        "passes_completed_a":   ["passes completed team1"],
        "passes_completed_b":   ["passes completed team2"],
    }
    for dest, candidates in stat_map.items():
        orig = _resolve_col(col_lower, candidates)
        if orig:
            df[dest] = pd.to_numeric(df[orig], errors="coerce")
        else:
            df[dest] = None

    # --- clean and validate ---
    df = df.dropna(subset=["_score_a", "_score_b", "_year"])
    df = df[df["_team_a"].notna() & df["_team_b"].notna()]
    df = df[df["_team_a"] != ""] # drop placeholder/NaN rows

    df["_score_a"] = df["_score_a"].astype(int)
    df["_score_b"] = df["_score_b"].astype(int)
    df["_year"]    = df["_year"].astype(int)

    df["_winner"] = df.apply(
        lambda row: _compute_winner(
            pd.Series({"team_a": row["_team_a"], "team_b": row["_team_b"],
                       "score_a": row["_score_a"], "score_b": row["_score_b"]})
        ),
        axis=1,
    )

    stat_cols = list(stat_map.keys())
    out = df[["_date", "_year", "_stage", "_team_a", "_team_b",
              "_score_a", "_score_b", "_winner"] + stat_cols].copy()
    out.columns = ["date", "year", "stage", "team_a", "team_b",
                   "score_a", "score_b", "winner"] + stat_cols

    logger.info(
        "CSV %s: %d rows, years: %s",
        path.name,
        len(out),
        sorted(out["year"].unique().tolist()),
    )
    return out


# ---------------------------------------------------------------------------
# Main match loader
# ---------------------------------------------------------------------------

def _load_matches(conn: sqlite3.Connection) -> None:
    """Load all available World Cup match CSVs, skipping years already present."""
    existing_keys = _existing_match_keys(conn)
    existing_years = {k[0] for k in existing_keys}

    frames: list[pd.DataFrame] = []

    # Load from each CSV source
    for csv_path in [HISTORICAL_CSV, WORLDCUP_2018_CSV, MATCHES_CSV]:
        df = _read_wc_csv(csv_path)
        if df is None:
            continue
        # Only keep rows whose year is not already fully represented in DB
        # A year is "fully represented" if at least one row already exists for it
        new_rows = df[~df.apply(
            lambda r: (int(r["year"]), str(r["team_a"]), str(r["team_b"])) in existing_keys,
            axis=1,
        )]
        if new_rows.empty:
            logger.info(
                "CSV %s: all %d rows already in DB — skipping.",
                csv_path.name, len(df),
            )
        else:
            logger.info(
                "CSV %s: loading %d new rows (skipping %d already in DB).",
                csv_path.name, len(new_rows), len(df) - len(new_rows),
            )
            frames.append(new_rows)

    if not frames:
        if not existing_keys:
            logger.warning("No match CSV files found and DB is empty.")
        else:
            logger.info("raw_matches already up to date (%d rows).", len(existing_keys))
        return

    combined = pd.concat(frames, ignore_index=True)
    stat_cols = [
        "possession_a", "possession_b", "attempts_a", "attempts_b",
        "on_target_attempts_a", "on_target_attempts_b", "corners_a", "corners_b",
        "yellow_cards_a", "yellow_cards_b", "red_cards_a", "red_cards_b",
        "fouls_against_a", "fouls_against_b", "passes_a", "passes_b",
        "passes_completed_a", "passes_completed_b",
    ]
    all_fields = ["date", "year", "stage", "team_a", "team_b",
                  "score_a", "score_b", "winner"] + stat_cols

    rows = []
    for _, row in combined[all_fields].iterrows():
        rows.append(tuple(
            None if (pd.isna(row[f]) if f in combined.columns else True) else row[f]
            for f in all_fields
        ))

    conn.executemany(
        f"INSERT INTO raw_matches ({', '.join(all_fields)}) VALUES "
        f"({', '.join('?' * len(all_fields))})",
        rows,
    )
    conn.commit()

    year_counts = combined["year"].value_counts().sort_index()
    logger.info(
        "raw_matches: inserted %d new row(s). Distribution by year:\n%s",
        len(rows),
        "\n".join(f"  {y}: {n}" for y, n in year_counts.items()),
    )

    # Warn if data is still thin after loading
    total = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]
    years_count = conn.execute(
        "SELECT COUNT(DISTINCT year) FROM raw_matches"
    ).fetchone()[0]
    if total < 300:
        logger.warning(
            "raw_matches has only %d rows across %d year(s). "
            "For reliable training, provide worldcup_historical.csv (1930–2014), "
            "worldcup_2018.csv, and matches.csv (2022).",
            total, years_count,
        )
    elif years_count < 5:
        logger.warning(
            "raw_matches covers only %d year(s). "
            "Trainer will use random split instead of chronological split.",
            years_count,
        )
    else:
        logger.info(
            "raw_matches: %d rows across %d tournament year(s). "
            "Chronological train/test split will be used.",
            total, years_count,
        )


# ---------------------------------------------------------------------------
# Rankings loader
# ---------------------------------------------------------------------------

def _load_rankings(conn: sqlite3.Connection) -> None:
    """Load historical rankings CSV if present."""
    if _table_has_rows(conn, "raw_rankings"):
        logger.info("raw_rankings already has data — checking for current rankings only.")
        _load_current_rankings(conn)
        return

    csv_path = RANKINGS_CSV
    if not csv_path.exists():
        logger.info(
            "Rankings CSV not found at %s — skipping (not required).", csv_path
        )
        _load_current_rankings(conn)
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read rankings CSV: %s — skipping.", exc)
        return

    df.columns = [c.strip() for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}

    required = {
        "rank_date": ["rank_date", "date"],
        "team":      ["country_full", "country", "team", "name"],
        "rank":      ["rank"],
        "points":    ["total_points", "points", "total points"],
    }

    rename: dict[str, str] = {}
    for dest, candidates in required.items():
        orig = _resolve_col(col_lower, candidates)
        if orig:
            rename[orig] = dest
        else:
            logger.warning(
                "Rankings CSV: could not find column for '%s' — skipping rankings load.", dest
            )
            return

    df = df.rename(columns=rename)[list(required.keys())].copy()
    df["year"]   = pd.to_datetime(df["rank_date"], errors="coerce").dt.year
    df["rank"]   = pd.to_numeric(df["rank"],   errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["year", "rank", "points"])
    if before > len(df):
        logger.warning(
            "Dropped %d ranking row(s) with unparseable fields.", before - len(df)
        )

    if df.empty:
        logger.info("raw_rankings: no valid rows to insert after cleaning.")
        _load_current_rankings(conn)
        return

    df["year"] = df["year"].astype(int)
    df["rank"] = df["rank"].astype(int)
    df["team"] = df["team"].apply(normalise_team)

    rows = list(df[["team", "year", "rank", "points"]].itertuples(index=False, name=None))
    conn.executemany(
        "INSERT INTO raw_rankings (team, year, rank, points) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(
        "raw_rankings: inserted %d historical row(s) (years %d–%d).",
        len(rows), df["year"].min(), df["year"].max(),
    )
    _load_current_rankings(conn)


def _load_current_rankings(conn: sqlite3.Connection) -> None:
    """
    Load current_rankings.csv into raw_rankings with year=2026.

    Expected CSV columns (flexible): team/country, rank, points, [confederation], [year/date]
    If the file is absent, logs a warning and continues.
    """
    csv_path = CURRENT_RANKINGS_CSV
    if not csv_path.exists():
        logger.warning(
            "Current FIFA rankings file not found at %s; "
            "2026 predictions will use fallback historical rankings. "
            "To improve prediction quality, place a current_rankings.csv "
            "(columns: team, rank, points) in the datasets/ folder.",
            csv_path,
        )
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read current rankings CSV: %s — skipping.", exc)
        return

    df.columns = [c.strip() for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}

    team_col = _resolve_col(
        col_lower, ["team", "country", "country_full", "name", "nation"]
    )
    rank_col  = _resolve_col(col_lower, ["rank", "position", "pos"])
    pts_col   = _resolve_col(col_lower, ["points", "total_points", "pts", "total points"])

    if not team_col or not rank_col:
        logger.warning(
            "current_rankings.csv: cannot find team (%s) or rank (%s) column.",
            team_col, rank_col,
        )
        return

    df["_team"]   = df[team_col].apply(normalise_team)
    df["_rank"]   = pd.to_numeric(df[rank_col],  errors="coerce")
    df["_points"] = pd.to_numeric(df[pts_col], errors="coerce") if pts_col else 0.0

    # Determine year: use 2026 if CSV has no year column, else use provided year
    year_col = _resolve_col(col_lower, ["year"])
    if year_col:
        df["_year"] = pd.to_numeric(df[year_col], errors="coerce").fillna(2026).astype(int)
    else:
        df["_year"] = 2026

    df = df.dropna(subset=["_team", "_rank"])
    df["_rank"] = df["_rank"].astype(int)

    # Remove any existing rows for the same year to allow refresh
    years = df["_year"].unique().tolist()
    for yr in years:
        conn.execute(
            "DELETE FROM raw_rankings WHERE year = ?", (int(yr),)
        )

    rows = list(
        df[["_team", "_year", "_rank", "_points"]]
        .itertuples(index=False, name=None)
    )
    conn.executemany(
        "INSERT INTO raw_rankings (team, year, rank, points) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(
        "raw_rankings: loaded %d current ranking row(s) for year(s) %s from %s.",
        len(rows), sorted(years), csv_path.name,
    )


# ---------------------------------------------------------------------------
# Player stats loader (optional)
# ---------------------------------------------------------------------------

WC_PLAYERS_CSV_PATHS = [
    DATASETS_DIR / "squads" / "wc_players_with_stats.csv",
    DATASETS_DIR / "wc_players_with_stats.csv",
    DATASETS_DIR / "wc_players_with_stats(1).csv",
    DATASETS_DIR / "players_data_2025_2026.csv",
    DATASETS_DIR / "players_data-2025_2026.csv",
]

LEAGUE_STRENGTHS_CSV = DATASETS_DIR / "league_strengths.csv"

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

# Nation code → national team (WC 2026 + common FBref codes)
_NATION_CODE_MAP: dict[str, str] = {
    "ARG": "Argentina", "BRA": "Brazil", "ENG": "England", "FRA": "France",
    "GER": "Germany", "DEU": "Germany", "ESP": "Spain", "POR": "Portugal",
    "GHA": "Ghana", "USA": "United States", "MAR": "Morocco", "SEN": "Senegal",
    "URU": "Uruguay", "COL": "Colombia", "JPN": "Japan", "KOR": "Korea Republic",
    "MEX": "Mexico", "BEL": "Belgium", "SUI": "Switzerland", "CRO": "Croatia",
    "ITA": "Italy", "AUS": "Australia", "QAT": "Qatar", "KSA": "Saudi Arabia",
    "EGY": "Egypt", "IRN": "IR Iran", "TUN": "Tunisia", "ALG": "Algeria",
    "ECU": "Ecuador", "PAR": "Paraguay", "PRY": "Paraguay", "CHI": "Chile",
    "CRC": "Costa Rica", "NZL": "New Zealand", "CAN": "Canada", "PAN": "Panama",
    "UKR": "Ukraine", "AUT": "Austria", "SCO": "Scotland", "CZE": "Czechia",
    "NOR": "Norway", "SWE": "Sweden", "DEN": "Denmark", "SRB": "Serbia",
    "TUR": "Türkiye", "NED": "Netherlands", "CIV": "Côte d'Ivoire",
    "CMR": "Cameroon", "NGA": "Nigeria", "RSA": "South Africa", "ZAF": "South Africa",
    "UZB": "Uzbekistan", "JOR": "Jordan", "IRQ": "Iraq", "CPV": "Cabo Verde",
    "COD": "Congo DR", "CUW": "Curaçao", "HAI": "Haiti", "MEX": "Mexico",
    "BIH": "Bosnia and Herzegovina",
}

_TEAM_TO_CODE: dict[str, str] = {v: k for k, v in _NATION_CODE_MAP.items()}

# Squad CSV names → names used in 2026 predictions / rankings
_CANONICAL_TEAM_NAME: dict[str, str] = {
    "Czech Republic": "Czechia",
    "South Korea": "Korea Republic",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Turkey": "Türkiye",
    "DR Congo": "Congo DR",
    "Iran": "IR Iran",
}


def _canonical_team(team: str) -> str:
    s = team.strip()
    return _CANONICAL_TEAM_NAME.get(s, s)


_LEAGUE_WEIGHT_CACHE: dict[str, float] | None = None


def _norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_s = nfkd.encode("ascii", "ignore").decode("ascii").lower()
    clean = re.sub(r"[^a-z0-9\s]", " ", ascii_s)
    return re.sub(r"\s+", " ", clean).strip()


def _parse_nation_code(nation_str: str) -> tuple[str, str]:
    """Parse FBref Nation field like 'ar ARG' → ('ARG', 'Argentina')."""
    if not isinstance(nation_str, str) or not nation_str.strip():
        return ("", "")
    parts = nation_str.strip().split()
    code = parts[-1].upper() if len(parts) >= 2 else parts[0].upper()
    team = _NATION_CODE_MAP.get(code, "")
    return (code, team)


def _load_league_weight_map() -> dict[str, float]:
    global _LEAGUE_WEIGHT_CACHE
    if _LEAGUE_WEIGHT_CACHE is not None:
        return _LEAGUE_WEIGHT_CACHE

    weights: dict[str, float] = {
        "eng premier league": 1.0, "es la liga": 0.95, "de bundesliga": 0.92,
        "it serie a": 0.92, "fr ligue 1": 0.88, "pt primeira liga": 0.78,
        "mls": 0.65, "saudi pro league": 0.62,
    }
    if LEAGUE_STRENGTHS_CSV.exists():
        try:
            ldf = pd.read_csv(LEAGUE_STRENGTHS_CSV)
            for _, lr in ldf.iterrows():
                norm = str(lr.get("normalized_league_name", "")).strip().lower()
                if not norm:
                    norm = _norm_name(str(lr.get("league_name", "")))
                try:
                    weights[norm] = float(lr["strength_weight"])
                except (KeyError, TypeError, ValueError):
                    pass
            logger.info("Loaded %d league weights from league_strengths.csv.", len(ldf))
        except Exception as exc:
            logger.warning("Could not read league_strengths.csv: %s", exc)
    else:
        logger.info(
            "league_strengths.csv not found — using inline fallback league weights."
        )

    _LEAGUE_WEIGHT_CACHE = weights
    return weights


def _league_weight_for(league: str) -> tuple[str, float]:
    weights = _load_league_weight_map()
    n = _norm_name(league)
    if not n:
        return ("unknown", 0.45)
    if n in weights:
        return (n, weights[n])
    for key, w in weights.items():
        if key in n or n in key:
            return (key, w)
    return (n, 0.50)


def _migrate_player_stats_schema(conn: sqlite3.Connection) -> None:
    """Add any missing columns to raw_player_stats (safe migration)."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(raw_player_stats)").fetchall()}
    migrations = [
        ("normalized_league_name", "TEXT"),
        ("goals_assists", "REAL"),
        ("caps", "REAL"),
        ("goals_intl", "REAL"),
        ("is_captain", "INTEGER"),
    ]
    for col, ctype in migrations:
        if col not in existing:
            conn.execute(f"ALTER TABLE raw_player_stats ADD COLUMN {col} {ctype}")
            conn.commit()
            logger.info("raw_player_stats: added column '%s'.", col)


def _find_players_csv() -> Path | None:
    for path in WC_PLAYERS_CSV_PATHS:
        if path.exists():
            return path
    return None


def _float_val(row: pd.Series, *cols: str) -> float | None:
    for col in cols:
        if col not in row.index:
            continue
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        try:
            s = str(v).replace(",", "").strip()
            if s in ("", "nan"):
                continue
            return float(s)
        except (TypeError, ValueError):
            continue
    return None


def _rows_from_wc_csv(df: pd.DataFrame, source_name: str, now: str) -> list[tuple]:
    """Parse wc_players_with_stats.csv format."""
    rows = []
    for idx, row in df.iterrows():
        player = str(row.get("player_name", row.get("Player", ""))).strip()
        if not player:
            continue
        team = str(row.get("team", row.get("national_team", ""))).strip()
        if not team:
            _, team = _parse_nation_code(str(row.get("Nation", "")))
        team = _canonical_team(team)
        code = _TEAM_TO_CODE.get(team, "")

        league = str(
            row.get("stats_league", row.get("league", row.get("Comp", "")))
        ).strip()
        norm_league, lw = _league_weight_for(league)

        goals = _float_val(row, "goals_stats", "goals", "Gls")
        ast   = _float_val(row, "assists", "Ast")
        ga    = _float_val(row, "G+A", "goals_assists")
        if ga is None and goals is not None and ast is not None:
            ga = goals + ast

        cap = row.get("is_captain", 0)
        try:
            is_cap = int(float(cap)) if cap not in ("", None) else 0
        except (TypeError, ValueError):
            is_cap = 0

        pid = f"wc_{_norm_name(team)}_{_norm_name(player)}_{idx}"
        rows.append((
            pid, player, _norm_name(player), team, code,
            str(row.get("stats_position", row.get("position_wc", row.get("Pos", "")))).strip(),
            str(row.get("club_wc", row.get("club", row.get("Squad", "")))).strip(),
            league, norm_league, lw,
            _float_val(row, "age_wc", "age", "Age"),
            _float_val(row, "caps"),
            _float_val(row, "goals_intl"),
            is_cap,
            _float_val(row, "matches_played", "appearances", "MP"),
            _float_val(row, "starts", "Starts"),
            _float_val(row, "minutes", "Min"),
            _float_val(row, "90s", "nineties"),
            goals, ast, ga,
            _float_val(row, "shots", "Sh"),
            _float_val(row, "shots_on_target", "SoT"),
            _float_val(row, "saves", "Saves"),
            _float_val(row, "clean_sheets", "CS"),
            _float_val(row, "yellow_cards", "CrdY"),
            _float_val(row, "red_cards", "CrdR"),
            _float_val(row, "crosses", "Crs"),
            _float_val(row, "interceptions", "Int"),
            _float_val(row, "tackles_won", "TklW"),
            _float_val(row, "plus_minus", "+/-"),
            _float_val(row, "points_per_match", "PPM"),
            source_name, now,
        ))
    return rows


def _rows_from_fbref_csv(df: pd.DataFrame, source_name: str, now: str) -> list[tuple]:
    """Parse classic FBref players_data CSV format."""
    rows = []
    for idx, row in df.iterrows():
        player = str(row.get("Player", "")).strip()
        if not player:
            continue
        code, team = _parse_nation_code(str(row.get("Nation", "")))
        league = str(row.get("Comp", row.get("league", ""))).strip()
        norm_league, lw = _league_weight_for(league)
        goals = _float_val(row, "Gls", "goals")
        ast   = _float_val(row, "Ast", "assists")
        ga    = _float_val(row, "G+A")
        if ga is None and goals is not None and ast is not None:
            ga = goals + ast
        rows.append((
            f"ps_{idx}", player, _norm_name(player), team, code,
            str(row.get("Pos", "")).strip(),
            str(row.get("Squad", "")).strip(),
            league, norm_league, lw,
            _float_val(row, "Age"),
            None, None, 0,
            _float_val(row, "MP"), _float_val(row, "Starts"), _float_val(row, "Min"),
            _float_val(row, "90s"), goals, ast, ga,
            _float_val(row, "Sh"), _float_val(row, "SoT"),
            _float_val(row, "Saves"), _float_val(row, "CS"),
            _float_val(row, "CrdY"), _float_val(row, "CrdR"),
            _float_val(row, "Crs"), _float_val(row, "Int"), _float_val(row, "TklW"),
            _float_val(row, "+/-"), _float_val(row, "PPM"),
            source_name, now,
        ))
    return rows


def _load_player_stats(conn: sqlite3.Connection) -> None:
    """
    Load WC merged player stats CSV into raw_player_stats.
    Prefers datasets/squads/wc_players_with_stats.csv; reloads when that file exists.
    """
    import datetime as _dt

    csv_path = _find_players_csv()
    if csv_path is None:
        logger.info("No player stats CSV found — skipping raw_player_stats load.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read player stats CSV %s: %s", csv_path, exc)
        return

    df.columns = [c.strip() for c in df.columns]
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()

    is_wc = "player_name" in df.columns or "stats_league" in df.columns
    if is_wc:
        rows = _rows_from_wc_csv(df, csv_path.name, now)
    else:
        rows = _rows_from_fbref_csv(df, csv_path.name, now)

    if not rows:
        logger.warning("Player stats CSV parsed to 0 rows.")
        return

    _migrate_player_stats_schema(conn)

    # Reload WC/merged file each cycle so updated stats are reflected
    if is_wc:
        conn.execute("DELETE FROM raw_player_stats")
        conn.commit()

    placeholders = ", ".join("?" * len(_PLAYER_STATS_COLS))
    conn.executemany(
        f"INSERT OR REPLACE INTO raw_player_stats ({', '.join(_PLAYER_STATS_COLS)}) "
        f"VALUES ({placeholders})",
        rows,
    )
    conn.commit()

    teams = {r[3] for r in rows if r[3]}
    leagues = {r[7] for r in rows if r[7]}
    unknown_codes = [r[4] for r in rows if r[4] == "" and r[3]]
    logger.info(
        "raw_player_stats: loaded %d rows from %s — %d national teams, %d leagues.",
        len(rows), csv_path.name, len(teams), len(leagues),
    )
    if unknown_codes:
        logger.warning(
            "%d rows have no nation_code mapping (team name still stored).",
            len(unknown_codes),
        )
    per_team = {}
    for r in rows:
        per_team[r[3]] = per_team.get(r[3], 0) + 1
    top = sorted(per_team.items(), key=lambda x: -x[1])[:5]
    bot = sorted(per_team.items(), key=lambda x: x[1])[:5]
    logger.info("Players per team (top): %s", top)
    logger.info("Players per team (bottom): %s", bot)


def _load_manual_squads(conn: sqlite3.Connection) -> None:
    """Load datasets/national_team_squads.csv into raw_national_squads if present."""
    csv_path = DATASETS_DIR / "national_team_squads.csv"
    if not csv_path.exists():
        return

    existing = conn.execute("SELECT COUNT(*) FROM raw_national_squads").fetchone()[0]
    if existing > 0:
        logger.info("raw_national_squads already has %d rows — skipping.", existing)
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read manual squads CSV: %s", exc)
        return

    df.columns = [c.strip().lower() for c in df.columns]
    if not {"team", "player_name"}.issubset(df.columns):
        logger.warning("national_team_squads.csv missing team/player_name columns.")
        return

    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    rows = []
    for i, r in df.iterrows():
        rows.append((
            f"sq_manual_{i}",
            str(r.get("team", "")).strip(),
            str(r.get("player_name", "")).strip(),
            str(r.get("position",    "")).strip(),
            float(r["age"]) if str(r.get("age", "")) not in ("", "nan") else None,
            str(r.get("club",   "")).strip(),
            str(r.get("league", "")).strip(),
            "manual",
            int(r["is_key_player"]) if "is_key_player" in r else 0,
            int(r["is_injured"])    if "is_injured"    in r else 0,
            now,
        ))

    conn.executemany(
        "INSERT OR IGNORE INTO raw_national_squads "
        "(id, team, player_name, position, age, club, league, source, "
        " is_key_player, is_injured, collected_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info("raw_national_squads: inserted %d rows from manual CSV.", len(rows))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_football_data() -> None:
    """Load all available match, ranking, and player stats CSVs into the database.

    Safe to call on every collector cycle — only inserts data not already present.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            _create_tables(conn)
            # Squad tables
            conn.execute(_DDL_PLAYER_STATS)
            conn.execute(_DDL_NATIONAL_SQUADS)
            conn.commit()
            _load_matches(conn)
            _load_rankings(conn)
            _load_player_stats(conn)
            _load_manual_squads(conn)
    except sqlite3.Error as exc:
        logger.error("football_loader: database error: %s", exc)
