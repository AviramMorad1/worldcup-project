"""
football_loader.py
------------------
Loads World Cup CSV datasets into the worldcup SQLite database.

Tables populated:
  raw_matches   — one row per match (2022 data, rich match stats)
  raw_rankings  — one row per country/year ranking snapshot (optional)
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH      = "/app/data/worldcup.db"
MATCHES_CSV  = "/app/datasets/matches.csv"
RANKINGS_CSV = "/app/datasets/rankings.csv"

# ---------------------------------------------------------------------------
# Column maps
# ---------------------------------------------------------------------------

# Required columns: csv_column_name (lowercase) -> dest field
_REQUIRED_COL_MAP: dict[str, list[str]] = {
    "date":    ["date"],
    "stage":   ["category", "stage", "round"],
    "team_a":  ["team1", "home team name", "home_team_name", "home team"],
    "team_b":  ["team2", "away team name", "away_team_name", "away team"],
    "score_a": ["number of goals team1", "home team goals", "home_team_goals",
                "goals team1", "score_a"],
    "score_b": ["number of goals team2", "away team goals", "away_team_goals",
                "goals team2", "score_b"],
}

# Optional stat columns: dest field -> candidate csv column names (lowercase)
_OPTIONAL_COL_MAP: dict[str, list[str]] = {
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

# All optional stat dest names in insertion order
_OPTIONAL_FIELDS = list(_OPTIONAL_COL_MAP.keys())

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


def _resolve_col(col_map_lower: dict[str, str], candidates: list[str]) -> str | None:
    """Return the first candidate that exists in the lowercased column map."""
    for cand in candidates:
        if cand in col_map_lower:
            return col_map_lower[cand]
    return None


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


def _parse_year(date_str: str, fallback: int = 2022) -> int:
    """Extract a 4-digit year from a date string; return fallback on failure."""
    if not isinstance(date_str, str):
        return fallback
    # Try pandas — handles "20 NOV 2022", "2022-11-20", "20/11/2022", etc.
    try:
        parsed = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        if not pd.isna(parsed):
            return int(parsed.year)
    except Exception:
        pass
    # Fallback: scan for a 4-digit token that looks like a year
    import re
    m = re.search(r"\b(19|20)\d{2}\b", date_str)
    if m:
        return int(m.group())
    return fallback


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_matches(conn: sqlite3.Connection) -> None:
    if _table_has_rows(conn, "raw_matches"):
        logger.info("raw_matches already has data — skipping CSV load.")
        return

    csv_path = Path(MATCHES_CSV)
    if not csv_path.exists():
        logger.warning("Matches CSV not found at %s — skipping.", MATCHES_CSV)
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.error("Failed to read matches CSV: %s", exc)
        return

    # Build a lowercase -> original-name lookup for flexible matching
    df.columns = [c.strip() for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}

    # --- Resolve required columns ---
    rename: dict[str, str] = {}
    missing: list[str] = []
    for dest, candidates in _REQUIRED_COL_MAP.items():
        orig = _resolve_col(col_lower, candidates)
        if orig:
            rename[orig] = dest
        else:
            missing.append(dest)

    if missing:
        logger.error(
            "Matches CSV is missing required columns for: %s. "
            "Columns present: %s",
            missing, list(df.columns),
        )
        return

    df = df.rename(columns=rename)

    # --- Resolve optional stat columns (no failure if absent) ---
    optional_present: dict[str, str] = {}  # dest -> original col name
    for dest, candidates in _OPTIONAL_COL_MAP.items():
        orig = _resolve_col(col_lower, candidates)
        if orig and orig not in rename:          # not already renamed
            optional_present[dest] = orig
        elif orig and rename.get(orig) == dest:  # already renamed above
            optional_present[dest] = dest

    # Rename optional cols that haven't been renamed yet
    opt_rename = {v: k for k, v in optional_present.items()
                  if v not in df.columns or v != k}
    if opt_rename:
        df = df.rename(columns=opt_rename)

    # --- Coerce numeric fields ---
    for col in ["score_a", "score_b"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["score_a", "score_b"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d match row(s) with unparseable scores.", dropped)

    df["score_a"] = df["score_a"].astype(int)
    df["score_b"] = df["score_b"].astype(int)

    for field in _OPTIONAL_FIELDS:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")
        else:
            df[field] = None   # column absent in CSV → NULL in DB

    # --- Derive year and winner ---
    df["year"]   = df["date"].apply(lambda d: _parse_year(str(d)))
    df["winner"] = df.apply(_compute_winner, axis=1)

    # --- Insert ---
    all_fields = [
        "date", "year", "stage", "team_a", "team_b",
        "score_a", "score_b", "winner",
    ] + _OPTIONAL_FIELDS

    placeholders = ", ".join("?" * len(all_fields))
    col_names    = ", ".join(all_fields)

    rows = [
        tuple(
            None if (pd.isna(row[f]) if f in df.columns else True) else row[f]
            for f in all_fields
        )
        for _, row in df[all_fields].iterrows()
    ]

    conn.executemany(
        f"INSERT INTO raw_matches ({col_names}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()

    years = df["year"].dropna()
    logger.info(
        "raw_matches: inserted %d row(s) (year(s): %s).",
        len(rows),
        f"{int(years.min())}–{int(years.max())}" if not years.empty else "unknown",
    )


def _load_rankings(conn: sqlite3.Connection) -> None:
    if _table_has_rows(conn, "raw_rankings"):
        logger.info("raw_rankings already has data — skipping CSV load.")
        return

    csv_path = Path(RANKINGS_CSV)
    if not csv_path.exists():
        logger.info("Rankings CSV not found at %s — skipping (not required).", RANKINGS_CSV)
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read rankings CSV: %s — skipping.", exc)
        return

    df.columns = [c.strip() for c in df.columns]
    col_lower  = {c.lower(): c for c in df.columns}

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

    # Coerce before dropna so invalid text → NaN → dropped safely
    df["year"]   = pd.to_datetime(df["rank_date"], errors="coerce").dt.year
    df["rank"]   = pd.to_numeric(df["rank"],   errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["year", "rank", "points"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d ranking row(s) with unparseable fields.", dropped)

    if df.empty:
        logger.info("raw_rankings: no valid rows to insert after cleaning — skipping.")
        return

    df["year"] = df["year"].astype(int)
    df["rank"] = df["rank"].astype(int)

    rows = list(df[["team", "year", "rank", "points"]].itertuples(index=False, name=None))
    conn.executemany(
        "INSERT INTO raw_rankings (team, year, rank, points) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(
        "raw_rankings: inserted %d row(s) (years %d–%d).",
        len(rows), df["year"].min(), df["year"].max(),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_football_data() -> None:
    """Load matches and rankings CSVs into the database.

    Safe to call on every cycle — skips any table that already has rows.
    rankings.csv is optional; its absence does not affect raw_matches loading.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            _create_tables(conn)
            _load_matches(conn)
            _load_rankings(conn)
    except sqlite3.Error as exc:
        logger.error("football_loader: database error: %s", exc)