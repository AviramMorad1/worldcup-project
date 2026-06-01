import logging
import os
import sqlite3

import pandas as pd

DB_PATH = "/app/data/worldcup.db"

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"year", "team_a", "team_b", "score_a", "score_b"}

# ELO constants
ELO_DEFAULT = 1500.0
ELO_K = 32.0

# Stage name → encoded integer (0=group … 4=final)
STAGE_MAP = {
    "final": 4,
    "semi-final": 3,
    "semi final": 3,
    "semifinal": 3,
    "third place": 3,
    "third-place": 3,
    "play-off for third place": 3,
    "quarter-final": 2,
    "quarter final": 2,
    "quarterfinal": 2,
    "round of 16": 1,
    "round of sixteen": 1,
    "second round": 1,
    "r16": 1,
}

# Known World Cup hosts by year
WORLD_CUP_HOSTS: dict[int, list[str]] = {
    1930: ["Uruguay"],
    1934: ["Italy"],
    1938: ["France"],
    1950: ["Brazil"],
    1954: ["Switzerland"],
    1958: ["Sweden"],
    1962: ["Chile"],
    1966: ["England"],
    1970: ["Mexico"],
    1974: ["West Germany"],
    1978: ["Argentina"],
    1982: ["Spain"],
    1986: ["Mexico"],
    1990: ["Italy"],
    1994: ["United States"],
    1998: ["France"],
    2002: ["South Korea", "Japan"],
    2006: ["Germany"],
    2010: ["South Africa"],
    2014: ["Brazil"],
    2018: ["Russia"],
    2022: ["Qatar"],
    2026: ["United States", "Canada", "Mexico"],
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def database_exists() -> bool:
    return os.path.exists(DB_PATH)


def table_exists(table_name: str) -> bool:
    if not database_exists():
        return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cur.fetchone() is not None
    except sqlite3.Error as exc:
        logger.warning("Error checking table existence for '%s': %s", table_name, exc)
        return False


def load_table(table_name: str) -> pd.DataFrame:
    if not table_exists(table_name):
        logger.warning("Table '%s' does not exist — returning empty DataFrame.", table_name)
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM %s" % table_name,  # table names cannot be parameterised
                conn,
            )
        logger.info("Loaded %d rows from table '%s'.", len(df), table_name)
        return df
    except Exception as exc:
        logger.warning("Failed to load table '%s': %s", table_name, exc)
        return pd.DataFrame()


def load_raw_matches() -> pd.DataFrame:
    return load_table("raw_matches")


def load_raw_rankings() -> pd.DataFrame:
    return load_table("raw_rankings")


def validate_training_data(matches_df: pd.DataFrame) -> bool:
    if matches_df.empty:
        logger.warning("Training data validation failed: raw_matches is empty.")
        return False

    missing = REQUIRED_COLUMNS - set(matches_df.columns)
    if missing:
        logger.warning(
            "Training data validation failed: missing columns %s in raw_matches.",
            sorted(missing),
        )
        return False

    logger.info(
        "Training data validated: %d rows, all required columns present.", len(matches_df)
    )
    return True


# ---------------------------------------------------------------------------
# ELO computation
# ---------------------------------------------------------------------------

def _create_team_elo_table() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_elo (
                team       TEXT    NOT NULL,
                year       INTEGER NOT NULL,
                elo_rating REAL    NOT NULL,
                PRIMARY KEY (team, year)
            )
        """)
    logger.info("team_elo table ready.")


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_and_store_elo(matches_df: pd.DataFrame) -> dict[str, dict[int, float]]:
    """Compute ELO ratings from match history and persist year snapshots to team_elo.

    Ratings are snapshotted before each tournament starts so they reflect
    accumulated form going into that year's competition.

    Returns dict: {team: {year: elo_before_tournament}}.
    """
    _create_team_elo_table()

    df = matches_df.copy()
    df["score_a"] = pd.to_numeric(df["score_a"], errors="coerce")
    df["score_b"] = pd.to_numeric(df["score_b"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["score_a", "score_b", "year"]).sort_values("year")

    elo: dict[str, float] = {}
    year_elo_snapshots: dict[str, dict[int, float]] = {}

    for year, year_group in df.groupby("year", sort=True):
        year = int(year)
        teams_in_year = set(year_group["team_a"]).union(set(year_group["team_b"]))

        # Snapshot ELO before this tournament begins
        for team in teams_in_year:
            year_elo_snapshots.setdefault(team, {})[year] = elo.get(team, ELO_DEFAULT)

        # Update ELO after each match in this tournament
        for _, row in year_group.iterrows():
            team_a, team_b = row["team_a"], row["team_b"]
            rating_a = elo.get(team_a, ELO_DEFAULT)
            rating_b = elo.get(team_b, ELO_DEFAULT)

            if row["score_a"] > row["score_b"]:
                actual_a = 1.0
            elif row["score_a"] == row["score_b"]:
                actual_a = 0.5
            else:
                actual_a = 0.0

            expected_a = _elo_expected(rating_a, rating_b)
            elo[team_a] = rating_a + ELO_K * (actual_a - expected_a)
            elo[team_b] = rating_b + ELO_K * ((1.0 - actual_a) - (1.0 - expected_a))

    rows = [
        (team, year, rating)
        for team, years_dict in year_elo_snapshots.items()
        for year, rating in years_dict.items()
    ]

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM team_elo")
        conn.executemany(
            "INSERT INTO team_elo (team, year, elo_rating) VALUES (?, ?, ?)",
            rows,
        )
    logger.info("Stored %d ELO snapshots in team_elo table.", len(rows))
    return year_elo_snapshots


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _encode_stage(stage: object) -> int:
    """Map a stage string to an integer (group=0 … final=4)."""
    if not isinstance(stage, str):
        return 0
    key = stage.strip().lower()
    # Exact / substring match (longest key wins via sorted order)
    for pattern in sorted(STAGE_MAP, key=len, reverse=True):
        if pattern in key:
            return STAGE_MAP[pattern]
    return 0  # default: group stage


def _is_host(team: str, year: int) -> int:
    hosts = WORLD_CUP_HOSTS.get(int(year), [])
    return int(any(h.lower() == team.strip().lower() for h in hosts))


def _get_rank(rankings_df: pd.DataFrame, team: str, year: int) -> float:
    """Return the FIFA rank for a team at a given year.

    Falls back to the closest available year if an exact match is not found.
    Returns 100 (mid-table neutral) when no ranking data is available at all.
    """
    if rankings_df.empty:
        return 100.0

    team_lower = team.strip().lower()
    team_rows = rankings_df[rankings_df["team"].str.strip().str.lower() == team_lower]
    if team_rows.empty:
        return 100.0

    exact = team_rows[team_rows["year"].astype(int) == int(year)]
    if not exact.empty:
        return float(exact["rank"].mean())

    # Fall back to year with smallest distance
    diffs = (team_rows["year"].astype(int) - int(year)).abs()
    closest = team_rows.loc[diffs.idxmin()]
    return float(closest["rank"])


def _compute_h2h(
    past_matches: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> tuple[float, float]:
    """Compute head-to-head win rate and average goal difference for team_a vs team_b.

    Uses only matches from before the current tournament year to prevent leakage.
    Returns (win_rate_a, avg_goal_diff_a). Defaults to (0.5, 0.0) when no history exists.
    """
    mask = (
        ((past_matches["team_a"] == team_a) & (past_matches["team_b"] == team_b))
        | ((past_matches["team_a"] == team_b) & (past_matches["team_b"] == team_a))
    )
    h2h = past_matches[mask]
    if h2h.empty:
        return 0.5, 0.0

    wins_a = 0
    total_goal_diff = 0.0
    for _, r in h2h.iterrows():
        if r["team_a"] == team_a:
            diff = float(r["score_a"]) - float(r["score_b"])
            if r["score_a"] > r["score_b"]:
                wins_a += 1
        else:
            diff = float(r["score_b"]) - float(r["score_a"])
            if r["score_b"] > r["score_a"]:
                wins_a += 1
        total_goal_diff += diff

    return wins_a / len(h2h), total_goal_diff / len(h2h)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    matches_df: pd.DataFrame,
    rankings_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full feature matrix from raw match and ranking data.

    Features produced per match:
        rank_diff       — rank_a minus rank_b (lower rank number = better team)
        elo_diff        — elo_a minus elo_b at tournament start
        h2h_win_rate    — historical win rate of team_a against team_b
        h2h_goal_diff   — average goal difference (team_a perspective) in past meetings
        stage_encoded   — group=0, r16=1, quarter=2, semi=3, final=4
        is_host_a       — 1 if team_a is a tournament host
        is_host_b       — 1 if team_b is a tournament host

    Target:
        outcome — 0=team_b wins, 1=draw, 2=team_a wins
    """
    if not validate_training_data(matches_df):
        logger.warning("Cannot build feature matrix — training data is invalid.")
        return pd.DataFrame(), pd.Series(dtype=int)

    if rankings_df is None:
        rankings_df = pd.DataFrame()

    df = matches_df.copy()
    df["score_a"] = pd.to_numeric(df["score_a"], errors="coerce")
    df["score_b"] = pd.to_numeric(df["score_b"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["score_a", "score_b", "year"]).sort_values("year").reset_index(drop=True)

    # Compute ELO ratings and persist to team_elo table
    year_elo = compute_and_store_elo(df)

    feature_rows = []
    for _, row in df.iterrows():
        year = int(row["year"])
        team_a = row["team_a"]
        team_b = row["team_b"]

        # ELO at start of this tournament (before any match in this year)
        elo_a = year_elo.get(team_a, {}).get(year, ELO_DEFAULT)
        elo_b = year_elo.get(team_b, {}).get(year, ELO_DEFAULT)
        elo_diff = elo_a - elo_b

        # FIFA rankings
        rank_a = _get_rank(rankings_df, team_a, year)
        rank_b = _get_rank(rankings_df, team_b, year)
        rank_diff = rank_a - rank_b

        # Head-to-head from matches strictly before this year
        past = df[df["year"] < year]
        h2h_win_rate, h2h_goal_diff = _compute_h2h(past, team_a, team_b)

        # Stage and host
        stage_encoded = _encode_stage(row.get("stage", ""))
        is_host_a = _is_host(team_a, year)
        is_host_b = _is_host(team_b, year)

        # Target label
        if row["score_a"] > row["score_b"]:
            outcome = 2
        elif row["score_a"] == row["score_b"]:
            outcome = 1
        else:
            outcome = 0

        feature_rows.append({
            "year": year,
            "rank_diff": rank_diff,
            "elo_diff": elo_diff,
            "h2h_win_rate": h2h_win_rate,
            "h2h_goal_diff": h2h_goal_diff,
            "stage_encoded": stage_encoded,
            "is_host_a": is_host_a,
            "is_host_b": is_host_b,
            "outcome": outcome,
        })

    result_df = pd.DataFrame(feature_rows)
    feature_cols = [
        "rank_diff", "elo_diff", "h2h_win_rate",
        "h2h_goal_diff", "stage_encoded", "is_host_a", "is_host_b",
    ]
    X = result_df[feature_cols].reset_index(drop=True)
    y = result_df["outcome"].reset_index(drop=True)

    logger.info("Feature matrix built: %d rows, features=%s.", len(X), feature_cols)
    return X, y
