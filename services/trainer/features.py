"""
features.py
-----------
Feature engineering for the World Cup match prediction model.

Features per match:
  rank_diff            — FIFA rank_a minus rank_b at tournament start (year-matched)
  elo_diff             — ELO_a minus ELO_b at tournament start
  h2h_win_rate         — win rate of team_a vs team_b (all prior WC meetings)
  h2h_goal_diff        — avg goal diff (team_a perspective) in prior meetings
  stage_encoded        — group=0, r16=1, quarter=2, semi=3, final=4
  is_host_a            — 1 if team_a is a tournament host
  is_host_b            — 1 if team_b is a tournament host
  current_rank_diff    — most-recent FIFA rank_a minus rank_b (modern strength proxy)
  current_points_diff  — most-recent FIFA points_a minus points_b

Target: outcome — 0=team_b wins, 1=draw, 2=team_a wins

build_feature_matrix returns (X, y, years, sample_weight) where:
  - sample_weight applies recency weighting to down-weight old tournaments
"""

import logging
import math
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
STAGE_MAP: dict[str, int] = {
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
    "quarter-finals": 2,
    "semi-finals": 3,
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
    2002: ["South Korea", "Japan", "Korea Republic"],
    2006: ["Germany"],
    2010: ["South Africa"],
    2014: ["Brazil"],
    2018: ["Russia"],
    2022: ["Qatar"],
    2026: ["United States", "Canada", "Mexico"],
}

# ---------------------------------------------------------------------------
# Recency weighting configuration (read from environment)
# ---------------------------------------------------------------------------

_RECENCY_ENABLED  = os.environ.get("RECENCY_WEIGHTING_ENABLED", "true").lower() == "true"
_RECENCY_DECAY    = float(os.environ.get("RECENCY_DECAY_RATE",  "0.08"))
_MIN_WEIGHT       = float(os.environ.get("MIN_RECENCY_WEIGHT",  "0.35"))
_TARGET_YEAR      = 2026   # anchor year for decay


def _compute_recency_weight(match_year: int) -> float:
    """Exponential decay: w = max(min_weight, exp(-decay * (target - year)))."""
    if not _RECENCY_ENABLED:
        return 1.0
    gap = max(0, _TARGET_YEAR - match_year)
    w = math.exp(-_RECENCY_DECAY * gap)
    return max(_MIN_WEIGHT, w)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def database_exists() -> bool:
    return os.path.exists(DB_PATH)


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def table_exists(table_name: str) -> bool:
    if not database_exists():
        return False
    try:
        conn = _open_db()
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cur.fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("Error checking table existence for '%s': %s", table_name, exc)
        return False


def load_table(table_name: str) -> pd.DataFrame:
    if not table_exists(table_name):
        logger.warning(
            "Table '%s' does not exist — returning empty DataFrame.", table_name
        )
        return pd.DataFrame()
    try:
        conn = _open_db()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM %s" % table_name,  # table names cannot be parameterised
                conn,
            )
        finally:
            conn.close()
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


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_and_store_elo(matches_df: pd.DataFrame) -> dict[str, dict[int, float]]:
    """Compute ELO ratings chronologically and persist year snapshots to team_elo."""
    df = matches_df.copy()
    df["score_a"] = pd.to_numeric(df["score_a"], errors="coerce")
    df["score_b"] = pd.to_numeric(df["score_b"], errors="coerce")
    df["year"]    = pd.to_numeric(df["year"],    errors="coerce")
    df = df.dropna(subset=["score_a", "score_b", "year"]).sort_values("year")

    elo: dict[str, float] = {}
    year_elo_snapshots: dict[str, dict[int, float]] = {}

    for year, year_group in df.groupby("year", sort=True):
        year = int(year)
        teams_in_year = set(year_group["team_a"]).union(set(year_group["team_b"]))

        # Snapshot ELO before this tournament begins
        for team in teams_in_year:
            year_elo_snapshots.setdefault(team, {})[year] = elo.get(team, ELO_DEFAULT)

        # Update ELO after each match
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

    snapshot_rows = [
        (team, year, rating)
        for team, years_dict in year_elo_snapshots.items()
        for year, rating in years_dict.items()
    ]

    conn = _open_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_elo (
                team       TEXT    NOT NULL,
                year       INTEGER NOT NULL,
                elo_rating REAL    NOT NULL,
                PRIMARY KEY (team, year)
            )
        """)
        conn.execute("DELETE FROM team_elo")
        conn.executemany(
            "INSERT INTO team_elo (team, year, elo_rating) VALUES (?, ?, ?)",
            snapshot_rows,
        )
        conn.commit()
    finally:
        conn.close()
    logger.info(
        "team_elo table ready. Stored %d ELO snapshots.", len(snapshot_rows)
    )
    return year_elo_snapshots


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def _encode_stage(stage: object) -> int:
    if not isinstance(stage, str):
        return 0
    key = stage.strip().lower()
    for pattern in sorted(STAGE_MAP, key=len, reverse=True):
        if pattern in key:
            return STAGE_MAP[pattern]
    return 0  # default: group stage


def _is_host(team: str, year: int) -> int:
    hosts = WORLD_CUP_HOSTS.get(int(year), [])
    return int(any(h.lower() == team.strip().lower() for h in hosts))


def _get_rank(
    rankings_df: pd.DataFrame,
    team: str,
    year: int,
    max_year: int | None = None,
) -> float:
    """Return the FIFA rank for a team at a given year (closest available).

    If max_year is provided, only considers rankings from years <= max_year
    (used when building current_rank features to avoid using post-2026 data).
    Falls back to 150.0 when no data is available.
    """
    if rankings_df.empty:
        return 150.0

    team_lower = team.strip().lower()
    team_rows = rankings_df[rankings_df["team"].str.strip().str.lower() == team_lower]
    if team_rows.empty:
        return 150.0

    if max_year is not None:
        team_rows = team_rows[team_rows["year"].astype(int) <= max_year]
    if team_rows.empty:
        return 150.0

    exact = team_rows[team_rows["year"].astype(int) == int(year)]
    if not exact.empty:
        return float(exact["rank"].mean())

    # Fall back to year with smallest distance
    diffs = (team_rows["year"].astype(int) - int(year)).abs()
    closest = team_rows.loc[diffs.idxmin()]
    return float(closest["rank"])


def _get_points(
    rankings_df: pd.DataFrame,
    team: str,
    max_year: int | None = None,
) -> float:
    """Return the most recent FIFA points for a team (0.0 if missing)."""
    if rankings_df.empty or "points" not in rankings_df.columns:
        return 0.0

    team_lower = team.strip().lower()
    team_rows = rankings_df[rankings_df["team"].str.strip().str.lower() == team_lower]
    if team_rows.empty:
        return 0.0

    if max_year is not None:
        team_rows = team_rows[team_rows["year"].astype(int) <= max_year]
    if team_rows.empty:
        return 0.0

    idx = team_rows["year"].idxmax()
    pts = team_rows.loc[idx, "points"]
    try:
        return float(pts)
    except (ValueError, TypeError):
        return 0.0


def _get_current_rank(rankings_df: pd.DataFrame, team: str) -> float:
    """Return the most recent FIFA rank for a team regardless of tournament year."""
    return _get_rank(rankings_df, team, year=9999)


def _get_current_points(rankings_df: pd.DataFrame, team: str) -> float:
    """Return the most recent FIFA points for a team."""
    return _get_points(rankings_df, team)


def _compute_h2h(
    past_matches: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> tuple[float, float]:
    """Win rate and avg goal diff for team_a vs team_b (past matches only)."""
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

FEATURE_COLS = [
    "rank_diff",
    "elo_diff",
    "h2h_win_rate",
    "h2h_goal_diff",
    "stage_encoded",
    "is_host_a",
    "is_host_b",
    "current_rank_diff",
    "current_points_diff",
]

LABEL_NAMES = {0: "B wins", 1: "Draw", 2: "A wins"}


def _label_distribution(y: pd.Series, title: str) -> dict[int, int]:
    counts = y.value_counts().to_dict()
    dist = {int(k): int(v) for k, v in counts.items()}
    total = int(len(y))
    parts = ", ".join(
        f"{LABEL_NAMES.get(k, k)}={v} ({100 * v / total:.1f}%)"
        for k, v in sorted(dist.items())
    )
    logger.info("%s label distribution (n=%d): %s", title, total, parts or "empty")
    return dist


def log_training_bias_diagnostics(
    matches_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    rankings_df: pd.DataFrame | None = None,
) -> None:
    """Log Team A/B asymmetry in raw data and built features (labels: 0=B, 1=Draw, 2=A)."""
    _label_distribution(y, "Feature matrix (before augmentation)")

    if matches_df.empty:
        return

    df = matches_df.copy()
    df["score_a"] = pd.to_numeric(df["score_a"], errors="coerce")
    df["score_b"] = pd.to_numeric(df["score_b"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["score_a", "score_b", "year"])

    a_wins = int((df["score_a"] > df["score_b"]).sum())
    b_wins = int((df["score_a"] < df["score_b"]).sum())
    draws = int((df["score_a"] == df["score_b"]).sum())
    total = len(df)
    logger.info(
        "Raw match outcomes (team_a perspective): A wins=%d (%.1f%%), "
        "B wins=%d (%.1f%%), draws=%d (%.1f%%)",
        a_wins, 100 * a_wins / total if total else 0,
        b_wins, 100 * b_wins / total if total else 0,
        draws, 100 * draws / total if total else 0,
    )

    for year in sorted(df["year"].unique()):
        yr = df[df["year"] == year]
        aw = int((yr["score_a"] > yr["score_b"]).sum())
        logger.info("  Year %d: team_a wins %d/%d matches", int(year), aw, len(yr))

    if rankings_df is not None and not rankings_df.empty and not X.empty:
        better_a = int((X["rank_diff"] < 0).sum())
        better_b = int((X["rank_diff"] > 0).sum())
        tied = int((X["rank_diff"] == 0).sum())
        logger.info(
            "FIFA rank at match year: lower rank (better) is team_a in %d rows, "
            "team_b in %d, tied %d (rank_diff = rank_a - rank_b)",
            better_a, better_b, tied,
        )


def augment_symmetric_matches(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """
    Mirror each row (Team B vs Team A) for training only.

    Swaps directional features and flips labels 0<->2; draw stays 1.
    Mirrored rows keep the same recency sample_weight as the original.
    """
    if X.empty:
        return X, y, sample_weight

    _label_distribution(y, "Before symmetric augmentation")

    X_mirror = X.copy()
    X_mirror["rank_diff"] = -X["rank_diff"]
    X_mirror["elo_diff"] = -X["elo_diff"]
    X_mirror["h2h_win_rate"] = 1.0 - X["h2h_win_rate"]
    X_mirror["h2h_goal_diff"] = -X["h2h_goal_diff"]
    X_mirror["current_rank_diff"] = -X["current_rank_diff"]
    X_mirror["current_points_diff"] = -X["current_points_diff"]
    X_mirror["is_host_a"] = X["is_host_b"].values
    X_mirror["is_host_b"] = X["is_host_a"].values

    y_mirror = y.map({0: 2, 2: 0, 1: 1})

    X_aug = pd.concat([X, X_mirror], ignore_index=True)
    y_aug = pd.concat([y, y_mirror], ignore_index=True)

    if sample_weight is not None:
        sw_aug = pd.concat([sample_weight, sample_weight], ignore_index=True)
    else:
        sw_aug = None

    _label_distribution(y_aug, "After symmetric augmentation")
    logger.info(
        "Symmetric augmentation: %d -> %d training rows (mirrored duplicates).",
        len(X),
        len(X_aug),
    )
    return X_aug, y_aug, sw_aug


def build_feature_matrix(
    matches_df: pd.DataFrame,
    rankings_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build the full feature matrix from raw match and ranking data.

    Returns (X, y, years, sample_weight):
      X              — DataFrame with FEATURE_COLS
      y              — Series with outcome labels (0/1/2)
      years          — Series with tournament year per row
      sample_weight  — Series with recency-based sample weights
    """
    if not validate_training_data(matches_df):
        logger.warning("Cannot build feature matrix — training data is invalid.")
        empty = pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int), pd.Series(dtype=float)
        return empty

    if rankings_df is None:
        rankings_df = pd.DataFrame()

    df = matches_df.copy()
    df["score_a"] = pd.to_numeric(df["score_a"], errors="coerce")
    df["score_b"] = pd.to_numeric(df["score_b"], errors="coerce")
    df["year"]    = pd.to_numeric(df["year"],    errors="coerce")
    df = df.dropna(subset=["score_a", "score_b", "year"]).sort_values("year").reset_index(drop=True)

    years_present = sorted(df["year"].unique().tolist())
    logger.info("Building features from %d matches across years: %s", len(df), years_present)

    # Compute ELO ratings and persist to team_elo
    year_elo = compute_and_store_elo(df)

    feature_rows: list[dict] = []

    for _, row in df.iterrows():
        year   = int(row["year"])
        team_a = row["team_a"]
        team_b = row["team_b"]

        # Year-matched ELO
        elo_a    = year_elo.get(team_a, {}).get(year, ELO_DEFAULT)
        elo_b    = year_elo.get(team_b, {}).get(year, ELO_DEFAULT)
        elo_diff = elo_a - elo_b

        # Year-matched FIFA rankings (for rank_diff)
        rank_a   = _get_rank(rankings_df, team_a, year)
        rank_b   = _get_rank(rankings_df, team_b, year)
        rank_diff = rank_a - rank_b

        # Current (most recent) rankings — modern strength proxy
        curr_rank_a = _get_current_rank(rankings_df, team_a)
        curr_rank_b = _get_current_rank(rankings_df, team_b)
        current_rank_diff = curr_rank_a - curr_rank_b

        curr_pts_a = _get_current_points(rankings_df, team_a)
        curr_pts_b = _get_current_points(rankings_df, team_b)
        current_points_diff = curr_pts_a - curr_pts_b

        # Head-to-head (strictly before this tournament year)
        past = df[df["year"] < year]
        h2h_win_rate, h2h_goal_diff = _compute_h2h(past, team_a, team_b)

        # Stage and host flags
        stage_encoded = _encode_stage(row.get("stage", ""))
        is_host_a     = _is_host(team_a, year)
        is_host_b     = _is_host(team_b, year)

        # Target label
        if row["score_a"] > row["score_b"]:
            outcome = 2
        elif row["score_a"] == row["score_b"]:
            outcome = 1
        else:
            outcome = 0

        # Recency weight
        weight = _compute_recency_weight(year)

        feature_rows.append({
            "year":               year,
            "rank_diff":          rank_diff,
            "elo_diff":           elo_diff,
            "h2h_win_rate":       h2h_win_rate,
            "h2h_goal_diff":      h2h_goal_diff,
            "stage_encoded":      stage_encoded,
            "is_host_a":          is_host_a,
            "is_host_b":          is_host_b,
            "current_rank_diff":  current_rank_diff,
            "current_points_diff": current_points_diff,
            "outcome":            outcome,
            "sample_weight":      weight,
        })

    result_df = pd.DataFrame(feature_rows)
    X = result_df[FEATURE_COLS].reset_index(drop=True)
    y = result_df["outcome"].reset_index(drop=True)
    years = result_df["year"].reset_index(drop=True)
    sample_weight = result_df["sample_weight"].reset_index(drop=True)

    logger.info(
        "Feature matrix built: %d rows, features=%s.",
        len(X), FEATURE_COLS,
    )
    log_training_bias_diagnostics(df, X, y, rankings_df)
    if _RECENCY_ENABLED:
        logger.info(
            "Recency weighting enabled (decay=%.2f, min=%.2f). "
            "2022 weight=1.00, 2018 weight=%.2f, 2010 weight=%.2f.",
            _RECENCY_DECAY, _MIN_WEIGHT,
            _compute_recency_weight(2018),
            _compute_recency_weight(2010),
        )

    return X, y, years, sample_weight
