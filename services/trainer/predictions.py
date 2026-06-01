"""
predictions.py
--------------
Generates FIFA World Cup 2026 group stage predictions using the trained model.

All 72 group stage matchups (12 groups × 6 matches each) are hardcoded from
the official FIFA 2026 draw. Predictions are stored in the match_predictions
table with confidence scores.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import pandas as pd

DB_PATH = "/app/data/worldcup.db"
MODEL_PATH = Path("/app/data/models/model.pkl")

logger = logging.getLogger(__name__)

ELO_DEFAULT = 1500.0

FEATURE_COLS = [
    "rank_diff", "elo_diff", "h2h_win_rate",
    "h2h_goal_diff", "stage_encoded", "is_host_a", "is_host_b",
]

# ---------------------------------------------------------------------------
# Official FIFA World Cup 2026 group draw
# Source: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026
# 12 groups × 4 teams = 48 teams → 12 × C(4,2) = 72 group stage matches
# ---------------------------------------------------------------------------
GROUPS_2026: dict[str, list[str]] = {
    "A": ["Mexico", "Korea Republic", "South Africa", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Haiti", "Brazil", "Scotland", "Morocco"],
    "D": ["United States", "Australia", "Paraguay", "Türkiye"],
    "E": ["Côte d'Ivoire", "Germany", "Ecuador", "Curaçao"],
    "F": ["Netherlands", "Sweden", "Japan", "Tunisia"],
    "G": ["IR Iran", "Belgium", "New Zealand", "Egypt"],
    "H": ["Saudi Arabia", "Spain", "Uruguay", "Cabo Verde"],
    "I": ["France", "Iraq", "Senegal", "Norway"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Congo DR", "Colombia"],
    "L": ["Ghana", "England", "Panama", "Croatia"],
}

# 2026 is co-hosted by USA, Canada, and Mexico
HOSTS_2026: set[str] = {"United States", "Canada", "Mexico"}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def create_match_predictions_table() -> None:
    conn = _open_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS match_predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_year  INTEGER,
                group_name       TEXT,
                team_a           TEXT,
                team_b           TEXT,
                stage            TEXT,
                predicted_winner TEXT,
                confidence       REAL,
                run_at           TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()
    logger.info("match_predictions table ready.")


# ---------------------------------------------------------------------------
# Data loaders for prediction features
# ---------------------------------------------------------------------------

def _load_latest_elo() -> dict[str, float]:
    """Return the most recent ELO rating per team from the team_elo table."""
    conn = _open_db()
    try:
        df = pd.read_sql_query("SELECT team, year, elo_rating FROM team_elo", conn)
    except Exception as exc:
        logger.warning("Could not load team_elo: %s", exc)
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return {}

    # One row per team: take the snapshot from the most recent year
    idx = df.groupby("team")["year"].idxmax()
    return df.loc[idx].set_index("team")["elo_rating"].to_dict()


def _load_latest_rankings() -> pd.DataFrame:
    """Return all raw_rankings rows for rank lookups."""
    conn = _open_db()
    try:
        df = pd.read_sql_query("SELECT team, year, rank FROM raw_rankings", conn)
    except Exception as exc:
        logger.warning("Could not load raw_rankings: %s", exc)
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def _load_all_matches() -> pd.DataFrame:
    """Return historical match results for H2H calculation."""
    conn = _open_db()
    try:
        df = pd.read_sql_query(
            "SELECT team_a, team_b, score_a, score_b FROM raw_matches", conn
        )
    except Exception as exc:
        logger.warning("Could not load raw_matches: %s", exc)
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------------------
# Feature helpers (mirrors features.py but for inference, not training)
# ---------------------------------------------------------------------------

def _get_rank(rankings_df: pd.DataFrame, team: str) -> float:
    """Return the most recent FIFA rank for a team; fall back to 100."""
    if rankings_df.empty:
        return 100.0
    team_lower = team.strip().lower()
    rows = rankings_df[rankings_df["team"].str.strip().str.lower() == team_lower]
    if rows.empty:
        return 100.0
    return float(rows.loc[rows["year"].idxmax(), "rank"])


def _compute_h2h(
    matches_df: pd.DataFrame, team_a: str, team_b: str
) -> tuple[float, float]:
    """Return (win_rate_a, avg_goal_diff_a) from all historical meetings."""
    if matches_df.empty:
        return 0.5, 0.0

    mask = (
        ((matches_df["team_a"] == team_a) & (matches_df["team_b"] == team_b))
        | ((matches_df["team_a"] == team_b) & (matches_df["team_b"] == team_a))
    )
    h2h = matches_df[mask]
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


def _build_row(
    team_a: str,
    team_b: str,
    elo_map: dict[str, float],
    rankings_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> dict:
    elo_a = elo_map.get(team_a, ELO_DEFAULT)
    elo_b = elo_map.get(team_b, ELO_DEFAULT)

    rank_a = _get_rank(rankings_df, team_a)
    rank_b = _get_rank(rankings_df, team_b)

    h2h_win_rate, h2h_goal_diff = _compute_h2h(matches_df, team_a, team_b)

    return {
        "rank_diff": rank_a - rank_b,
        "elo_diff": elo_a - elo_b,
        "h2h_win_rate": h2h_win_rate,
        "h2h_goal_diff": h2h_goal_diff,
        "stage_encoded": 0,  # group stage
        "is_host_a": int(team_a in HOSTS_2026),
        "is_host_b": int(team_b in HOSTS_2026),
    }


# ---------------------------------------------------------------------------
# Main prediction runner
# ---------------------------------------------------------------------------

def run_2026_predictions() -> None:
    """Predict outcomes for all 72 FIFA World Cup 2026 group stage matches.

    Loads the trained model from model.pkl, builds inference features for each
    matchup, and upserts results into the match_predictions table.
    Safe to call even if model or DB data is unavailable — logs a warning and
    returns without crashing.
    """
    if not MODEL_PATH.exists():
        logger.warning(
            "No trained model at %s — skipping 2026 predictions.", MODEL_PATH
        )
        return

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.warning("Failed to load model: %s — skipping 2026 predictions.", exc)
        return

    logger.info("Running 2026 group stage predictions ...")

    elo_map = _load_latest_elo()
    rankings_df = _load_latest_rankings()
    matches_df = _load_all_matches()

    create_match_predictions_table()

    run_at = datetime.now(timezone.utc).isoformat()
    prediction_rows = []

    for group_name, teams in GROUPS_2026.items():
        for team_a, team_b in combinations(teams, 2):
            feat = _build_row(team_a, team_b, elo_map, rankings_df, matches_df)
            X = pd.DataFrame([feat])[FEATURE_COLS]

            try:
                proba = model.predict_proba(X)[0]
                predicted_class = int(proba.argmax())
                confidence = float(proba.max())
            except Exception as exc:
                logger.warning(
                    "Prediction failed for %s vs %s: %s — using default.",
                    team_a, team_b, exc,
                )
                predicted_class, confidence = 2, 0.34  # team_a wins as fallback

            if predicted_class == 2:
                predicted_winner = team_a
            elif predicted_class == 0:
                predicted_winner = team_b
            else:
                predicted_winner = "Draw"

            prediction_rows.append((
                2026, group_name, team_a, team_b,
                "Group Stage", predicted_winner, confidence, run_at,
            ))

    # Overwrite any existing 2026 predictions and insert fresh ones
    conn = _open_db()
    try:
        conn.execute(
            "DELETE FROM match_predictions WHERE tournament_year = ?", (2026,)
        )
        conn.executemany(
            "INSERT INTO match_predictions"
            " (tournament_year, group_name, team_a, team_b, stage,"
            "  predicted_winner, confidence, run_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            prediction_rows,
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "Stored %d 2026 group stage predictions in match_predictions table.",
        len(prediction_rows),
    )
