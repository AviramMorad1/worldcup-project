import logging
import os
import sqlite3

import pandas as pd

DB_PATH = "/app/data/worldcup.db"

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"year", "team_a", "team_b", "score_a", "score_b"}


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
                "SELECT * FROM %s" % table_name,  # table names cannot be parameterised in SQLite
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

    logger.info("Training data validated: %d rows, all required columns present.", len(matches_df))
    return True


def build_feature_matrix(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if not validate_training_data(matches_df):
        logger.warning("Cannot build feature matrix — training data is invalid.")
        return pd.DataFrame(), pd.Series(dtype=int)

    df = matches_df.copy()

    df["score_a"] = pd.to_numeric(df["score_a"], errors="coerce")
    df["score_b"] = pd.to_numeric(df["score_b"], errors="coerce")
    df = df.dropna(subset=["score_a", "score_b"])

    df["goal_diff"] = df["score_a"] - df["score_b"]

    df["outcome"] = df.apply(
        lambda row: 2 if row["score_a"] > row["score_b"]
        else (1 if row["score_a"] == row["score_b"] else 0),
        axis=1,
    )

    feature_cols = [c for c in ["score_a", "score_b", "goal_diff", "year"] if c in df.columns]
    X = df[feature_cols].reset_index(drop=True)
    y = df["outcome"].reset_index(drop=True)

    logger.info(
        "Feature matrix built: %d rows, features=%s.",
        len(X),
        feature_cols,
    )
    return X, y
