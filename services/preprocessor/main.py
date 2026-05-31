import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

from sentiment import get_basic_sentiment
from text_cleaner import clean_text, tokenize_text

DB_PATH = "/app/data/worldcup.db"

logging.basicConfig(
    format="[PREPROCESSOR][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB helpers
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
        logger.warning("Error checking table '%s': %s", table_name, exc)
        return False


def create_output_tables() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_posts (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id             TEXT UNIQUE,
                cleaned_text        TEXT,
                tokens_json         TEXT,
                vader_compound      REAL,
                vader_pos           REAL,
                vader_neg           REAL,
                vader_neu           REAL,
                textblob_polarity   REAL,
                textblob_subjectivity REAL,
                processed_at        TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_sentiment_daily (
                team        TEXT,
                date        TEXT,
                avg_vader   REAL,
                avg_textblob REAL,
                post_count  INTEGER,
                hype_index  REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trending_words (
                team        TEXT,
                date        TEXT,
                word        TEXT,
                frequency   INTEGER
            )
        """)
        conn.commit()
    logger.info("Output tables created / verified.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_unprocessed_posts(limit: int = 100) -> pd.DataFrame:
    if not table_exists("raw_reddit_posts"):
        return pd.DataFrame()

    query = """
        SELECT r.*
        FROM raw_reddit_posts AS r
        LEFT JOIN processed_posts AS p ON r.id = p.post_id
        WHERE p.post_id IS NULL
        LIMIT ?
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn, params=(limit,))
        logger.info("Loaded %d unprocessed post(s).", len(df))
        return df
    except Exception as exc:
        logger.warning("Failed to load unprocessed posts: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_posts(posts_df: pd.DataFrame) -> int:
    if posts_df.empty:
        return 0

    processed = 0
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        for _, row in posts_df.iterrows():
            post_id = str(row.get("id", ""))
            title = row.get("title", "") or ""
            body = row.get("body", "") or ""
            raw_text = f"{title} {body}"

            cleaned = clean_text(raw_text)
            tokens = tokenize_text(cleaned)
            sentiment = get_basic_sentiment(cleaned)

            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO processed_posts (
                        post_id, cleaned_text, tokens_json,
                        vader_compound, vader_pos, vader_neg, vader_neu,
                        textblob_polarity, textblob_subjectivity, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post_id,
                        cleaned,
                        json.dumps(tokens),
                        sentiment["compound"],
                        sentiment["positive"],
                        sentiment["negative"],
                        sentiment["neutral"],
                        None,   # textblob_polarity — real VADER/TextBlob not yet implemented
                        None,   # textblob_subjectivity
                        now,
                    ),
                )
                processed += 1
            except sqlite3.Error as exc:
                logger.warning("Could not insert post_id '%s': %s", post_id, exc)

        conn.commit()

    return processed


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_preprocessing_cycle() -> None:
    logger.info("Starting preprocessing cycle")

    if not database_exists():
        logger.warning("Database not found — skipping preprocessing.")
        return

    if not table_exists("raw_reddit_posts"):
        logger.warning("raw_reddit_posts table not found — waiting for collector.")
        create_output_tables()
        return

    create_output_tables()

    posts_df = load_unprocessed_posts()

    if posts_df.empty:
        logger.info("No unprocessed Reddit posts found.")
        return

    count = process_posts(posts_df)
    logger.info("Processed %d post(s).", count)
    logger.info("Preprocessing cycle completed")


def main() -> None:
    logger.info("Preprocessor service started")
    run_preprocessing_cycle()

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
