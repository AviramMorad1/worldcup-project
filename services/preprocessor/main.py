"""
main.py
-------
Preprocessor service entry point for the World Cup sentiment pipeline.

Every cycle:
  1. Processes any new raw_reddit_posts → processed_posts
     (VADER + TextBlob scores stored for every post).
  2. Recomputes team_sentiment_daily aggregations.
  3. Recomputes trending_words frequencies.
"""

import json
import logging
import os
import sqlite3
import time
from collections import Counter
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
# Team keyword map
# ---------------------------------------------------------------------------

TEAM_KEYWORDS: dict[str, list[str]] = {
    "Brazil":      ["brazil", "brasileirao", "selecao", "neymar", "vinicius"],
    "France":      ["france", "les bleus", "mbappe", "giroud", "deschamps"],
    "Germany":     ["germany", "deutschland", "die mannschaft", "muller", "kroos"],
    "Argentina":   ["argentina", "messi", "albiceleste", "scaloni", "di maria"],
    "England":     ["england", "three lions", "kane", "southgate", "bellingham"],
    "Spain":       ["spain", "espana", "la roja", "pedri", "morata", "yamal"],
    "Portugal":    ["portugal", "ronaldo", "selecao", "leao", "cancelo"],
    "Netherlands": ["netherlands", "holland", "oranje", "van dijk", "depay"],
    "Japan":       ["japan", "samurai blue", "minamino", "doan"],
    "Morocco":     ["morocco", "atlas lions", "hakimi", "ziyech"],
    "USA":         ["usa", "usmnt", "pulisic", "reyna", "weah"],
    "Mexico":      ["mexico", "el tri", "lozano", "jimenez"],
    "Australia":   ["australia", "socceroos", "leckie", "irvine"],
    "Senegal":     ["senegal", "teranga lions", "mane", "koulibaly"],
    "Croatia":     ["croatia", "vatreni", "modric", "gvardiol"],
}

STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "this", "that", "was", "are", "be", "as",
    "by", "from", "they", "we", "he", "she", "his", "her", "their",
    "have", "has", "had", "will", "would", "not", "no", "so", "if",
    "my", "our", "your", "do", "did", "can", "could", "just", "up",
    "about", "out", "what", "who", "how", "all", "more", "also", "get",
    "one", "i", "you", "s", "t", "re", "ve", "ll", "d", "m", "its",
    "than", "then", "when", "which", "there", "been", "were",
})

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
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id               TEXT UNIQUE,
                cleaned_text          TEXT,
                tokens_json           TEXT,
                vader_compound        REAL,
                vader_pos             REAL,
                vader_neg             REAL,
                vader_neu             REAL,
                textblob_polarity     REAL,
                textblob_subjectivity REAL,
                processed_at          TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_sentiment_daily (
                team         TEXT,
                date         TEXT,
                avg_vader    REAL,
                avg_textblob REAL,
                post_count   INTEGER,
                hype_index   REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trending_words (
                team      TEXT,
                date      TEXT,
                word      TEXT,
                frequency INTEGER
            )
        """)
        conn.commit()
    logger.info("Output tables created / verified.")


# ---------------------------------------------------------------------------
# Team detection
# ---------------------------------------------------------------------------

def detect_team(text: str) -> str | None:
    """Return the first matching team name found in *text*, or None."""
    lower = text.lower()
    for team, keywords in TEAM_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return team
    return None


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
# Post processing  (VADER + TextBlob stored per post)
# ---------------------------------------------------------------------------

def process_posts(posts_df: pd.DataFrame) -> int:
    if posts_df.empty:
        return 0

    processed = 0
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        for _, row in posts_df.iterrows():
            post_id  = str(row.get("id", ""))
            title    = row.get("title", "") or ""
            body     = row.get("body",  "") or ""
            raw_text = f"{title} {body}"

            cleaned   = clean_text(raw_text)
            tokens    = tokenize_text(cleaned)
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
                        sentiment["textblob_polarity"],
                        sentiment["textblob_subjectivity"],
                        now,
                    ),
                )
                processed += 1
            except sqlite3.Error as exc:
                logger.warning("Could not insert post_id '%s': %s", post_id, exc)

        conn.commit()

    return processed


# ---------------------------------------------------------------------------
# Backfill TextBlob for rows that still have NULL textblob_polarity
# (handles posts processed before this version was deployed)
# ---------------------------------------------------------------------------

def backfill_textblob() -> None:
    query = """
        SELECT post_id, cleaned_text
        FROM processed_posts
        WHERE textblob_polarity IS NULL OR textblob_subjectivity IS NULL
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(query).fetchall()
            if not rows:
                logger.info("backfill_textblob: nothing to backfill.")
                return
            logger.info("backfill_textblob: backfilling %d row(s).", len(rows))
            for post_id, cleaned_text in rows:
                sentiment = get_basic_sentiment(cleaned_text or "")
                conn.execute(
                    """
                    UPDATE processed_posts
                    SET textblob_polarity     = ?,
                        textblob_subjectivity = ?
                    WHERE post_id = ?
                    """,
                    (
                        sentiment["textblob_polarity"],
                        sentiment["textblob_subjectivity"],
                        post_id,
                    ),
                )
            conn.commit()
        logger.info("backfill_textblob: done.")
    except sqlite3.Error as exc:
        logger.warning("backfill_textblob failed: %s", exc)


# ---------------------------------------------------------------------------
# Aggregation — team_sentiment_daily
# ---------------------------------------------------------------------------

def compute_team_sentiment_daily() -> None:
    """Recompute team_sentiment_daily from all rows in processed_posts."""
    query = """
        SELECT p.post_id,
               p.vader_compound,
               p.textblob_polarity,
               p.processed_at,
               r.title,
               r.body
        FROM processed_posts AS p
        LEFT JOIN raw_reddit_posts AS r ON r.id = p.post_id
        WHERE p.vader_compound IS NOT NULL
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as exc:
        logger.warning("compute_team_sentiment_daily: load failed: %s", exc)
        return

    if df.empty:
        logger.info("compute_team_sentiment_daily: processed_posts is empty — skipping.")
        return

    # Detect team from raw title+body
    df["combined_text"] = (
        df["title"].fillna("") + " " + df["body"].fillna("")
    ).str.strip()
    df["team"] = df["combined_text"].apply(detect_team)
    df = df[df["team"].notna()].copy()

    if df.empty:
        logger.info("compute_team_sentiment_daily: no team-tagged rows found.")
        return

    # Derive date from processed_at (UTC ISO string)
    df["date"] = (
        pd.to_datetime(df["processed_at"], utc=True, errors="coerce")
        .dt.date.astype(str)
    )
    df = df[df["date"].notna() & (df["date"] != "NaT")]

    agg = (
        df.groupby(["team", "date"])
        .agg(
            avg_vader    = ("vader_compound",    "mean"),
            avg_textblob = ("textblob_polarity", "mean"),
            post_count   = ("post_id",           "count"),
        )
        .reset_index()
    )

    # hype_index = (post_count / max_post_count_that_day) * max(0, avg_vader)
    daily_max = (
        agg.groupby("date")["post_count"]
        .max()
        .rename("max_post_count")
        .reset_index()
    )
    agg = agg.merge(daily_max, on="date")
    agg["hype_index"] = (
        (agg["post_count"] / agg["max_post_count"]) *
        agg["avg_vader"].clip(lower=0)
    )

    rows = list(
        agg[["team", "date", "avg_vader", "avg_textblob",
             "post_count", "hype_index"]].itertuples(index=False, name=None)
    )

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM team_sentiment_daily")
            conn.executemany(
                """
                INSERT INTO team_sentiment_daily
                    (team, date, avg_vader, avg_textblob, post_count, hype_index)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        logger.info(
            "team_sentiment_daily: wrote %d row(s) across %d team(s).",
            len(rows),
            agg["team"].nunique(),
        )
    except sqlite3.Error as exc:
        logger.warning("compute_team_sentiment_daily: write failed: %s", exc)


# ---------------------------------------------------------------------------
# Aggregation — trending_words
# ---------------------------------------------------------------------------

def compute_trending_words() -> None:
    """Recompute trending_words from tokens in processed_posts."""
    query = """
        SELECT p.post_id,
               p.tokens_json,
               p.processed_at,
               r.title,
               r.body
        FROM processed_posts AS p
        LEFT JOIN raw_reddit_posts AS r ON r.id = p.post_id
        WHERE p.tokens_json IS NOT NULL
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as exc:
        logger.warning("compute_trending_words: load failed: %s", exc)
        return

    if df.empty:
        logger.info("compute_trending_words: no data — skipping.")
        return

    df["combined_text"] = (
        df["title"].fillna("") + " " + df["body"].fillna("")
    ).str.strip()
    df["team"] = df["combined_text"].apply(detect_team)
    df = df[df["team"].notna()].copy()

    if df.empty:
        logger.info("compute_trending_words: no team-tagged rows found.")
        return

    df["date"] = (
        pd.to_datetime(df["processed_at"], utc=True, errors="coerce")
        .dt.date.astype(str)
    )
    df = df[df["date"].notna() & (df["date"] != "NaT")]

    freq: dict[tuple[str, str], Counter] = {}
    for _, row in df.iterrows():
        team = row["team"]
        date = row["date"]
        try:
            tokens: list[str] = json.loads(row["tokens_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        key = (team, date)
        if key not in freq:
            freq[key] = Counter()
        meaningful = [
            t for t in tokens
            if t
            and len(t) > 2
            and t.isalpha()
            and t not in STOPWORDS
        ]
        freq[key].update(meaningful)

    rows: list[tuple] = []
    for (team, date), counter in freq.items():
        for word, count in counter.most_common(50):
            rows.append((team, date, word, count))

    if not rows:
        logger.info("compute_trending_words: no word frequencies to write.")
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM trending_words")
            conn.executemany(
                "INSERT INTO trending_words (team, date, word, frequency) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        logger.info("trending_words: wrote %d row(s).", len(rows))
    except sqlite3.Error as exc:
        logger.warning("compute_trending_words: write failed: %s", exc)


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

    # 1. Process any new posts (VADER + TextBlob)
    posts_df = load_unprocessed_posts()
    if posts_df.empty:
        logger.info("No unprocessed Reddit posts found.")
    else:
        count = process_posts(posts_df)
        logger.info("Processed %d new post(s).", count)

    # 2. Backfill TextBlob for any rows that pre-date this deployment
    backfill_textblob()

    # 3. Recompute aggregations unconditionally — runs even when no new posts
    compute_team_sentiment_daily()
    compute_trending_words()

    logger.info("Preprocessing cycle completed")


def main() -> None:
    logger.info("Preprocessor service started")
    run_preprocessing_cycle()

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()