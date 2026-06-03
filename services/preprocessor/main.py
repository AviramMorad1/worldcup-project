"""
main.py
-------
Preprocessor service entry point for the World Cup sentiment pipeline.

Every cycle:
  1. Processes any new raw posts from all sources (Reddit, Telegram, Telegram comments)
     → processed_posts  (VADER + TextBlob + detected_team stored per post)
  2. Backfills detected_team for old rows that pre-date this version.
  3. Recomputes team_sentiment_daily aggregations.
  4. Recomputes trending_words frequencies.

Telegram comment team-inheritance rule:
  - detect team from parent post title/body first
  - fallback to comment body if parent post has no detectable team
  - store the result in processed_posts.detected_team
  This ensures short comments ("Ooh no", "Dem all hurt") still contribute to
  the correct national team's sentiment when their parent post is team-tagged.
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
COLLECTOR_READY_FLAG = "/app/data/collector_ready.flag"
COLLECTOR_WAIT_POLL_SECONDS = 10
COLLECTOR_WAIT_MAX_SECONDS = 1800
DEFAULT_PREPROCESS_INTERVAL_MINUTES = 60

logging.basicConfig(
    format="[PREPROCESSOR][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team keyword map  (order matters — first match wins)
# ---------------------------------------------------------------------------

TEAM_KEYWORDS: dict[str, list[str]] = {
    "Argentina":   ["argentina", "argentine", "albiceleste", "messi", "scaloni",
                    "di maria", "lautaro", "martinez"],
    "Australia":   ["australia", "socceroos", "leckie", "irvine"],
    "Belgium":     ["belgium", "red devils", "de bruyne", "lukaku", "courtois", "hazard"],
    "Brazil":      ["brazil", "brasil", "brasileirao", "selecao", "seleção",
                    "neymar", "vinicius", "rodrygo"],
    "Colombia":    ["colombia", "cafeteros", "falcao", "james rodriguez"],
    "Croatia":     ["croatia", "vatreni", "modric", "gvardiol", "kovacic"],
    "England":     ["england", "three lions", "kane", "southgate", "bellingham",
                    "rashford", "saka"],
    "France":      ["france", "french", "les bleus", "mbappe", "giroud", "deschamps",
                    "griezmann", "cherki", "camavinga", "kounde"],
    "Germany":     ["germany", "deutschland", "die mannschaft", "muller", "kroos", "neuer"],
    "Ghana":       ["ghana", "black stars", "ayew", "partey", "kudus", "salisu"],
    "Italy":       ["italy", "italia", "azzurri", "mancini", "chiesa", "donnarumma"],
    "Japan":       ["japan", "samurai blue", "minamino", "doan", "kubo"],
    "Korea":       ["south korea", "korea republic", "son heung", "hwang in", "kim min"],
    "Mexico":      ["mexico", "el tri", "lozano", "jimenez", "ochoa"],
    "Morocco":     ["morocco", "atlas lions", "hakimi", "ziyech", "amrabat", "ounahi"],
    "Netherlands": ["netherlands", "holland", "oranje", "van dijk", "depay", "dumfries"],
    "Portugal":    ["portugal", "ronaldo", "cristiano", "leao", "cancelo", "bernardo silva"],
    "Senegal":     ["senegal", "teranga lions", "mane", "koulibaly", "diatta"],
    "Spain":       ["spain", "espana", "la roja", "pedri", "morata", "yamal", "fabian"],
    "Switzerland": ["switzerland", "swiss nati", "xhaka", "shaqiri", "sommer"],
    "Uruguay":     ["uruguay", "celeste", "suarez", "cavani", "valverde", "nunez"],
    "USA":         ["usa", "united states", "usmnt", "pulisic", "reyna", "weah", "dest"],
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


def _get_table_columns(table_name: str) -> list[str]:
    """Return the list of column names for a table, or [] if table does not exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            return [row[1] for row in rows]
    except sqlite3.Error:
        return []


def create_output_tables() -> None:
    """Create output tables if they do not exist, then run safe column migrations."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_posts (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id               TEXT UNIQUE,
                source                TEXT DEFAULT 'reddit',
                detected_team         TEXT,
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
    _migrate_processed_posts()


def _migrate_processed_posts() -> None:
    """Add columns to processed_posts that older versions did not include."""
    cols = _get_table_columns("processed_posts")
    if not cols:
        return

    migrations: list[tuple[str, str]] = [
        ("source",        "ALTER TABLE processed_posts ADD COLUMN source TEXT DEFAULT 'reddit'"),
        ("detected_team", "ALTER TABLE processed_posts ADD COLUMN detected_team TEXT"),
    ]

    try:
        with sqlite3.connect(DB_PATH) as conn:
            for col_name, sql in migrations:
                if col_name not in cols:
                    conn.execute(sql)
                    conn.commit()
                    logger.info("Migrated processed_posts: added '%s' column.", col_name)

            # Ensure old Reddit rows that have NULL source are labelled correctly
            conn.execute(
                "UPDATE processed_posts SET source = 'reddit' WHERE source IS NULL"
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning("processed_posts migration failed: %s", exc)


# ---------------------------------------------------------------------------
# Team detection
# ---------------------------------------------------------------------------


def detect_team(text: str) -> str | None:
    """
    Return the first matching team name found in *text*, or None.

    Matching is case-insensitive substring search.
    The TEAM_KEYWORDS dict is ordered: first match wins.
    """
    if not text:
        return None
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
    """
    Load unprocessed posts from all three raw sources.

    For Telegram comments we also JOIN the parent post so that the comment
    processing step can inherit the team context from the parent post.
    """
    frames: list[pd.DataFrame] = []

    # --- Reddit posts ---
    if table_exists("raw_reddit_posts"):
        q = """
            SELECT r.id,
                   r.title,
                   r.body,
                   r.created_utc,
                   'reddit'    AS source,
                   NULL        AS parent_title,
                   NULL        AS parent_body
            FROM raw_reddit_posts AS r
            LEFT JOIN processed_posts AS p ON r.id = p.post_id
            WHERE p.post_id IS NULL
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                frames.append(pd.read_sql_query(q, conn))
        except Exception as exc:
            logger.warning("Failed to load unprocessed Reddit posts: %s", exc)

    # --- Telegram posts ---
    if table_exists("raw_telegram_posts"):
        q = """
            SELECT t.id,
                   t.title,
                   t.body,
                   t.created_utc,
                   'telegram'  AS source,
                   NULL        AS parent_title,
                   NULL        AS parent_body
            FROM raw_telegram_posts AS t
            LEFT JOIN processed_posts AS p ON t.id = p.post_id
            WHERE p.post_id IS NULL
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                frames.append(pd.read_sql_query(q, conn))
        except Exception as exc:
            logger.warning("Failed to load unprocessed Telegram posts: %s", exc)

    # --- Telegram comments (with parent post body for team inheritance) ---
    if table_exists("raw_telegram_comments"):
        q = """
            SELECT c.id,
                   NULL              AS title,
                   c.body,
                   c.created_utc,
                   'telegram_comment' AS source,
                   tp.title          AS parent_title,
                   tp.body           AS parent_body
            FROM raw_telegram_comments AS c
            LEFT JOIN raw_telegram_posts AS tp ON c.parent_post_id = tp.id
            LEFT JOIN processed_posts   AS p  ON c.id = p.post_id
            WHERE p.post_id IS NULL
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                df_tc = pd.read_sql_query(q, conn)
            frames.append(df_tc)
        except Exception as exc:
            logger.warning("Failed to load unprocessed Telegram comments: %s", exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).head(limit)

    # Log per-source breakdown
    if not combined.empty and "source" in combined.columns:
        counts = combined["source"].value_counts().to_dict()
        logger.info(
            "Loaded %d unprocessed row(s): %s",
            len(combined),
            ", ".join(f"{s}={n}" for s, n in counts.items()),
        )
    else:
        logger.info("Loaded %d unprocessed row(s).", len(combined))

    return combined


# ---------------------------------------------------------------------------
# Post processing  (VADER + TextBlob + detected_team stored per post)
# ---------------------------------------------------------------------------


def process_posts(posts_df: pd.DataFrame) -> int:
    """
    Run sentiment analysis and team detection on all rows in *posts_df*.

    Team-detection strategy per source:
      reddit / telegram  — detect from title + body
      telegram_comment   — detect from parent post title/body first;
                           fallback to comment body if parent has no team signal
    """
    if posts_df.empty:
        return 0

    processed = 0
    inherited_team_count = 0
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        for _, row in posts_df.iterrows():
            post_id = str(row.get("id", ""))
            title   = row.get("title", "") or ""
            body    = row.get("body",  "") or ""
            source  = str(row.get("source", "reddit") or "reddit")
            raw_text = f"{title} {body}".strip()

            cleaned   = clean_text(raw_text)
            tokens    = tokenize_text(cleaned)
            sentiment = get_basic_sentiment(cleaned)

            # --- Team detection ---
            detected_team: str | None = None

            if source == "telegram_comment":
                # Try parent post first — this is the key inheritance step
                parent_title = row.get("parent_title", "") or ""
                parent_body  = row.get("parent_body",  "") or ""
                parent_text  = f"{parent_title} {parent_body}".strip()

                if parent_text:
                    detected_team = detect_team(parent_text)
                    if detected_team:
                        inherited_team_count += 1

                # Fallback: try the comment text itself
                if not detected_team:
                    detected_team = detect_team(raw_text)
            else:
                detected_team = detect_team(raw_text)

            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO processed_posts (
                        post_id, source, detected_team, cleaned_text, tokens_json,
                        vader_compound, vader_pos, vader_neg, vader_neu,
                        textblob_polarity, textblob_subjectivity, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post_id,
                        source,
                        detected_team,
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

    if inherited_team_count:
        logger.info(
            "process_posts: %d telegram_comment row(s) inherited team from parent post.",
            inherited_team_count,
        )
    return processed


# ---------------------------------------------------------------------------
# Backfill helpers
# ---------------------------------------------------------------------------


def backfill_textblob() -> None:
    """Backfill textblob scores for rows that pre-date TextBlob deployment."""
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


def backfill_detected_team() -> None:
    """
    Fill detected_team for processed_posts rows that were inserted before this
    column was added (or before team detection ran).

    Strategy:
      - For telegram_comment rows: join raw_telegram_comments → raw_telegram_posts
        to get parent post body; detect team from parent first.
      - For all other rows: join the raw_union to get title/body; detect from there.

    Only touches rows where detected_team IS NULL.
    """
    cols = _get_table_columns("processed_posts")
    if "detected_team" not in cols:
        return  # column migration hasn't run yet — skip

    try:
        with sqlite3.connect(DB_PATH) as conn:
            null_count = conn.execute(
                "SELECT COUNT(*) FROM processed_posts WHERE detected_team IS NULL"
            ).fetchone()[0]

        if null_count == 0:
            logger.info("backfill_detected_team: nothing to backfill.")
            return

        logger.info("backfill_detected_team: backfilling %d row(s).", null_count)
    except sqlite3.Error as exc:
        logger.warning("backfill_detected_team: count query failed: %s", exc)
        return

    # --- Backfill telegram_comment rows using parent post ---
    if table_exists("raw_telegram_comments") and table_exists("raw_telegram_posts"):
        try:
            q = """
                SELECT p.post_id, tp.title, tp.body, c.body AS comment_body
                FROM processed_posts AS p
                JOIN raw_telegram_comments AS c  ON c.id = p.post_id
                LEFT JOIN raw_telegram_posts AS tp ON c.parent_post_id = tp.id
                WHERE p.detected_team IS NULL
                  AND p.source = 'telegram_comment'
            """
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql_query(q, conn)

            updated = 0
            with sqlite3.connect(DB_PATH) as conn:
                for _, row in df.iterrows():
                    parent_text = (
                        (row.get("title") or "") + " " + (row.get("body") or "")
                    ).strip()
                    team = detect_team(parent_text) or detect_team(
                        row.get("comment_body", "") or ""
                    )
                    if team:
                        conn.execute(
                            "UPDATE processed_posts SET detected_team=? WHERE post_id=?",
                            (team, row["post_id"]),
                        )
                        updated += 1
                conn.commit()
            logger.info(
                "backfill_detected_team: updated %d telegram_comment row(s).", updated
            )
        except Exception as exc:
            logger.warning(
                "backfill_detected_team: telegram_comment pass failed: %s", exc
            )

    # --- Backfill all other rows using raw post union ---
    union_parts = []
    for tbl, has_title in [
        ("raw_reddit_posts", True),
        ("raw_telegram_posts", True),
    ]:
        if table_exists(tbl):
            union_parts.append(
                f"SELECT id, title, body FROM {tbl}"  # noqa: S608
                if has_title
                else f"SELECT id, NULL AS title, body FROM {tbl}"  # noqa: S608
            )

    if not union_parts:
        return

    union_sql = " UNION ALL ".join(union_parts)
    try:
        q = f"""
            SELECT p.post_id, r.title, r.body
            FROM processed_posts AS p
            LEFT JOIN ({union_sql}) AS r ON r.id = p.post_id
            WHERE p.detected_team IS NULL
              AND p.source != 'telegram_comment'
        """  # noqa: S608
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(q, conn)

        updated = 0
        with sqlite3.connect(DB_PATH) as conn:
            for _, row in df.iterrows():
                text = (
                    (row.get("title") or "") + " " + (row.get("body") or "")
                ).strip()
                team = detect_team(text)
                if team:
                    conn.execute(
                        "UPDATE processed_posts SET detected_team=? WHERE post_id=?",
                        (team, row["post_id"]),
                    )
                    updated += 1
            conn.commit()
        logger.info(
            "backfill_detected_team: updated %d reddit/telegram-post row(s).", updated
        )
    except Exception as exc:
        logger.warning("backfill_detected_team: general pass failed: %s", exc)


# ---------------------------------------------------------------------------
# Aggregation — team_sentiment_daily
# ---------------------------------------------------------------------------


def _raw_union_created_utc_sql() -> str:
    """Return SQL for a UNION of (id, created_utc) across all raw post tables."""
    parts = ["SELECT id, created_utc FROM raw_reddit_posts"]
    if table_exists("raw_telegram_posts"):
        parts.append("SELECT id, created_utc FROM raw_telegram_posts")
    if table_exists("raw_telegram_comments"):
        parts.append("SELECT id, created_utc FROM raw_telegram_comments")
    return " UNION ALL ".join(parts)


def _raw_union_sql() -> str:
    """Return SQL for a UNION of (id, title, body, created_utc) across all raw post tables."""
    parts = ["SELECT id, title, body, created_utc FROM raw_reddit_posts"]
    if table_exists("raw_telegram_posts"):
        parts.append("SELECT id, title, body, created_utc FROM raw_telegram_posts")
    if table_exists("raw_telegram_comments"):
        parts.append(
            "SELECT id, NULL AS title, body, created_utc FROM raw_telegram_comments"
        )
    return " UNION ALL ".join(parts)


def _resolve_date_column(df: pd.DataFrame, fn_name: str) -> pd.DataFrame:
    """Derive date from created_utc (post publish time), fallback to processed_at."""
    if "created_utc" in df.columns:
        primary = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
    else:
        primary = pd.Series(pd.NaT, index=df.index)

    fallback = pd.to_datetime(df["processed_at"], utc=True, errors="coerce")

    n_fallback = primary.isna().sum()
    if n_fallback > 0:
        logger.warning(
            "%s: %d row(s) have missing/invalid created_utc — falling back to processed_at.",
            fn_name,
            n_fallback,
        )

    resolved = primary.where(primary.notna(), fallback)
    df = df.copy()
    df["date"] = resolved.dt.date.astype(str)

    before = len(df)
    df = df[df["date"].notna() & (df["date"] != "NaT") & (df["date"] != "None")]
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "%s: dropped %d row(s) where both created_utc and processed_at were invalid.",
            fn_name,
            dropped,
        )
    return df


def compute_team_sentiment_daily() -> None:
    """
    Recompute team_sentiment_daily from all rows in processed_posts.

    Uses stored processed_posts.detected_team (preferred).
    For rows where detected_team IS NULL, re-attempts detection from raw text
    so that existing data before this migration still contributes.
    Rows with no detectable team are skipped from aggregation.
    """
    # Join processed_posts with raw union only to get created_utc and fallback text
    query = f"""
        SELECT p.post_id,
               p.vader_compound,
               p.textblob_polarity,
               p.processed_at,
               p.detected_team,
               r.title,
               r.body,
               r.created_utc
        FROM processed_posts AS p
        LEFT JOIN ({_raw_union_sql()}) AS r ON r.id = p.post_id
        WHERE p.vader_compound IS NOT NULL
    """  # noqa: S608

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as exc:
        logger.warning("compute_team_sentiment_daily: load failed: %s", exc)
        return

    if df.empty:
        logger.info("compute_team_sentiment_daily: processed_posts is empty — skipping.")
        return

    # Use stored detected_team; fallback to on-the-fly detection for legacy rows
    if "detected_team" in df.columns:
        mask_null = df["detected_team"].isna()
        if mask_null.any():
            combined = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
            df.loc[mask_null, "detected_team"] = combined[mask_null].apply(detect_team)

        df["team"] = df["detected_team"]
    else:
        combined = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
        df["team"] = combined.apply(detect_team)

    skipped = df["team"].isna().sum()
    if skipped:
        logger.info(
            "compute_team_sentiment_daily: skipping %d row(s) with no detectable team.",
            skipped,
        )

    df = df[df["team"].notna()].copy()
    if df.empty:
        logger.info("compute_team_sentiment_daily: no team-tagged rows found.")
        return

    df = _resolve_date_column(df, "compute_team_sentiment_daily")

    agg = (
        df.groupby(["team", "date"])
        .agg(
            avg_vader    = ("vader_compound",    "mean"),
            avg_textblob = ("textblob_polarity", "mean"),
            post_count   = ("post_id",           "count"),
        )
        .reset_index()
    )

    daily_max = (
        agg.groupby("date")["post_count"]
        .max()
        .rename("max_post_count")
        .reset_index()
    )
    agg = agg.merge(daily_max, on="date")
    agg["hype_index"] = (
        (agg["post_count"] / agg["max_post_count"]) * agg["avg_vader"].clip(lower=0)
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
    query = f"""
        SELECT p.post_id,
               p.tokens_json,
               p.processed_at,
               p.detected_team,
               r.title,
               r.body,
               r.created_utc
        FROM processed_posts AS p
        LEFT JOIN ({_raw_union_sql()}) AS r ON r.id = p.post_id
        WHERE p.tokens_json IS NOT NULL
    """  # noqa: S608

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as exc:
        logger.warning("compute_trending_words: load failed: %s", exc)
        return

    if df.empty:
        logger.info("compute_trending_words: no data — skipping.")
        return

    # Use stored detected_team; fallback to text detection for legacy rows
    if "detected_team" in df.columns:
        mask_null = df["detected_team"].isna()
        if mask_null.any():
            combined = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
            df.loc[mask_null, "detected_team"] = combined[mask_null].apply(detect_team)
        df["team"] = df["detected_team"]
    else:
        combined = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
        df["team"] = combined.apply(detect_team)

    df = df[df["team"].notna()].copy()
    if df.empty:
        logger.info("compute_trending_words: no team-tagged rows found.")
        return

    df = _resolve_date_column(df, "compute_trending_words")

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
            if t and len(t) > 2 and t.isalpha() and t not in STOPWORDS
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
# Collector coordination
# ---------------------------------------------------------------------------


def _raw_post_count() -> int:
    """Return total number of raw posts across all sources."""
    total = 0
    for tbl in ("raw_reddit_posts", "raw_telegram_posts"):
        if table_exists(tbl):
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    row = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()  # noqa: S608
                    total += int(row[0]) if row else 0
            except sqlite3.Error:
                pass
    return total


def wait_for_collector() -> None:
    """Wait until the collector finishes its first cycle or posts exist in SQLite."""
    if os.path.exists(COLLECTOR_READY_FLAG):
        logger.info("Collector ready flag already present — proceeding.")
        return

    if _raw_post_count() > 0:
        logger.info("Found existing raw posts — proceeding without wait.")
        return

    deadline = time.time() + COLLECTOR_WAIT_MAX_SECONDS
    logger.info(
        "Waiting for collector (flag or posts), up to %d seconds...",
        COLLECTOR_WAIT_MAX_SECONDS,
    )

    while time.time() < deadline:
        if os.path.exists(COLLECTOR_READY_FLAG):
            try:
                with open(COLLECTOR_READY_FLAG, encoding="utf-8") as fh:
                    payload = json.load(fh)
                logger.info(
                    "Collector ready flag found (completed_at=%s, posts=%s).",
                    payload.get("completed_at"),
                    payload.get("posts_collected_this_cycle"),
                )
            except (OSError, json.JSONDecodeError):
                logger.info("Collector ready flag found — proceeding.")
            return

        count = _raw_post_count()
        if count > 0:
            logger.info("Found %d raw post(s) — proceeding.", count)
            return

        time.sleep(COLLECTOR_WAIT_POLL_SECONDS)

    logger.warning(
        "Timed out after %d seconds waiting for collector — proceeding anyway.",
        COLLECTOR_WAIT_MAX_SECONDS,
    )


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------


def _preprocess_interval_seconds() -> int:
    raw = os.environ.get(
        "PREPROCESS_INTERVAL_MINUTES",
        str(DEFAULT_PREPROCESS_INTERVAL_MINUTES),
    )
    try:
        minutes = max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid PREPROCESS_INTERVAL_MINUTES=%r — using default %d.",
            raw,
            DEFAULT_PREPROCESS_INTERVAL_MINUTES,
        )
        minutes = DEFAULT_PREPROCESS_INTERVAL_MINUTES
    return minutes * 60


def run_preprocessing_cycle() -> int:
    logger.info("Starting preprocessing cycle")

    if not database_exists():
        logger.warning("Database not found — skipping preprocessing.")
        return 0

    has_reddit   = table_exists("raw_reddit_posts")
    has_telegram = table_exists("raw_telegram_posts")
    has_tg_comm  = table_exists("raw_telegram_comments")

    if not has_reddit and not has_telegram and not has_tg_comm:
        logger.warning("No raw post tables found — waiting for collector.")
        create_output_tables()
        return 0

    create_output_tables()

    # 1. Process new posts/comments
    posts_df = load_unprocessed_posts()
    processed_count = 0
    if posts_df.empty:
        logger.info("No unprocessed posts found in any source.")
    else:
        processed_count = process_posts(posts_df)

    logger.info("Processed %d new row(s) this cycle.", processed_count)

    # 2. Backfill old rows
    backfill_textblob()
    backfill_detected_team()

    # 3. Recompute aggregations
    compute_team_sentiment_daily()
    compute_trending_words()

    logger.info("Preprocessing cycle completed")
    return processed_count


def main() -> None:
    interval_seconds = _preprocess_interval_seconds()
    interval_minutes = interval_seconds // 60

    logger.info("Preprocessor service started")
    logger.info("Preprocess interval: %d minute(s).", interval_minutes)
    wait_for_collector()

    while True:
        try:
            run_preprocessing_cycle()
        except Exception as exc:
            logger.exception("Preprocessing cycle failed: %s", exc)

        logger.info(
            "Sleeping for %d minutes before next preprocessing cycle.",
            interval_minutes,
        )
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
