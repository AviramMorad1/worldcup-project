import hashlib
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape

import feedparser
import requests


DB_PATH = "/app/data/worldcup.db"

SUBREDDITS = ["worldcup", "soccer", "FIFA"]
POST_LIMIT_PER_SUBREDDIT = 50
COMMENT_LIMIT_PER_POST = 10

USER_AGENT = "worldcup-project/1.0 student research"
REQUEST_TIMEOUT_SECONDS = 30

logger = logging.getLogger(__name__)


def create_reddit_tables(conn: sqlite3.Connection) -> None:
    """
    Create Reddit raw data tables if they do not exist.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_reddit_posts (
            id TEXT PRIMARY KEY,
            subreddit TEXT,
            title TEXT,
            body TEXT,
            author TEXT,
            score INTEGER,
            created_utc REAL,
            collected_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_reddit_comments (
            id TEXT PRIMARY KEY,
            post_id TEXT,
            body TEXT,
            score INTEGER,
            created_utc REAL,
            collected_at TEXT
        )
        """
    )

    conn.commit()


def strip_html(value: str) -> str:
    """
    Convert simple RSS HTML summaries into plain text.
    """
    if not value:
        return ""

    value = unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_datetime_to_utc_timestamp(value: str | None) -> float:
    """
    Convert RSS published/updated string to UTC timestamp.
    """
    if not value:
        return datetime.now(timezone.utc).timestamp()

    try:
        parsed = parsedate_to_datetime(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.timestamp()

    except Exception:
        return datetime.now(timezone.utc).timestamp()


def extract_post_id_from_link(link: str) -> str | None:
    """
    Extract Reddit post id from links like:
    https://www.reddit.com/r/soccer/comments/abc123/title/
    """
    if not link:
        return None

    match = re.search(r"/comments/([^/]+)/", link)

    if not match:
        return None

    return match.group(1)


def stable_id(prefix: str, value: str) -> str:
    """
    Generate stable IDs for RSS entries when Reddit does not provide clean IDs.
    """
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def fetch_rss(url: str):
    """
    Fetch RSS using requests so we can set a proper User-Agent.
    """
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"RSS request failed. status={response.status_code}, url={url}, "
            f"body={response.text[:200]}"
        )

    return feedparser.parse(response.text)


def insert_post(
    conn: sqlite3.Connection,
    post_id: str,
    subreddit: str,
    title: str,
    body: str,
    author: str | None,
    score: int,
    created_utc: float,
    collected_at: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO raw_reddit_posts
        (id, subreddit, title, body, author, score, created_utc, collected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            post_id,
            subreddit,
            title,
            body,
            author,
            score,
            created_utc,
            collected_at,
        ),
    )


def insert_comment(
    conn: sqlite3.Connection,
    comment_id: str,
    post_id: str,
    body: str,
    score: int,
    created_utc: float,
    collected_at: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO raw_reddit_comments
        (id, post_id, body, score, created_utc, collected_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            comment_id,
            post_id,
            body,
            score,
            created_utc,
            collected_at,
        ),
    )


def collect_comments_from_post_rss(
    conn: sqlite3.Connection,
    post_id: str,
    post_link: str,
    collected_at: str,
) -> int:
    """
    Try to collect comments from a Reddit comments RSS feed.

    This is best-effort. Reddit RSS comment feeds may vary by page and may not
    always expose all comments.
    """
    if not post_link:
        return 0

    rss_url = post_link.rstrip("/") + ".rss"

    try:
        feed = fetch_rss(rss_url)
    except Exception:
        logger.exception("Failed fetching comments RSS for post_id=%s", post_id)
        return 0

    comments_inserted = 0

    for entry in feed.entries[:COMMENT_LIMIT_PER_POST + 1]:
        entry_link = getattr(entry, "link", "")
        entry_id = getattr(entry, "id", "") or entry_link

        # Skip the original submission if it appears in the comments RSS.
        if post_id in entry_id and comments_inserted == 0:
            title = getattr(entry, "title", "")
            if title:
                continue

        body = strip_html(getattr(entry, "summary", ""))

        if not body:
            continue

        comment_id = stable_id("rss_comment", entry_id or body)
        created_utc = parse_datetime_to_utc_timestamp(
            getattr(entry, "published", None) or getattr(entry, "updated", None)
        )

        insert_comment(
            conn=conn,
            comment_id=comment_id,
            post_id=post_id,
            body=body,
            score=0,
            created_utc=created_utc,
            collected_at=collected_at,
        )

        comments_inserted += 1

        if comments_inserted >= COMMENT_LIMIT_PER_POST:
            break

    return comments_inserted


def collect_from_subreddit(conn: sqlite3.Connection, subreddit_name: str) -> tuple[int, int]:
    """
    Collect real Reddit posts from subreddit RSS and best-effort comments.
    """
    logger.info("Collecting Reddit RSS data from r/%s", subreddit_name)

    collected_at = datetime.now(timezone.utc).isoformat()

    rss_urls = [
        f"https://www.reddit.com/r/{subreddit_name}/.rss",
        f"https://old.reddit.com/r/{subreddit_name}/.rss",
    ]

    feed = None
    last_error = None

    for rss_url in rss_urls:
        try:
            feed = fetch_rss(rss_url)
            logger.info("RSS fetch succeeded for r/%s using %s", subreddit_name, rss_url)
            break
        except Exception as error:
            last_error = error
            logger.warning("RSS fetch failed for %s: %s", rss_url, error)

    if feed is None:
        raise RuntimeError(f"Failed fetching RSS for r/{subreddit_name}: {last_error}")

    posts_inserted = 0
    comments_inserted = 0

    for entry in feed.entries[:POST_LIMIT_PER_SUBREDDIT]:
        link = getattr(entry, "link", "")
        post_id = extract_post_id_from_link(link)

        if not post_id:
            post_id = stable_id("rss_post", getattr(entry, "id", "") or link)

        title = strip_html(getattr(entry, "title", ""))
        body = strip_html(getattr(entry, "summary", ""))
        author = getattr(entry, "author", None)
        created_utc = parse_datetime_to_utc_timestamp(
            getattr(entry, "published", None) or getattr(entry, "updated", None)
        )

        insert_post(
            conn=conn,
            post_id=post_id,
            subreddit=subreddit_name,
            title=title,
            body=body,
            author=author,
            score=0,
            created_utc=created_utc,
            collected_at=collected_at,
        )

        posts_inserted += 1

        comments_inserted += collect_comments_from_post_rss(
            conn=conn,
            post_id=post_id,
            post_link=link,
            collected_at=collected_at,
        )

        time.sleep(1)

    conn.commit()

    logger.info(
        "Finished r/%s RSS collection: posts=%s comments=%s",
        subreddit_name,
        posts_inserted,
        comments_inserted,
    )

    return posts_inserted, comments_inserted


def collect_reddit_data() -> None:
    """
    Main Reddit collection function using real Reddit RSS data.
    """
    total_posts = 0
    total_comments = 0

    with sqlite3.connect(DB_PATH) as conn:
        create_reddit_tables(conn)

        for subreddit_name in SUBREDDITS:
            try:
                posts_count, comments_count = collect_from_subreddit(
                    conn=conn,
                    subreddit_name=subreddit_name,
                )

                total_posts += posts_count
                total_comments += comments_count

            except Exception:
                logger.exception("Failed collecting RSS data from r/%s", subreddit_name)

    logger.info(
        "Reddit RSS collection finished. total_posts=%s total_comments=%s",
        total_posts,
        total_comments,
    )