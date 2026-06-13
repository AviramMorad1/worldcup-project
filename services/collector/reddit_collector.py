"""Collect Reddit posts via public RSS feeds (no API credentials required)."""

import calendar
import hashlib
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from html import unescape
from urllib.parse import quote

import feedparser
import requests


DB_PATH = "/app/data/worldcup.db"

SUBREDDITS = [
    "worldcup",
    "soccer",
    "football",
    "sports",
    "soccercirclejerk"
]

POST_LIMIT_PER_SUBREDDIT = 25
SUBREDDIT_SLEEP_SECONDS = 8

# Direct subreddit RSS is blocked for some communities (e.g. r/FIFA returns HTTP 403).
# Fall back to Reddit search RSS for FIFA-related posts across the site.
SEARCH_FALLBACK_QUERIES: dict[str, str] = {
    "FIFA": "FIFA World Cup 2026",
}

# Targeted search queries run in addition to subreddit feeds.
# Each hits the public Reddit search RSS — no API auth required.
# 25 posts per query × ~40 queries ≈ 1000 extra posts per cycle.
SEARCH_QUERIES: list[str] = [
    # Tournament
    "world cup 2026",
    "FIFA world cup 2026",
    "WC2026",
    "worldcup2026",
    # Top teams
    "Argentina world cup 2026",
    "Brazil world cup 2026",
    "France world cup 2026",
    "England world cup 2026",
    "Germany world cup 2026",
    "Spain world cup 2026",
    "Portugal world cup 2026",
    "Netherlands world cup 2026",
    "Morocco world cup 2026",
    "USA world cup 2026",
    "USMNT 2026",
    "Mexico world cup 2026",
    "Croatia world cup 2026",
    "Japan world cup 2026",
    "Canada world cup 2026",
    "Nigeria world cup 2026",
    "Senegal world cup 2026",
    "Turkey world cup 2026",
    "Colombia world cup 2026",
    "Uruguay world cup 2026",
    "Ecuador world cup 2026",
    "South Korea world cup 2026",
    "Australia world cup 2026",
    # Players
    "Mbappe world cup",
    "Messi world cup 2026",
    "Vinicius world cup",
    "Bellingham world cup",
    "Pedri world cup",
    "Yamal world cup",
    "Pulisic world cup",
    "Kane world cup",
    # Match events
    "world cup 2026 prediction",
    "world cup 2026 group stage",
    "world cup 2026 squad",
    "world cup 2026 qualifier",
]

SEARCH_SLEEP_SECONDS = 5


USER_AGENT = "worldcup-project/1.0 student research"
REQUEST_TIMEOUT_SECONDS = 30
RSS_MAX_RETRIES = 4
RSS_RETRY_BASE_SECONDS = 15

logger = logging.getLogger(__name__)


def create_reddit_tables(conn: sqlite3.Connection) -> None:
    """Create Reddit raw data tables if they do not exist."""
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

    # Kept for schema compatibility; comment collection is currently disabled.
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
    """Convert simple RSS HTML summaries into plain text."""
    if not value:
        return ""

    value = unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_entry_timestamp(entry) -> float:
    """Return UTC Unix timestamp from a feedparser entry.

    Prefers feedparser's pre-parsed ``published_parsed`` / ``updated_parsed``
    fields (already normalised to UTC ``time.struct_time``). Falls back to
    ``datetime.now()`` only when both are missing.
    """
    parsed_struct = (
        getattr(entry, "published_parsed", None)
        or getattr(entry, "updated_parsed", None)
    )
    if parsed_struct is not None:
        try:
            return float(calendar.timegm(parsed_struct))
        except Exception:
            logger.warning(
                "Failed to convert parsed RSS timestamp for entry id=%s",
                getattr(entry, "id", "unknown"),
            )

    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if raw:
        logger.warning(
            "Entry id=%s has no parsed timestamp — using collection time as fallback.",
            getattr(entry, "id", "unknown"),
        )

    return datetime.now(timezone.utc).timestamp()


def extract_post_id_from_link(link: str) -> str | None:
    """Extract Reddit post id from links like /comments/abc123/."""
    if not link:
        return None

    match = re.search(r"/comments/([^/]+)/", link)
    if not match:
        return None

    return match.group(1)


def stable_id(prefix: str, value: str) -> str:
    """Generate stable IDs when Reddit does not provide a clean post id."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _looks_like_feed_body(text: str) -> bool:
    """Return True when response text appears to be RSS/Atom XML."""
    stripped = text.lstrip()
    return (
        stripped.startswith("<?xml")
        or stripped.startswith("<feed")
        or stripped.startswith("<rss")
    )


def fetch_rss(url: str):
    """Fetch RSS with retries and exponential backoff on HTTP 429."""
    last_error: Exception | None = None

    for attempt in range(RSS_MAX_RETRIES):
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 200:
            return feedparser.parse(response.text)

        if response.status_code == 429:
            wait_seconds = RSS_RETRY_BASE_SECONDS * (2 ** attempt)
            logger.warning(
                "Rate limited (429) for %s — retry %d/%d in %ds.",
                url,
                attempt + 1,
                RSS_MAX_RETRIES,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            last_error = RuntimeError(
                f"RSS request failed. status=429, url={url}, body={response.text[:200]}"
            )
            continue

        # Reddit sometimes returns 403 (or other non-200) but still includes a valid feed body.
        if _looks_like_feed_body(response.text):
            feed = feedparser.parse(response.text)
            if getattr(feed, "entries", None):
                logger.warning(
                    "RSS returned HTTP %d for %s but body has %d entries — using feed anyway.",
                    response.status_code,
                    url,
                    len(feed.entries),
                )
                return feed

        raise RuntimeError(
            f"RSS request failed. status={response.status_code}, url={url}, "
            f"body={response.text[:200]}"
        )

    raise last_error or RuntimeError(f"RSS request failed after retries: {url}")


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


def _insert_entries_from_feed(
    conn: sqlite3.Connection,
    feed,
    subreddit_label: str,
    collected_at: str,
    *,
    posts_only: bool = False,
) -> int:
    """Insert up to POST_LIMIT_PER_SUBREDDIT entries from a parsed feed."""
    posts_inserted = 0

    for entry in feed.entries:
        if posts_inserted >= POST_LIMIT_PER_SUBREDDIT:
            break

        link = getattr(entry, "link", "")
        if posts_only and "/comments/" not in link:
            continue

        post_id = extract_post_id_from_link(link)
        if not post_id:
            if posts_only:
                continue
            post_id = stable_id("rss_post", getattr(entry, "id", "") or link)

        title = strip_html(getattr(entry, "title", ""))
        body = strip_html(getattr(entry, "summary", ""))
        author = getattr(entry, "author", None)
        created_utc = parse_entry_timestamp(entry)

        insert_post(
            conn=conn,
            post_id=post_id,
            subreddit=subreddit_label,
            title=title,
            body=body,
            author=author,
            score=0,
            created_utc=created_utc,
            collected_at=collected_at,
        )
        posts_inserted += 1

    return posts_inserted


def _collect_from_search_fallback(
    conn: sqlite3.Connection,
    subreddit_name: str,
    query: str,
    collected_at: str,
) -> int:
    """Collect posts via Reddit search RSS when a subreddit feed is blocked."""
    search_url = f"https://www.reddit.com/search.rss?q={quote(query)}&sort=new"
    logger.info(
        "r/%s direct RSS unavailable — using search RSS fallback (query=%r).",
        subreddit_name,
        query,
    )
    feed = fetch_rss(search_url)
    posts_inserted = _insert_entries_from_feed(
        conn,
        feed,
        subreddit_name,
        collected_at,
        posts_only=True,
    )
    conn.commit()
    logger.info(
        "Finished r/%s search fallback: posts=%s",
        subreddit_name,
        posts_inserted,
    )
    return posts_inserted

def collect_from_search_queries(conn: sqlite3.Connection) -> int:
    """Run all SEARCH_QUERIES against Reddit search RSS. Returns total posts inserted."""
    total = 0
    collected_at = datetime.now(timezone.utc).isoformat()

    for index, query in enumerate(SEARCH_QUERIES):
        url = f"https://www.reddit.com/search.rss?q={quote(query)}&sort=new&t=week"
        try:
            feed = fetch_rss(url)
            inserted = _insert_entries_from_feed(
                conn, feed, f"search:{query[:30]}", collected_at, posts_only=True
            )
            conn.commit()
            total += inserted
            logger.info("Search query %r → %d post(s).", query, inserted)
        except Exception:
            logger.exception("Search RSS failed for query %r", query)

        if index < len(SEARCH_QUERIES) - 1:
            time.sleep(SEARCH_SLEEP_SECONDS)

    logger.info("Search query collection finished. total_posts=%d", total)
    return total



def collect_from_subreddit(conn: sqlite3.Connection, subreddit_name: str) -> int:
    """Collect posts from one subreddit RSS feed. Returns posts attempted."""
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
            if getattr(feed, "entries", None):
                logger.info("RSS fetch succeeded for r/%s using %s", subreddit_name, rss_url)
                break
            logger.warning("RSS feed for %s returned zero entries.", rss_url)
            feed = None
        except Exception as error:
            last_error = error
            logger.warning("RSS fetch failed for %s: %s", rss_url, error)

    if feed is not None and getattr(feed, "entries", None):
        posts_inserted = _insert_entries_from_feed(
            conn, feed, subreddit_name, collected_at
        )
        conn.commit()
        logger.info("Finished r/%s RSS collection: posts=%s", subreddit_name, posts_inserted)
        return posts_inserted

    fallback_query = SEARCH_FALLBACK_QUERIES.get(subreddit_name)
    if fallback_query:
        return _collect_from_search_fallback(
            conn, subreddit_name, fallback_query, collected_at
        )

    raise RuntimeError(f"Failed fetching RSS for r/{subreddit_name}: {last_error}")


def collect_reddit_data() -> int:
    """Collect Reddit posts from all configured subreddits. Returns total posts attempted."""
    total_posts = 0

    with sqlite3.connect(DB_PATH) as conn:
        create_reddit_tables(conn)

        for index, subreddit_name in enumerate(SUBREDDITS):
            try:
                posts_count = collect_from_subreddit(conn=conn, subreddit_name=subreddit_name)
                total_posts += posts_count
            except Exception:
                logger.exception("Failed collecting RSS data from r/%s", subreddit_name)

            if index < len(SUBREDDITS) - 1:
                time.sleep(SUBREDDIT_SLEEP_SECONDS)

        # NEW — run targeted search queries on top of subreddit feeds
        try:
            total_posts += collect_from_search_queries(conn)
        except Exception:
            logger.exception("Search query collection failed")

    logger.info(
        "Reddit RSS collection finished. total_posts=%s across %d subreddit(s).",
        total_posts,
        len(SUBREDDITS),
    )
    return total_posts
