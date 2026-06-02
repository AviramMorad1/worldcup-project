"""
telegram_collector.py
---------------------
Optional Telegram public-channel collector for the World Cup sentiment pipeline.

Collects two levels of data (both optional):
  1. Channel posts   → raw_telegram_posts   (always when Telegram is configured)
  2. Post comments   → raw_telegram_comments (only when TELEGRAM_COLLECT_COMMENTS=true)

Collection is skipped gracefully if:
  - TELEGRAM_API_ID or TELEGRAM_API_HASH are not set
  - TELEGRAM_CHANNELS is empty

Comment collection notes:
  - Telegram channel comments live in a linked "discussion group".
    Telethon fetches them via reply_to=<post_id> on the same channel entity.
  - Only channels with comments enabled (linked group) will return results.
  - Private discussion groups that the account cannot access are skipped silently.
  - Limits are intentionally conservative to avoid rate-limit bans.

First-run authentication:
  docker compose run --rm collector python /app/telegram_auth.py
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

TELEGRAM_API_ID: str | None = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH: str | None = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_SESSION_NAME: str = os.getenv("TELEGRAM_SESSION_NAME", "worldcup_telegram")
TELEGRAM_CHANNELS: list[str] = [
    c.strip() for c in os.getenv("TELEGRAM_CHANNELS", "").split(",") if c.strip()
]
TELEGRAM_LIMIT_PER_CHANNEL: int = int(os.getenv("TELEGRAM_LIMIT_PER_CHANNEL", "50"))

# Comment collection — disabled by default; enable with TELEGRAM_COLLECT_COMMENTS=true
TELEGRAM_COLLECT_COMMENTS: bool = (
    os.getenv("TELEGRAM_COLLECT_COMMENTS", "false").strip().lower() == "true"
)
# How many of the most recent posts to fetch comments for (per channel)
TELEGRAM_COMMENT_POST_LIMIT_PER_CHANNEL: int = int(
    os.getenv("TELEGRAM_COMMENT_POST_LIMIT_PER_CHANNEL", "5")
)
# Max comment replies to fetch per post
TELEGRAM_COMMENT_LIMIT_PER_POST: int = int(
    os.getenv("TELEGRAM_COMMENT_LIMIT_PER_POST", "30")
)

# Store session files in the persistent Docker volume so login survives restarts
TELEGRAM_SESSION_DIR: str = "/app/data/telegram_sessions"

DB_PATH: str = "/app/data/worldcup.db"

# Rate-limiting: sleep between channels / posts to avoid flooding
CHANNEL_SLEEP_SECONDS: int = 3
POST_COMMENT_SLEEP_SECONDS: int = 1


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _create_table(conn: sqlite3.Connection) -> None:
    """Create raw_telegram_posts if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_telegram_posts (
            id          TEXT PRIMARY KEY,
            source      TEXT DEFAULT 'telegram',
            channel     TEXT,
            message_id  INTEGER,
            title       TEXT,
            body        TEXT,
            author      TEXT,
            views       INTEGER,
            forwards    INTEGER,
            created_utc REAL,
            collected_at TEXT
        )
        """
    )
    conn.commit()


def _create_comments_table(conn: sqlite3.Connection) -> None:
    """Create raw_telegram_comments if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_telegram_comments (
            id              TEXT PRIMARY KEY,
            source          TEXT DEFAULT 'telegram_comment',
            channel         TEXT,
            parent_post_id  TEXT,
            message_id      INTEGER,
            reply_to_msg_id INTEGER,
            body            TEXT,
            author          TEXT,
            created_utc     REAL,
            collected_at    TEXT
        )
        """
    )
    conn.commit()


def _insert_message(
    conn: sqlite3.Connection,
    *,
    post_id: str,
    channel: str,
    message_id: int,
    title: str | None,
    body: str,
    author: str | None,
    views: int | None,
    forwards: int | None,
    created_utc: float | None,
    collected_at: str,
) -> int:
    """Insert a single message; returns 1 if inserted, 0 if already exists."""
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO raw_telegram_posts
            (id, source, channel, message_id, title, body, author,
             views, forwards, created_utc, collected_at)
        VALUES (?, 'telegram', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            post_id, channel, message_id, title, body, author,
            views, forwards, created_utc, collected_at,
        ),
    )
    return cursor.rowcount  # 1 = inserted, 0 = ignored (duplicate)


def _insert_comment(
    conn: sqlite3.Connection,
    *,
    comment_id: str,
    channel: str,
    parent_post_id: str,
    message_id: int,
    reply_to_msg_id: int,
    body: str,
    author: str | None,
    created_utc: float | None,
    collected_at: str,
) -> int:
    """Insert a single comment; returns 1 if inserted, 0 if already exists."""
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO raw_telegram_comments
            (id, source, channel, parent_post_id, message_id, reply_to_msg_id,
             body, author, created_utc, collected_at)
        VALUES (?, 'telegram_comment', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            comment_id, channel, parent_post_id, message_id, reply_to_msg_id,
            body, author, created_utc, collected_at,
        ),
    )
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Configuration check
# ---------------------------------------------------------------------------


def is_configured() -> bool:
    """Return True only when all required env vars are present and non-empty."""
    if not TELEGRAM_API_ID:
        logger.info("Telegram: TELEGRAM_API_ID not set — skipping Telegram collection.")
        return False
    if not TELEGRAM_API_HASH:
        logger.info("Telegram: TELEGRAM_API_HASH not set — skipping Telegram collection.")
        return False
    if not TELEGRAM_CHANNELS:
        logger.info("Telegram: TELEGRAM_CHANNELS is empty — skipping Telegram collection.")
        return False
    return True


# ---------------------------------------------------------------------------
# Channel collection
# ---------------------------------------------------------------------------


def _collect_channel(client, channel: str, conn: sqlite3.Connection) -> int:
    """
    Fetch the latest messages from *channel* and persist new ones.
    Returns the number of newly inserted messages.
    """
    collected_at = datetime.now(timezone.utc).isoformat()
    inserted = 0

    messages = client.get_messages(channel, limit=TELEGRAM_LIMIT_PER_CHANNEL)
    logger.info("Telegram: fetched %d message(s) from %s", len(messages), channel)

    for msg in messages:
        if not msg.text:
            # Skip media-only messages — no text to analyse
            continue

        post_id = f"{channel}:{msg.id}"
        created_utc = msg.date.timestamp() if msg.date else None
        title = msg.text[:120] if msg.text else None  # brief preview for title field

        inserted += _insert_message(
            conn,
            post_id=post_id,
            channel=channel,
            message_id=msg.id,
            title=title,
            body=msg.text,
            author=channel,                          # channel name as author
            views=getattr(msg, "views", None),
            forwards=getattr(msg, "forwards", None),
            created_utc=created_utc,
            collected_at=collected_at,
        )

    conn.commit()
    return inserted


def _collect_channel_comments(client, channel: str, conn: sqlite3.Connection) -> int:
    """
    Fetch comments (replies) for the most recent posts in *channel*.

    Uses Telethon's reply_to parameter which routes through the channel's
    linked discussion group when comments are enabled.  If the channel has no
    linked group, or the account cannot access it, the call returns an empty
    list and we log an info message — no exception is raised to the caller.

    Returns the number of newly inserted comments.
    """
    collected_at = datetime.now(timezone.utc).isoformat()
    total_inserted = 0

    # Fetch the most recent posts to look for comments on
    try:
        recent_posts = client.get_messages(
            channel, limit=TELEGRAM_COMMENT_POST_LIMIT_PER_CHANNEL
        )
    except Exception as exc:
        logger.warning(
            "Telegram comments: could not fetch recent posts from '%s': %s", channel, exc
        )
        return 0

    for post in recent_posts:
        if not post.id:
            continue

        parent_post_id = f"{channel}:{post.id}"

        try:
            # get_messages with reply_to fetches replies to that message id.
            # For channels with a linked discussion group Telethon resolves
            # the group transparently.  Returns [] if comments are unavailable.
            replies = client.get_messages(
                channel,
                reply_to=post.id,
                limit=TELEGRAM_COMMENT_LIMIT_PER_POST,
            )
        except Exception as exc:
            logger.info(
                "Telegram comments: '%s' post %d — could not fetch replies "
                "(channel may have no linked discussion group): %s",
                channel, post.id, exc,
            )
            time.sleep(POST_COMMENT_SLEEP_SECONDS)
            continue

        if not replies:
            time.sleep(POST_COMMENT_SLEEP_SECONDS)
            continue

        inserted_for_post = 0
        for reply in replies:
            if not reply.text:
                continue  # skip media-only replies

            comment_id = f"{channel}:{post.id}:{reply.id}"
            created_utc = reply.date.timestamp() if reply.date else None

            # Try to resolve sender username; fall back to channel name
            sender = None
            try:
                sender_obj = reply.sender
                if sender_obj:
                    sender = getattr(sender_obj, "username", None) or getattr(
                        sender_obj, "first_name", None
                    )
            except Exception:
                pass
            author = sender or channel

            inserted_for_post += _insert_comment(
                conn,
                comment_id=comment_id,
                channel=channel,
                parent_post_id=parent_post_id,
                message_id=reply.id,
                reply_to_msg_id=post.id,
                body=reply.text,
                author=author,
                created_utc=created_utc,
                collected_at=collected_at,
            )

        conn.commit()
        total_inserted += inserted_for_post
        logger.debug(
            "Telegram comments: post %s — %d new comment(s).", parent_post_id, inserted_for_post
        )
        time.sleep(POST_COMMENT_SLEEP_SECONDS)

    return total_inserted


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect_telegram_data() -> int:
    """
    Collect messages from all configured Telegram channels.

    Returns the total number of newly inserted messages.
    Skips gracefully if Telegram is not configured.
    """
    if not is_configured():
        return 0

    try:
        # Import lazily so the collector still boots if Telethon is not installed
        from telethon.sync import TelegramClient  # type: ignore[import]
    except ImportError:
        logger.error(
            "Telegram: telethon is not installed. "
            "Add it to collector/requirements.txt and rebuild."
        )
        return 0

    os.makedirs(TELEGRAM_SESSION_DIR, exist_ok=True)
    session_path = os.path.join(TELEGRAM_SESSION_DIR, TELEGRAM_SESSION_NAME)

    total_posts = 0
    total_comments = 0

    if TELEGRAM_COLLECT_COMMENTS:
        logger.info(
            "Telegram: comment collection ENABLED "
            "(post_limit=%d, comment_limit=%d per post).",
            TELEGRAM_COMMENT_POST_LIMIT_PER_CHANNEL,
            TELEGRAM_COMMENT_LIMIT_PER_POST,
        )
    else:
        logger.info("Telegram: comment collection disabled (TELEGRAM_COLLECT_COMMENTS not set).")

    try:
        # Note: int() conversion is safe here — is_configured() ensures API_ID is set
        with TelegramClient(session_path, int(TELEGRAM_API_ID), TELEGRAM_API_HASH) as client:
            with sqlite3.connect(DB_PATH) as conn:
                _create_table(conn)
                if TELEGRAM_COLLECT_COMMENTS:
                    _create_comments_table(conn)

                for channel in TELEGRAM_CHANNELS:
                    # --- Posts ---
                    try:
                        logger.info("Telegram: collecting posts from '%s' ...", channel)
                        n = _collect_channel(client, channel, conn)
                        total_posts += n
                        logger.info(
                            "Telegram: inserted %d new post(s) from '%s'.", n, channel
                        )
                    except Exception as exc:
                        logger.warning(
                            "Telegram: failed to collect posts from '%s': %s", channel, exc
                        )

                    # --- Comments (only when opted in) ---
                    if TELEGRAM_COLLECT_COMMENTS:
                        try:
                            logger.info(
                                "Telegram: collecting comments from '%s' ...", channel
                            )
                            nc = _collect_channel_comments(client, channel, conn)
                            total_comments += nc
                            logger.info(
                                "Telegram: inserted %d new comment(s) from '%s'.", nc, channel
                            )
                        except Exception as exc:
                            logger.warning(
                                "Telegram: failed to collect comments from '%s': %s",
                                channel, exc,
                            )

                    time.sleep(CHANNEL_SLEEP_SECONDS)

    except Exception as exc:
        logger.error("Telegram: client-level error: %s", exc)
        logger.info(
            "Telegram: if this is the first run, the session file may not exist yet. "
            "Authenticate interactively with: "
            "docker compose run --rm collector python /app/telegram_auth.py"
        )

    logger.info(
        "Telegram: collection complete. Posts: %d new, Comments: %d new.",
        total_posts,
        total_comments,
    )
    return total_posts + total_comments
