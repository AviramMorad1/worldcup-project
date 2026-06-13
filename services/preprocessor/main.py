"""
main.py
-------
Preprocessor service entry point for the World Cup sentiment pipeline.

Every cycle:
  1. Processes new raw posts from all sources (Reddit, Telegram posts, Telegram comments)
     → processed_posts  (VADER + TextBlob + detected_team stored per post)
  2. Backfills detected_team and textblob scores for legacy rows.
  3. Recomputes team_sentiment_daily aggregations.
  4. Recomputes trending_words — unigrams + bigrams, with category labels,
     filtered by expanded HARD_STOPWORDS so generic boilerplate does not appear.

Telegram comment team-inheritance rule:
  detect team from parent post title/body first; fallback to comment body.
  This means short comments like "Ooh no" or "Dem all hurt" still contribute
  to the correct national team's sentiment.
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
                    "griezmann", "cherki", "camavinga", "kounde", "dembele"],
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
        "Canada":        ["canada", "canmnt", "canadians", "alphonso davies",
                      "jonathan david", "larin", "buchanan"],
    "Turkey":        ["turkey", "turkiye", "turkish", "calhanoglu", "guler",
                      "yildiz", "demiral", "akaydin"],
    "Nigeria":       ["nigeria", "super eagles", "osimhen", "iheanacho",
                      "lookman", "troost ekong", "ndidi"],
    "Ecuador":       ["ecuador", "tri color", "tricolor", "caicedo",
                      "enner valencia", "plata"],
    "Saudi Arabia":  ["saudi arabia", "saudi", "green falcons", "al dawsari",
                      "al shahrani", "al malki"],
    "Iran":          ["iran", "team melli", "taremi", "azmoun", "jahanbakhsh"],
    "Ivory Coast":   ["ivory coast", "cote d ivoire", "elephants", "zaha",
                      "pepe", "gradel", "kessie"],
    "Egypt":         ["egypt", "pharaohs", "salah", "el shahat", "trezeguet"],
    "Cameroon":      ["cameroon", "indomitable lions", "aboubakar",
                      "anguissa", "choupo moting"],
    "Serbia":        ["serbia", "orlovi", "eagles", "mitrovic", "tadic",
                      "vlahovic", "milinkovic"],
    "Denmark":       ["denmark", "danish dynamite", "eriksen", "hojbjerg",
                      "maehle", "kjaer"],
    "Poland":        ["poland", "bialo czerwoni", "lewandowski", "szczesny",
                      "zielinski"],
    "Austria":       ["austria", "team austria", "alaba", "arnautovic",
                      "sabitzer"],
    "South Africa":  ["south africa", "bafana bafana", "percy tau",
                      "ronwen williams"],
    "Venezuela":     ["venezuela", "vinotinto", "soteldo", "bello", "romo"],
    "Panama":        ["panama", "los canaleros", "godoy", "carrasquilla"],
    "Jamaica":       ["jamaica", "reggae boyz", "antonio", "nicholson"],

}

# ---------------------------------------------------------------------------
# HARD_STOPWORDS — always removed from trending words
# Includes standard English + football/social-media boilerplate
# ---------------------------------------------------------------------------

HARD_STOPWORDS: frozenset[str] = frozenset({
    # Standard English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "this", "that", "was", "are", "be", "as",
    "by", "from", "they", "we", "he", "she", "his", "her", "their",
    "have", "has", "had", "will", "would", "not", "no", "so", "if",
    "my", "our", "your", "do", "did", "can", "could", "just", "up",
    "about", "out", "what", "who", "how", "all", "more", "also", "get",
    "one", "i", "you", "s", "t", "re", "ve", "ll", "d", "m", "its",
    "than", "then", "when", "which", "there", "been", "were", "am",
    "us", "him", "them", "me", "like", "got", "let", "know", "think",
    "see", "say", "going", "go", "make", "want", "really", "very",
    "well", "time", "way", "come", "back", "even", "too", "still",
    "people", "things", "thing", "new", "first", "last", "every",
    "after", "before", "same", "much", "many", "any", "some", "other",
    "those", "these", "should", "said", "into", "over", "between",
    "through", "against", "without", "because", "while", "again",
    "few", "need", "never", "always", "already", "been", "being",
    "having", "doing", "getting", "making", "taking", "putting",
    # Generic football/World Cup boilerplate — too common to be informative
    "football", "soccer", "worldcup", "world", "cup", "fifa",
    "match", "matches", "game", "games",
    "team", "teams", "player", "players",
    "fans", "fan",
    # Social media / platform boilerplate
    "comment", "comments", "post", "posts", "link", "submitted",
    "thread", "threads", "reddit", "telegram", "twitter", "channel",
    "official", "news", "update", "updates",
    "today", "yesterday", "tomorrow", "now", "live",
    "video", "photo", "image", "watch", "click",
    "http", "https", "com", "www", "amp", "rt", "via",
    "follow", "subscribe", "share", "repost", "source",
    "pattern", "new", "check",
    # URL / handle fragments
    "co", "me", "utm", "ref", "status",
    # Single/double character tokens caught at token-length filter, but explicit here
    "fc", "sc", "ac", "cf",
    "research", "consulting", "market", "insights", "firms",
    "crypto", "betting", "sportsbook", "fanduel", "draftkings",
    "poker", "promo", "odds", "wager", "parlay",
    "buy", "sale", "discount", "shop", "store",
    "attractor", "bsa", "sri", "lanka", "peacock",
})

# Keep backward-compatible alias for any legacy code that imported STOPWORDS
STOPWORDS = HARD_STOPWORDS

# ---------------------------------------------------------------------------
# DOMAIN_KEYWORDS — kept in trending words and labeled by category
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, str] = {
    # sentiment_positive
    "win":           "sentiment_positive",
    "winning":       "sentiment_positive",
    "won":           "sentiment_positive",
    "great":         "sentiment_positive",
    "amazing":       "sentiment_positive",
    "strong":        "sentiment_positive",
    "deserved":      "sentiment_positive",
    "brilliant":     "sentiment_positive",
    "love":          "sentiment_positive",
    "happy":         "sentiment_positive",
    "vamos":         "sentiment_positive",
    "respect":       "sentiment_positive",
    "goat":          "sentiment_positive",
    "legend":        "sentiment_positive",
    "proud":         "sentiment_positive",
    "excellent":     "sentiment_positive",
    "solid":         "sentiment_positive",
    "fire":          "sentiment_positive",
    "best":          "sentiment_positive",
    "incredible":    "sentiment_positive",
    "class":         "sentiment_positive",
    "quality":       "sentiment_positive",
    "superb":        "sentiment_positive",
    "perfect":       "sentiment_positive",
    "wow":           "sentiment_positive",
    "unstoppable":   "sentiment_positive",
    "exciting":      "sentiment_positive",
    # sentiment_negative
    "bad":           "sentiment_negative",
    "terrible":      "sentiment_negative",
    "awful":         "sentiment_negative",
    "weak":          "sentiment_negative",
    "poor":          "sentiment_negative",
    "disaster":      "sentiment_negative",
    "fraud":         "sentiment_negative",
    "disappointed":  "sentiment_negative",
    "angry":         "sentiment_negative",
    "embarrassing":  "sentiment_negative",
    "trash":         "sentiment_negative",
    "finished":      "sentiment_negative",
    "overrated":     "sentiment_negative",
    "bottling":      "sentiment_negative",
    "bottle":        "sentiment_negative",
    "worst":         "sentiment_negative",
    "pathetic":      "sentiment_negative",
    "useless":       "sentiment_negative",
    "boring":        "sentiment_negative",
    "robbery":       "sentiment_negative",
    "unlucky":       "sentiment_negative",
    "disappointing": "sentiment_negative",
    "frustrating":   "sentiment_negative",
    "horrible":      "sentiment_negative",
    "disgrace":      "sentiment_negative",
    "wasted":        "sentiment_negative",
    # injury_concern
    "injury":        "injury_concern",
    "injured":       "injury_concern",
    "hurt":          "injury_concern",
    "pain":          "injury_concern",
    "knock":         "injury_concern",
    "doubt":         "injury_concern",
    "missing":       "injury_concern",
    "fitness":       "injury_concern",
    "recovery":      "injury_concern",
    "hamstring":     "injury_concern",
    "knee":          "injury_concern",
    "ankle":         "injury_concern",
    "unavailable":   "injury_concern",
    "sidelined":     "injury_concern",
    "ruled":         "injury_concern",
    "comeback":      "injury_concern",
    "out":           "injury_concern",
    "absence":       "injury_concern",
    # squad_selection
    "squad":         "squad_selection",
    "lineup":        "squad_selection",
    "roster":        "squad_selection",
    "selection":     "squad_selection",
    "selected":      "squad_selection",
    "dropped":       "squad_selection",
    "benched":       "squad_selection",
    "starter":       "squad_selection",
    "starting":      "squad_selection",
    "substitute":    "squad_selection",
    "subs":          "squad_selection",
    "callup":        "squad_selection",
    "called":        "squad_selection",
    "included":      "squad_selection",
    "excluded":      "squad_selection",
    "pick":          "squad_selection",
    "picks":         "squad_selection",
    "omitted":       "squad_selection",
    # player_mention
    "messi":         "player_mention",
    "ronaldo":       "player_mention",
    "mbappe":        "player_mention",
    "neymar":        "player_mention",
    "vinicius":      "player_mention",
    "vini":          "player_mention",
    "bellingham":    "player_mention",
    "kane":          "player_mention",
    "saka":          "player_mention",
    "foden":         "player_mention",
    "musiala":       "player_mention",
    "kroos":         "player_mention",
    "modric":        "player_mention",
    "haaland":       "player_mention",
    "salah":         "player_mention",
    "son":           "player_mention",
    "pulisic":       "player_mention",
    "davies":        "player_mention",
    "hakimi":        "player_mention",
    "osimhen":       "player_mention",
    "oshimen":       "player_mention",
    "valverde":      "player_mention",
    "pedri":         "player_mention",
    "yamal":         "player_mention",
    "kounde":        "player_mention",
    "cherki":        "player_mention",
    "griezmann":     "player_mention",
    "dembele":       "player_mention",
    "camavinga":     "player_mention",
    "salisu":        "player_mention",
    "kudus":         "player_mention",
    "ayew":          "player_mention",
    "partey":        "player_mention",
    "rashford":      "player_mention",
    "giroud":        "player_mention",
    "deschamps":     "player_mention",
    "lukaku":        "player_mention",
    "de bruyne":     "player_mention",
    "muller":        "player_mention",
    "neuer":         "player_mention",
    "ochoa":         "player_mention",
    "jimenez":       "player_mention",
    "lozano":        "player_mention",
    "southgate":     "player_mention",
    "scaloni":       "player_mention",
    # tactical_performance
    "defense":       "tactical_performance",
    "attack":        "tactical_performance",
    "midfield":      "tactical_performance",
    "goalkeeper":    "tactical_performance",
    "keeper":        "tactical_performance",
    "pressing":      "tactical_performance",
    "tactics":       "tactical_performance",
    "pace":          "tactical_performance",
    "possession":    "tactical_performance",
    "finishing":     "tactical_performance",
    "scoring":       "tactical_performance",
    "goal":          "tactical_performance",
    "goals":         "tactical_performance",
    "assist":        "tactical_performance",
    "assists":       "tactical_performance",
    "penalty":       "tactical_performance",
    "penalties":     "tactical_performance",
    "counter":       "tactical_performance",
    "formation":     "tactical_performance",
    "blocks":        "tactical_performance",
    "saves":         "tactical_performance",
    "dribble":       "tactical_performance",
    "dribbling":     "tactical_performance",
    "header":        "tactical_performance",
    "shot":          "tactical_performance",
    "shots":         "tactical_performance",
    # competition_context
    "qualifier":     "competition_context",
    "qualifiers":    "competition_context",
    "group":         "competition_context",
    "final":         "competition_context",
    "semi":          "competition_context",
    "knockout":      "competition_context",
    "draw":          "competition_context",
    "playoff":       "competition_context",
    "semifinal":     "competition_context",
    "quarterfinal":  "competition_context",
    "round":         "competition_context",
    "stage":         "competition_context",
    "tournament":    "competition_context",
    "champion":      "competition_context",
    "champions":     "competition_context",
    "trophy":        "competition_context",
    "title":         "competition_context",
    "friendly":      "competition_context",
    "qualifier":     "competition_context",
}


def categorize_word(word: str) -> str:
    """Return the category for *word*, or 'other' if not in DOMAIN_KEYWORDS."""
    return DOMAIN_KEYWORDS.get(word.lower(), "other")


def _extract_bigrams(tokens: list[str]) -> list[str]:
    """
    Generate meaningful bigrams from a cleaned token list.

    Both tokens must:
    - be longer than 2 characters
    - be alphabetic
    - not be in HARD_STOPWORDS
    """
    bigrams: list[str] = []
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if (
            len(w1) > 2 and len(w2) > 2
            and w1.isalpha() and w2.isalpha()
            and w1 not in HARD_STOPWORDS
            and w2 not in HARD_STOPWORDS
        ):
            bigrams.append(f"{w1} {w2}")
    return bigrams


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
    """Return column names for a table, or [] if it does not exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()  # noqa: S608
            return [row[1] for row in rows]
    except sqlite3.Error:
        return []


def create_output_tables() -> None:
    """Create output tables if not present, then run safe column migrations."""
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
                team       TEXT,
                date       TEXT,
                word       TEXT,
                frequency  INTEGER,
                category   TEXT DEFAULT 'other',
                ngram_type TEXT DEFAULT 'unigram'
            )
        """)
        conn.commit()
    logger.info("Output tables created / verified.")
    _migrate_processed_posts()
    _migrate_trending_words()


def _migrate_processed_posts() -> None:
    """Add columns to processed_posts that older versions did not include."""
    cols = _get_table_columns("processed_posts")
    if not cols:
        return
    migrations = [
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
            conn.execute(
                "UPDATE processed_posts SET source='reddit' WHERE source IS NULL"
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning("processed_posts migration failed: %s", exc)


def _migrate_trending_words() -> None:
    """Add category and ngram_type columns to trending_words if missing."""
    cols = _get_table_columns("trending_words")
    if not cols:
        return
    migrations = [
        ("category",   "ALTER TABLE trending_words ADD COLUMN category TEXT DEFAULT 'other'"),
        ("ngram_type", "ALTER TABLE trending_words ADD COLUMN ngram_type TEXT DEFAULT 'unigram'"),
    ]
    try:
        with sqlite3.connect(DB_PATH) as conn:
            for col_name, sql in migrations:
                if col_name not in cols:
                    conn.execute(sql)
                    conn.commit()
                    logger.info("Migrated trending_words: added '%s' column.", col_name)
    except sqlite3.Error as exc:
        logger.warning("trending_words migration failed: %s", exc)


# ---------------------------------------------------------------------------
# Team detection
# ---------------------------------------------------------------------------


def detect_team(text: str) -> str | None:
    """Return the first matching team name found in *text*, or None."""
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

    For Telegram comments we JOIN parent posts so the team-detection step
    can inherit team context from the parent post body.
    """
    frames: list[pd.DataFrame] = []

    if table_exists("raw_reddit_posts"):
        q = """
            SELECT r.id, r.title, r.body, r.created_utc,
                   'reddit' AS source,
                   NULL     AS parent_title,
                   NULL     AS parent_body
            FROM raw_reddit_posts AS r
            LEFT JOIN processed_posts AS p ON r.id = p.post_id
            WHERE p.post_id IS NULL
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                frames.append(pd.read_sql_query(q, conn))
        except Exception as exc:
            logger.warning("Failed to load unprocessed Reddit posts: %s", exc)

    if table_exists("raw_telegram_posts"):
        q = """
            SELECT t.id, t.title, t.body, t.created_utc,
                   'telegram' AS source,
                   NULL       AS parent_title,
                   NULL       AS parent_body
            FROM raw_telegram_posts AS t
            LEFT JOIN processed_posts AS p ON t.id = p.post_id
            WHERE p.post_id IS NULL
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                frames.append(pd.read_sql_query(q, conn))
        except Exception as exc:
            logger.warning("Failed to load unprocessed Telegram posts: %s", exc)

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
                frames.append(pd.read_sql_query(q, conn))
        except Exception as exc:
            logger.warning("Failed to load unprocessed Telegram comments: %s", exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).head(limit)
    if not combined.empty and "source" in combined.columns:
        counts = combined["source"].value_counts().to_dict()
        logger.info(
            "Loaded %d unprocessed row(s): %s",
            len(combined),
            ", ".join(f"{s}={n}" for s, n in counts.items()),
        )
    return combined


# ---------------------------------------------------------------------------
# Post processing  (VADER + TextBlob + detected_team stored per post)
# ---------------------------------------------------------------------------


def process_posts(posts_df: pd.DataFrame) -> int:
    """
    Run sentiment analysis and team detection on all rows in posts_df.

    Team-detection strategy:
      reddit / telegram    → detect from title + body
      telegram_comment     → detect from parent post title/body first;
                             fallback to comment body
    """
    if posts_df.empty:
        return 0

    processed = 0
    inherited_team_count = 0
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        for _, row in posts_df.iterrows():
            post_id  = str(row.get("id", ""))
            title    = row.get("title", "") or ""
            body     = row.get("body",  "") or ""
            source   = str(row.get("source", "reddit") or "reddit")
            raw_text = f"{title} {body}".strip()

            cleaned   = clean_text(raw_text)
            tokens    = tokenize_text(cleaned)
            sentiment = get_basic_sentiment(cleaned)

            detected_team: str | None = None
            if source == "telegram_comment":
                parent_text = (
                    (row.get("parent_title", "") or "") + " " +
                    (row.get("parent_body",  "") or "")
                ).strip()
                if parent_text:
                    detected_team = detect_team(parent_text)
                    if detected_team:
                        inherited_team_count += 1
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
                        post_id, source, detected_team, cleaned, json.dumps(tokens),
                        sentiment["compound"], sentiment["positive"],
                        sentiment["negative"], sentiment["neutral"],
                        sentiment["textblob_polarity"], sentiment["textblob_subjectivity"],
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
    """Backfill textblob scores for rows processed before TextBlob was added."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT post_id, cleaned_text FROM processed_posts "
                "WHERE textblob_polarity IS NULL OR textblob_subjectivity IS NULL"
            ).fetchall()
            if not rows:
                logger.info("backfill_textblob: nothing to backfill.")
                return
            logger.info("backfill_textblob: backfilling %d row(s).", len(rows))
            for post_id, cleaned_text in rows:
                s = get_basic_sentiment(cleaned_text or "")
                conn.execute(
                    "UPDATE processed_posts "
                    "SET textblob_polarity=?, textblob_subjectivity=? WHERE post_id=?",
                    (s["textblob_polarity"], s["textblob_subjectivity"], post_id),
                )
            conn.commit()
        logger.info("backfill_textblob: done.")
    except sqlite3.Error as exc:
        logger.warning("backfill_textblob failed: %s", exc)


def backfill_detected_team() -> None:
    """Fill detected_team for rows inserted before this column was added."""
    cols = _get_table_columns("processed_posts")
    if "detected_team" not in cols:
        return
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

    # telegram_comment rows: use parent post for team detection
    if table_exists("raw_telegram_comments") and table_exists("raw_telegram_posts"):
        try:
            q = """
                SELECT p.post_id, tp.title, tp.body, c.body AS comment_body
                FROM processed_posts p
                JOIN raw_telegram_comments c  ON c.id = p.post_id
                LEFT JOIN raw_telegram_posts tp ON c.parent_post_id = tp.id
                WHERE p.detected_team IS NULL AND p.source = 'telegram_comment'
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
            logger.info("backfill_detected_team: updated %d telegram_comment row(s).", updated)
        except Exception as exc:
            logger.warning("backfill_detected_team: telegram_comment pass failed: %s", exc)

    # Reddit and Telegram posts: join with raw union
    union_parts = []
    for tbl in ("raw_reddit_posts", "raw_telegram_posts"):
        if table_exists(tbl):
            union_parts.append(f"SELECT id, title, body FROM {tbl}")  # noqa: S608
    if not union_parts:
        return
    try:
        q = f"""
            SELECT p.post_id, r.title, r.body
            FROM processed_posts p
            LEFT JOIN ({" UNION ALL ".join(union_parts)}) r ON r.id = p.post_id
            WHERE p.detected_team IS NULL AND p.source != 'telegram_comment'
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
# Aggregation helpers
# ---------------------------------------------------------------------------


def _raw_union_sql() -> str:
    """UNION of (id, title, body, created_utc) across all raw post/comment tables."""
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
    n_fb = primary.isna().sum()
    if n_fb > 0:
        logger.warning("%s: %d row(s) use processed_at as date fallback.", fn_name, n_fb)

    resolved = primary.where(primary.notna(), fallback)
    df = df.copy()
    df["date"] = resolved.dt.date.astype(str)
    before = len(df)
    df = df[df["date"].notna() & (df["date"] != "NaT") & (df["date"] != "None")]
    dropped = before - len(df)
    if dropped:
        logger.warning("%s: dropped %d row(s) with invalid dates.", fn_name, dropped)
    return df


# ---------------------------------------------------------------------------
# Aggregation — team_sentiment_daily
# ---------------------------------------------------------------------------


def compute_team_sentiment_daily() -> None:
    """
    Recompute team_sentiment_daily from processed_posts.

    Uses stored detected_team (preferred). For legacy rows where detected_team
    IS NULL, falls back to on-the-fly detection from raw text.
    Rows with no detectable team are skipped from aggregation.
    """
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
        agg.groupby("date")["post_count"].max().rename("max_post_count").reset_index()
    )
    agg = agg.merge(daily_max, on="date")
    agg["hype_index"] = (
        (agg["post_count"] / agg["max_post_count"]) * agg["avg_vader"].abs()
    )


    rows = list(
        agg[["team", "date", "avg_vader", "avg_textblob",
             "post_count", "hype_index"]].itertuples(index=False, name=None)
    )

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM team_sentiment_daily")
            conn.executemany(
                "INSERT INTO team_sentiment_daily "
                "(team, date, avg_vader, avg_textblob, post_count, hype_index) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        logger.info(
            "team_sentiment_daily: wrote %d row(s) across %d team(s).",
            len(rows), agg["team"].nunique(),
        )
    except sqlite3.Error as exc:
        logger.warning("compute_team_sentiment_daily: write failed: %s", exc)


# ---------------------------------------------------------------------------
# Aggregation — trending_words  (unigrams + bigrams, with categories)
# ---------------------------------------------------------------------------


def compute_trending_words() -> None:
    """
    Recompute trending_words from tokens in processed_posts.

    Improvements:
    - HARD_STOPWORDS filters generic football/social-media boilerplate.
    - DOMAIN_KEYWORDS preserves meaningful terms with category labels.
    - Bigrams extracted for phrases like "starting lineup", "injury concern".
    - category and ngram_type stored if columns exist.
    """
    tw_cols = _get_table_columns("trending_words")
    has_category   = "category"   in tw_cols
    has_ngram_type = "ngram_type" in tw_cols

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

    # Team assignment: prefer stored detected_team, fallback to text detection
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

    freq_uni: dict[tuple[str, str], Counter] = {}
    freq_bi:  dict[tuple[str, str], Counter] = {}

    for _, row in df.iterrows():
        team = row["team"]
        date = row["date"]
        key  = (team, date)

        try:
            tokens: list[str] = json.loads(row["tokens_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Unigrams — filter with HARD_STOPWORDS; keep DOMAIN_KEYWORDS
        meaningful = [
            t for t in tokens
            if t
            and len(t) > 2
            and t.isalpha()
            and t not in HARD_STOPWORDS
        ]
        if meaningful:
            if key not in freq_uni:
                freq_uni[key] = Counter()
            freq_uni[key].update(meaningful)

        # Bigrams
        bigrams = _extract_bigrams(tokens)
        if bigrams:
            if key not in freq_bi:
                freq_bi[key] = Counter()
            freq_bi[key].update(bigrams)

    rows: list[tuple] = []

    def _row(team: str, date: str, word: str, count: int, ngram: str) -> tuple:
        cat = categorize_word(word.split()[0])  # first token determines category
        if has_category and has_ngram_type:
            return (team, date, word, count, cat, ngram)
        if has_category:
            return (team, date, word, count, cat)
        return (team, date, word, count)

    for (team, date), counter in freq_uni.items():
        for word, count in counter.most_common(60):
            rows.append(_row(team, date, word, count, "unigram"))

    for (team, date), counter in freq_bi.items():
        for phrase, count in counter.most_common(20):
            rows.append(_row(team, date, phrase, count, "bigram"))

    if not rows:
        logger.info("compute_trending_words: no word frequencies to write.")
        return

    if has_category and has_ngram_type:
        insert_sql = (
            "INSERT INTO trending_words "
            "(team, date, word, frequency, category, ngram_type) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
    elif has_category:
        insert_sql = (
            "INSERT INTO trending_words "
            "(team, date, word, frequency, category) "
            "VALUES (?, ?, ?, ?, ?)"
        )
    else:
        insert_sql = (
            "INSERT INTO trending_words "
            "(team, date, word, frequency) "
            "VALUES (?, ?, ?, ?)"
        )
        rows = [(r[0], r[1], r[2], r[3]) for r in rows]

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM trending_words")
            conn.executemany(insert_sql, rows)
            conn.commit()
        logger.info(
            "trending_words: wrote %d row(s) (unigrams + bigrams).", len(rows)
        )
    except sqlite3.Error as exc:
        logger.warning("compute_trending_words: write failed: %s", exc)


# ---------------------------------------------------------------------------
# Collector coordination
# ---------------------------------------------------------------------------


def _raw_post_count() -> int:
    total = 0
    for tbl in ("raw_reddit_posts", "raw_telegram_posts"):
        if table_exists(tbl):
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM {tbl}"  # noqa: S608
                    ).fetchone()
                    total += int(row[0]) if row else 0
            except sqlite3.Error:
                pass
    return total


def wait_for_collector() -> None:
    """Wait until the collector finishes its first cycle or posts exist."""
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
        "PREPROCESS_INTERVAL_MINUTES", str(DEFAULT_PREPROCESS_INTERVAL_MINUTES)
    )
    try:
        minutes = max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid PREPROCESS_INTERVAL_MINUTES=%r — using default %d.",
            raw, DEFAULT_PREPROCESS_INTERVAL_MINUTES,
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

    posts_df = load_unprocessed_posts()
    processed_count = 0
    if posts_df.empty:
        logger.info("No unprocessed posts found in any source.")
    else:
        processed_count = process_posts(posts_df)
    logger.info("Processed %d new row(s) this cycle.", processed_count)

    backfill_textblob()
    backfill_detected_team()
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

    PREPROCESS_TRIGGER_FLAG = "/app/data/preprocess_trigger.flag"

    while True:
        try:
            run_preprocessing_cycle()
        except Exception as exc:
            logger.exception("Preprocessing cycle failed: %s", exc)

        logger.info(
            "Sleeping for %d minutes before next preprocessing cycle.", interval_minutes
        )
        elapsed = 0
        while elapsed < interval_seconds:
            time.sleep(10)
            elapsed += 10
            if os.path.exists(PREPROCESS_TRIGGER_FLAG):
                try:
                    os.remove(PREPROCESS_TRIGGER_FLAG)
                except OSError:
                    pass
                logger.info("Dashboard trigger detected — running immediate cycle.")
                break



if __name__ == "__main__":
    main()
