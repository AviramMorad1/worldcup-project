#!/usr/bin/env python3
"""
Lightweight integration validation for the World Cup analytics pipeline.

Checks SQLite tables, sentiment ranges, model artifacts, 2026 predictions,
and full Telegram source health after running `docker compose up`.

Usage:
  python scripts/integration_check.py
  DATABASE_PATH=/app/data/worldcup.db python scripts/integration_check.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution  (local + Docker)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DB_CANDIDATES = (
    "/app/data/worldcup.db",
    REPO_ROOT / "data" / "worldcup.db",
    REPO_ROOT / "worldcup.db",
)

MODEL_DIR_CANDIDATES = (
    "/app/data/models",
    REPO_ROOT / "data" / "models",
    REPO_ROOT / "models",
)

DASHBOARD_CANDIDATES = (
    REPO_ROOT / "services" / "dashboard" / "app.py",
    Path("/services/dashboard/app.py"),
    Path("/app/services/dashboard/app.py"),
)

CRITICAL_TABLES = (
    "raw_matches",
    "raw_rankings",
    "raw_reddit_posts",
    "processed_posts",
    "team_sentiment_daily",
    "trending_words",
    "match_predictions",
    "model_metrics",
)

OPTIONAL_TABLES = (
    "raw_reddit_comments",
    "team_elo",
)

LOW_VOLUME_THRESHOLD = 50
EXPECTED_PREDICTIONS_2026 = 72
METRIC_WARN_THRESHOLD = 0.50
# Minimum fraction of telegram_comment rows that should have a detected_team
TG_COMMENT_TEAM_PASS_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Report accumulator
# ---------------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.critical_failures = 0
        self.warnings = 0

    def pass_(self, message: str) -> None:
        self.lines.append(f"[PASS] {message}")

    def fail(self, message: str) -> None:
        self.critical_failures += 1
        self.lines.append(f"[FAIL] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        self.lines.append(f"[WARN] {message}")

    def info(self, message: str) -> None:
        self.lines.append(f"       {message}")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_db_path(report: Report) -> Path | None:
    env_path = os.environ.get("DATABASE_PATH")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            report.pass_(f"Database found (DATABASE_PATH): {path}")
            return path
        report.fail(f"DATABASE_PATH set but file not found: {path}")
        return None

    for candidate in DB_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            report.pass_(f"Database found: {path}")
            return path

    report.fail(f"Database not found. Tried: {', '.join(str(p) for p in DB_CANDIDATES)}")
    return None


def resolve_model_dir(report: Report) -> Path | None:
    env_dir = os.environ.get("MODEL_DIR")
    if env_dir:
        path = Path(env_dir)
        if path.is_dir():
            report.pass_(f"Model directory found (MODEL_DIR): {path}")
            return path
        report.fail(f"MODEL_DIR set but directory not found: {path}")
        return None

    for candidate in MODEL_DIR_CANDIDATES:
        path = Path(candidate)
        if path.is_dir():
            report.pass_(f"Model directory found: {path}")
            return path

    report.fail(
        f"Model directory not found. Tried: {', '.join(str(p) for p in MODEL_DIR_CANDIDATES)}"
    )
    return None


def open_db(db_path: Path) -> sqlite3.Connection | None:
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    if not table_exists(conn, table_name):
        return False
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_tables(conn: sqlite3.Connection, report: Report) -> None:
    for t in CRITICAL_TABLES:
        if table_exists(conn, t):
            report.pass_(f"Table exists: {t}")
        else:
            report.fail(f"Missing critical table: {t}")

    for t in OPTIONAL_TABLES:
        if table_exists(conn, t):
            report.pass_(f"Optional table exists: {t}")
        else:
            report.warn(f"Optional table missing: {t}")


def check_row_counts(conn: sqlite3.Connection, report: Report) -> None:
    for t in CRITICAL_TABLES:
        if not table_exists(conn, t):
            continue
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
        if count > 0:
            report.pass_(f"{t} rows: {count:,}")
        else:
            report.fail(f"{t} is empty")

        if t in {"raw_reddit_posts", "processed_posts"} and 0 < count < LOW_VOLUME_THRESHOLD:
            report.warn(
                f"{t} has only {count} rows "
                f"(expected >= {LOW_VOLUME_THRESHOLD} for a mature run)"
            )


def check_processed_posts_schema(conn: sqlite3.Connection, report: Report) -> None:
    """Verify that processed_posts has the required columns for multi-source support."""
    if not table_exists(conn, "processed_posts"):
        return

    for col in ("source", "detected_team"):
        if column_exists(conn, "processed_posts", col):
            report.pass_(f"processed_posts has '{col}' column")
        else:
            report.fail(f"processed_posts missing '{col}' column — run preprocessor to migrate")


def check_source_distribution(conn: sqlite3.Connection, report: Report) -> None:
    """Print source breakdown of processed_posts."""
    if not table_exists(conn, "processed_posts"):
        return
    if not column_exists(conn, "processed_posts", "source"):
        return

    rows = conn.execute(
        "SELECT source, COUNT(*) AS n FROM processed_posts "
        "GROUP BY source ORDER BY n DESC"
    ).fetchall()

    if not rows:
        return

    report.info("processed_posts source distribution:")
    for row in rows:
        report.info(f"  source={row['source'] or 'NULL':<20} rows={row['n']:,}")


def check_team_distribution(conn: sqlite3.Connection, report: Report) -> None:
    """Print top teams by source from processed_posts."""
    if not table_exists(conn, "processed_posts"):
        return
    if not column_exists(conn, "processed_posts", "detected_team"):
        return

    rows = conn.execute(
        """
        SELECT source, detected_team, COUNT(*) AS n
        FROM processed_posts
        WHERE detected_team IS NOT NULL
        GROUP BY source, detected_team
        ORDER BY source, n DESC
        """
    ).fetchall()

    if not rows:
        report.info("processed_posts: no rows with detected_team yet.")
        return

    report.info("Top teams by source (processed_posts):")
    prev_source = None
    for row in rows:
        src = row["source"] or "NULL"
        if src != prev_source:
            prev_source = src
            report.info(f"  [{src}]")
        report.info(f"    {row['detected_team']:<20} {row['n']:>5}")


def check_sentiment(conn: sqlite3.Connection, report: Report) -> None:
    if not table_exists(conn, "processed_posts"):
        return
    if not column_exists(conn, "processed_posts", "vader_compound"):
        report.fail("processed_posts missing column: vader_compound")
        return

    row = conn.execute(
        """
        SELECT COUNT(*) AS n,
               MIN(vader_compound) AS vmin,
               MAX(vader_compound) AS vmax
        FROM processed_posts
        WHERE vader_compound IS NOT NULL
        """
    ).fetchone()

    if row["n"] == 0:
        report.fail("processed_posts has no non-null vader_compound values")
        return

    vmin, vmax = row["vmin"], row["vmax"]
    if vmin >= -1.0 and vmax <= 1.0:
        report.pass_(f"vader_compound range: {vmin:.4f} to {vmax:.4f}")
    else:
        report.fail(f"vader_compound out of range [-1, 1]: {vmin:.4f} to {vmax:.4f}")

    if column_exists(conn, "processed_posts", "textblob_polarity"):
        tb = conn.execute(
            """
            SELECT COUNT(*) AS n,
                   MIN(textblob_polarity) AS vmin,
                   MAX(textblob_polarity) AS vmax
            FROM processed_posts
            WHERE textblob_polarity IS NOT NULL
            """
        ).fetchone()
        if tb["n"] == 0:
            report.warn("textblob_polarity is null for all rows")
        elif tb["vmin"] >= -1.0 and tb["vmax"] <= 1.0:
            report.pass_(
                f"textblob_polarity range: {tb['vmin']:.4f} to {tb['vmax']:.4f}"
            )
        else:
            report.fail(
                f"textblob_polarity out of range: {tb['vmin']:.4f} to {tb['vmax']:.4f}"
            )

    if table_exists(conn, "team_sentiment_daily") and column_exists(
        conn, "team_sentiment_daily", "hype_index"
    ):
        hype = conn.execute(
            """
            SELECT COUNT(*) AS n,
                   MIN(hype_index) AS vmin,
                   MAX(hype_index) AS vmax
            FROM team_sentiment_daily
            WHERE hype_index IS NOT NULL
            """
        ).fetchone()
        if hype["n"] == 0:
            report.fail("team_sentiment_daily hype_index is null for all rows")
        else:
            report.pass_(f"hype_index range: {hype['vmin']:.4f} to {hype['vmax']:.4f}")
            if hype["vmin"] < -0.01 or hype["vmax"] > 1.5:
                report.warn("hype_index values look unusual (expected 0.0–1.0)")


def check_reddit_timestamps(conn: sqlite3.Connection, report: Report) -> None:
    if not table_exists(conn, "raw_reddit_posts"):
        return
    if not column_exists(conn, "raw_reddit_posts", "created_utc"):
        report.fail("raw_reddit_posts missing column: created_utc")
        return

    stats = conn.execute(
        """
        SELECT COUNT(*) AS total,
               COUNT(DISTINCT date(datetime(created_utc,'unixepoch'))) AS distinct_days,
               MIN(date(datetime(created_utc,'unixepoch')))  AS min_date,
               MAX(date(datetime(created_utc,'unixepoch')))  AS max_date
        FROM raw_reddit_posts
        WHERE created_utc IS NOT NULL
        """
    ).fetchone()

    total = conn.execute("SELECT COUNT(*) FROM raw_reddit_posts").fetchone()[0]
    non_null = stats["total"] or 0

    if total == 0:
        return
    if non_null == 0:
        report.fail("raw_reddit_posts.created_utc is null for all rows")
        return

    report.pass_(f"raw_reddit_posts with created_utc: {non_null}/{total}")

    distinct_days = stats["distinct_days"] or 0
    min_date = stats["min_date"] or "unknown"
    max_date = stats["max_date"] or "unknown"

    if total > LOW_VOLUME_THRESHOLD and distinct_days < 2:
        report.fail(
            f"Only {distinct_days} distinct publish date(s) for {total} posts "
            f"({min_date} → {max_date})"
        )
    else:
        report.pass_(
            f"Reddit publish dates: {distinct_days} distinct days, "
            f"{min_date} → {max_date}"
        )

    if distinct_days == 1 and total > 1:
        report.warn(
            "All posts share one publish date — check RSS timestamp parsing"
        )


def check_predictions_2026(conn: sqlite3.Connection, report: Report) -> None:
    if not table_exists(conn, "match_predictions"):
        return

    if column_exists(conn, "match_predictions", "tournament_year"):
        count = conn.execute(
            "SELECT COUNT(*) FROM match_predictions WHERE tournament_year = 2026"
        ).fetchone()[0]
    else:
        count = conn.execute("SELECT COUNT(*) FROM match_predictions").fetchone()[0]
        if count > 0:
            report.warn("match_predictions has no tournament_year column; counting all rows")

    if count == 0:
        report.fail("No 2026 rows in match_predictions")
        return

    if count == EXPECTED_PREDICTIONS_2026:
        report.pass_(f"match_predictions 2026 rows: {count}")
    elif count < EXPECTED_PREDICTIONS_2026:
        report.warn(
            f"match_predictions 2026 rows: {count} "
            f"(expected {EXPECTED_PREDICTIONS_2026})"
        )
    else:
        report.pass_(f"match_predictions 2026 rows: {count}")
        report.warn(f"More than {EXPECTED_PREDICTIONS_2026} prediction rows ({count})")


def check_model_artifacts(model_dir: Path | None, report: Report) -> None:
    if model_dir is None:
        return

    model_pkl    = model_dir / "model.pkl"
    metrics_json = model_dir / "metrics.json"

    if model_pkl.is_file():
        report.pass_(f"model.pkl found: {model_pkl}")
    else:
        report.fail(f"model.pkl not found in {model_dir}")

    if not metrics_json.is_file():
        report.fail(f"metrics.json not found in {model_dir}")
        return

    try:
        with open(metrics_json, encoding="utf-8") as fh:
            metrics = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        report.fail(f"metrics.json unreadable: {exc}")
        return

    accuracy = metrics.get("accuracy")
    f1_macro = metrics.get("f1_macro")

    if accuracy is None or f1_macro is None:
        report.fail("metrics.json missing required keys: accuracy and/or f1_macro")
        return

    report.pass_(f"metrics.json found: accuracy={accuracy}, f1_macro={f1_macro}")

    try:
        if float(accuracy) < METRIC_WARN_THRESHOLD:
            report.warn(f"accuracy below {METRIC_WARN_THRESHOLD:.0%}: {float(accuracy):.4f}")
        if float(f1_macro) < METRIC_WARN_THRESHOLD:
            report.warn(f"f1_macro below {METRIC_WARN_THRESHOLD:.0%}: {float(f1_macro):.4f}")
    except (TypeError, ValueError):
        report.fail("metrics.json accuracy/f1_macro are not numeric")


def check_telegram(conn: sqlite3.Connection, report: Report) -> None:
    """
    Validate Telegram collection health.

    - raw_telegram_posts / raw_telegram_comments checks are warning-level unless
      TELEGRAM_API_ID + TELEGRAM_CHANNELS are explicitly configured.
    - Comment team-inheritance quality is checked separately.
    """
    api_id       = os.environ.get("TELEGRAM_API_ID", "").strip()
    channels_raw = os.environ.get("TELEGRAM_CHANNELS", "").strip()
    tg_configured = bool(api_id and channels_raw)
    comments_enabled = (
        os.environ.get("TELEGRAM_COLLECT_COMMENTS", "false").lower() == "true"
    )

    # --- raw_telegram_posts ---
    if table_exists(conn, "raw_telegram_posts"):
        count = conn.execute("SELECT COUNT(*) FROM raw_telegram_posts").fetchone()[0]
        if count == 0:
            if tg_configured:
                report.warn(
                    "raw_telegram_posts table exists but has 0 rows. "
                    "Telegram is configured — check collector logs."
                )
            else:
                report.info("raw_telegram_posts: empty (Telegram not configured — OK)")
        else:
            report.pass_(f"raw_telegram_posts rows: {count:,}")
    else:
        if tg_configured:
            report.warn(
                "Telegram is configured but raw_telegram_posts table does not exist. "
                "Run the collector to create and populate it."
            )
        else:
            report.info("raw_telegram_posts: table not present (Telegram not configured — OK)")

    # --- raw_telegram_comments ---
    if table_exists(conn, "raw_telegram_comments"):
        c_count = conn.execute("SELECT COUNT(*) FROM raw_telegram_comments").fetchone()[0]
        if c_count == 0:
            if comments_enabled:
                report.warn(
                    "raw_telegram_comments table exists but has 0 rows. "
                    "TELEGRAM_COLLECT_COMMENTS=true is set — check collector logs."
                )
            else:
                report.info(
                    "raw_telegram_comments: empty (TELEGRAM_COLLECT_COMMENTS not set — OK)"
                )
        else:
            report.pass_(f"raw_telegram_comments rows: {c_count:,}")
            _check_comment_team_inheritance(conn, report, c_count)
    elif comments_enabled:
        report.warn(
            "TELEGRAM_COLLECT_COMMENTS=true but raw_telegram_comments table not found. "
            "Run the collector to create and populate it."
        )


def _check_comment_team_inheritance(
    conn: sqlite3.Connection, report: Report, total_comments: int
) -> None:
    """
    Verify that Telegram comments have detected_team populated in processed_posts.

    PASS  if >= 70 % of telegram_comment rows in processed_posts have a detected_team.
    WARN  if < 70 % but > 0 %.
    FAIL  if 0 % (none have been team-tagged, inheritance is broken).
    """
    if not table_exists(conn, "processed_posts"):
        return
    if not column_exists(conn, "processed_posts", "detected_team"):
        report.warn(
            "processed_posts is missing detected_team column — "
            "team inheritance cannot be verified."
        )
        return
    if not column_exists(conn, "processed_posts", "source"):
        return

    row = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN detected_team IS NOT NULL THEN 1 ELSE 0 END) AS with_team
        FROM processed_posts
        WHERE source = 'telegram_comment'
        """
    ).fetchone()

    tg_proc_total = row["total"] or 0
    tg_proc_team  = row["with_team"] or 0

    if tg_proc_total == 0:
        report.info(
            f"raw_telegram_comments has {total_comments} row(s) but none have been "
            "processed yet — run the preprocessor."
        )
        return

    pct = tg_proc_team / tg_proc_total
    summary = (
        f"telegram_comment team inheritance: "
        f"{tg_proc_team}/{tg_proc_total} rows have detected_team "
        f"({pct:.0%})"
    )

    if tg_proc_team == 0:
        report.fail(
            summary + " — team inheritance is broken; check preprocessor logic."
        )
    elif pct < TG_COMMENT_TEAM_PASS_THRESHOLD:
        report.warn(
            summary
            + f" — below {TG_COMMENT_TEAM_PASS_THRESHOLD:.0%} threshold; "
            "many comments lack team context."
        )
    else:
        report.pass_(summary)


def check_dashboard_source(report: Report) -> None:
    for candidate in DASHBOARD_CANDIDATES:
        if candidate.is_file():
            report.pass_(f"Dashboard source found: {candidate}")
            return
    report.warn(
        "Dashboard source not found at services/dashboard/app.py "
        "(checked repo and /app paths)"
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_checks() -> int:
    report = Report()

    print("INTEGRATION CHECK REPORT")
    print("========================")
    print()

    db_path   = resolve_db_path(report)
    model_dir = resolve_model_dir(report)

    conn: sqlite3.Connection | None = None
    if db_path is not None:
        conn = open_db(db_path)
        if conn is None:
            report.fail(f"Could not open database: {db_path}")
        else:
            report.pass_(f"Database opened successfully: {db_path}")

    if conn is not None:
        check_tables(conn, report)
        check_row_counts(conn, report)
        check_processed_posts_schema(conn, report)
        check_sentiment(conn, report)
        check_reddit_timestamps(conn, report)
        check_predictions_2026(conn, report)
        check_telegram(conn, report)
        check_source_distribution(conn, report)
        check_team_distribution(conn, report)
        conn.close()

    check_model_artifacts(model_dir, report)
    check_dashboard_source(report)

    print()
    for line in report.lines:
        print(line)

    print()
    print("Summary:")
    print(f"  Critical failures : {report.critical_failures}")
    print(f"  Warnings          : {report.warnings}")
    print()

    if report.critical_failures == 0:
        print("Result: PASS")
        return 0

    print("Result: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(run_checks())
