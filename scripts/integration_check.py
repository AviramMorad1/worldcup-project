#!/usr/bin/env python3
"""
Lightweight integration validation for the World Cup analytics pipeline.

Checks SQLite tables, sentiment ranges, model artifacts, and 2026 predictions
after running `docker compose up`.

Usage:
  python scripts/integration_check.py
  DATABASE_PATH=/app/data/worldcup.db MODEL_DIR=/app/data/models python scripts/integration_check.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution (local + Docker)
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

ROW_COUNT_TABLES = CRITICAL_TABLES

LOW_VOLUME_THRESHOLD = 50
EXPECTED_PREDICTIONS_2026 = 72
METRIC_WARN_THRESHOLD = 0.50


class Report:
    """Collect PASS / WARN / FAIL lines for the final summary."""

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


def resolve_db_path(report: Report) -> Path | None:
    """Return the first existing database path from env or common locations."""
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

    tried = ", ".join(str(p) for p in DB_CANDIDATES)
    report.fail(f"Database not found. Tried: {tried}")
    return None


def resolve_model_dir(report: Report) -> Path | None:
    """Return the first existing model directory from env or common locations."""
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

    tried = ", ".join(str(p) for p in MODEL_DIR_CANDIDATES)
    report.fail(f"Model directory not found. Tried: {tried}")
    return None


def open_db(db_path: Path) -> sqlite3.Connection | None:
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        return None


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    if not table_exists(conn, table_name):
        return False
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def check_tables(conn: sqlite3.Connection, report: Report) -> None:
    for table in CRITICAL_TABLES:
        if table_exists(conn, table):
            report.pass_(f"Table exists: {table}")
        else:
            report.fail(f"Missing critical table: {table}")

    for table in OPTIONAL_TABLES:
        if table_exists(conn, table):
            report.pass_(f"Optional table exists: {table}")
        else:
            report.warn(f"Optional table missing: {table}")


def check_row_counts(conn: sqlite3.Connection, report: Report) -> None:
    for table in ROW_COUNT_TABLES:
        if not table_exists(conn, table):
            continue
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count > 0:
            report.pass_(f"{table} rows: {count}")
        else:
            report.fail(f"{table} is empty")

        if table in {"raw_reddit_posts", "processed_posts"} and 0 < count < LOW_VOLUME_THRESHOLD:
            report.warn(f"{table} has only {count} rows (expected >= {LOW_VOLUME_THRESHOLD} for a mature run)")


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
                f"textblob_polarity out of range [-1, 1]: {tb['vmin']:.4f} to {tb['vmax']:.4f}"
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
            report.pass_(
                f"hype_index range: {hype['vmin']:.4f} to {hype['vmax']:.4f}"
            )
            if hype["vmin"] < -0.01 or hype["vmax"] > 1.5:
                report.warn(
                    "hype_index values look unusual; expected roughly 0.0 to 1.0"
                )


def check_reddit_timestamps(conn: sqlite3.Connection, report: Report) -> None:
    if not table_exists(conn, "raw_reddit_posts"):
        return
    if not column_exists(conn, "raw_reddit_posts", "created_utc"):
        report.fail("raw_reddit_posts missing column: created_utc")
        return

    stats = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN created_utc IS NULL THEN 1 ELSE 0 END) AS null_count,
               COUNT(DISTINCT date(datetime(created_utc, 'unixepoch'))) AS distinct_days,
               MIN(date(datetime(created_utc, 'unixepoch'))) AS min_date,
               MAX(date(datetime(created_utc, 'unixepoch'))) AS max_date
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
            f"({min_date} to {max_date})"
        )
    else:
        report.pass_(
            f"Reddit publish dates: {distinct_days} distinct days, "
            f"{min_date} to {max_date}"
        )

    if distinct_days == 1 and total > 1:
        report.warn(
            "All posts share one publish date — check RSS timestamp parsing "
            "or collection depth"
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
            report.warn(
                "match_predictions has no tournament_year column; counting all rows"
            )

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
        report.warn(
            f"More than {EXPECTED_PREDICTIONS_2026} prediction rows found "
            f"({count})"
        )


def check_model_artifacts(model_dir: Path | None, report: Report) -> None:
    if model_dir is None:
        return

    model_pkl = model_dir / "model.pkl"
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
        report.fail(
            "metrics.json missing required keys: accuracy and/or f1_macro"
        )
        return

    report.pass_(
        f"metrics.json found: accuracy={accuracy}, f1_macro={f1_macro}"
    )

    try:
        acc_val = float(accuracy)
        f1_val = float(f1_macro)
    except (TypeError, ValueError):
        report.fail("metrics.json accuracy/f1_macro are not numeric")
        return

    if acc_val < METRIC_WARN_THRESHOLD:
        report.warn(f"accuracy below {METRIC_WARN_THRESHOLD:.0%}: {acc_val:.4f}")
    if f1_val < METRIC_WARN_THRESHOLD:
        report.warn(f"f1_macro below {METRIC_WARN_THRESHOLD:.0%}: {f1_val:.4f}")


def check_dashboard_source(report: Report) -> None:
    for candidate in DASHBOARD_CANDIDATES:
        if candidate.is_file():
            report.pass_(f"Dashboard source found: {candidate}")
            return
    report.warn(
        "Dashboard source not found at services/dashboard/app.py "
        "(checked repo and /app paths)"
    )


def run_checks() -> int:
    report = Report()

    print("INTEGRATION CHECK REPORT")
    print("========================")
    print()

    db_path = resolve_db_path(report)
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
        check_sentiment(conn, report)
        check_reddit_timestamps(conn, report)
        check_predictions_2026(conn, report)
        conn.close()

    check_model_artifacts(model_dir, report)
    check_dashboard_source(report)

    print()
    for line in report.lines:
        print(line)

    print()
    print("Summary:")
    print(f"  Critical failures: {report.critical_failures}")
    print(f"  Warnings: {report.warnings}")
    print()

    if report.critical_failures == 0:
        print("Result: PASS")
        return 0

    print("Result: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(run_checks())
