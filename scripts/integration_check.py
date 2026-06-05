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

    if metrics.get("status") != "trained":
        return

    baseline = metrics.get("baseline") or {}
    base_acc = baseline.get("accuracy")
    base_f1 = baseline.get("f1_macro")
    if base_acc is not None and base_f1 is not None:
        report.info(
            f"Ranking baseline: accuracy={float(base_acc):.4f}, "
            f"f1_macro={float(base_f1):.4f}"
        )
        if metrics.get("ml_beats_baseline_f1"):
            report.pass_("ML f1_macro beats ranking baseline")
        else:
            report.warn(
                f"ML f1_macro ({f1_val:.4f}) does not beat ranking baseline "
                f"({float(base_f1):.4f})"
            )

    pred_dist = metrics.get("predicted_class_distribution") or {}
    pred_count = metrics.get("predicted_class_count")
    if pred_count is None and pred_dist:
        pred_count = len(pred_dist)

    if metrics.get("single_class_collapse") or pred_count == 1:
        report.warn(
            "ML model predicts only one class on test set — "
            "confusion matrix is single-column"
        )
    elif pred_count is not None and pred_count >= 2:
        report.pass_(
            f"Test predicted classes: {pred_count} distinct — {pred_dist}"
        )
    else:
        report.warn("Could not verify predicted class diversity from metrics.json")

    bm = metrics.get("best_model")
    models = metrics.get("models") or {}
    cm = models.get(bm, {}).get("confusion_matrix") if bm else None
    if cm:
        col_sums = [sum(row[i] for row in cm) for i in range(len(cm[0]))]
        nonzero_cols = sum(1 for s in col_sums if s > 0)
        if nonzero_cols <= 1:
            report.warn(
                f"Confusion matrix for {bm} has predictions in only one column"
            )
        else:
            report.pass_(
                f"Confusion matrix uses {nonzero_cols} predicted-class columns"
            )
        draw_preds = col_sums[1] if len(col_sums) > 1 else 0
        if draw_preds == 0:
            report.warn(
                f"ML component predicts zero draws on test set (draw column sum=0)"
            )
        else:
            report.pass_(f"ML draw predictions on test set: {draw_preds}")

    draw_recall = metrics.get("draw_recall")
    if draw_recall is not None:
        report.info(f"ML draw recall (2022 test): {float(draw_recall):.4f}")
        if float(draw_recall) == 0:
            report.warn("ML draw recall is zero — draw handling still weak")

    bal_acc = metrics.get("balanced_accuracy")
    if bal_acc is not None:
        report.info(f"ML balanced accuracy: {float(bal_acc):.4f}")

    draw_rule = metrics.get("draw_decision_rule") or {}
    if draw_rule.get("enabled"):
        report.info(
            f"Draw decision rule: threshold={draw_rule.get('draw_threshold')}, "
            f"closeness={draw_rule.get('closeness_threshold')}"
        )

    argmax_cmp = metrics.get("argmax_comparison") or {}
    if argmax_cmp:
        report.info(
            f"Argmax-only comparison: accuracy={float(argmax_cmp.get('accuracy', 0)):.4f}, "
            f"f1_macro={float(argmax_cmp.get('f1_macro', 0)):.4f}, "
            f"draw_recall={float(argmax_cmp.get('draw_recall', 0)):.4f}"
        )


def check_squad_and_league(conn: sqlite3.Connection, report: Report) -> None:
    """Player stats, league weights, squad strength, and prediction adjustments."""
    global_csv = REPO_ROOT / "datasets" / "global_national_league_rankings.csv"
    league_csv = REPO_ROOT / "datasets" / "league_strengths.csv"
    if global_csv.is_file():
        report.pass_(f"global_national_league_rankings.csv exists: {global_csv}")
    else:
        report.warn(
            "global_national_league_rankings.csv missing — run scripts/build_league_strengths.py"
        )
    if league_csv.is_file():
        report.pass_(f"league_strengths.csv exists: {league_csv}")
    else:
        report.warn(
            "league_strengths.csv missing — trainer uses inline fallback weights"
        )

    if table_exists(conn, "raw_player_stats"):
        n = conn.execute("SELECT COUNT(*) FROM raw_player_stats").fetchone()[0]
        if n >= 800:
            report.pass_(f"raw_player_stats rows: {n}")
        elif n > 0:
            report.warn(f"raw_player_stats rows: {n} (expected ~900)")
        else:
            report.fail("raw_player_stats is empty")

        teams = conn.execute(
            "SELECT COUNT(DISTINCT national_team) FROM raw_player_stats "
            "WHERE national_team IS NOT NULL AND national_team != ''"
        ).fetchone()[0]
        leagues = conn.execute(
            "SELECT COUNT(DISTINCT league) FROM raw_player_stats "
            "WHERE league IS NOT NULL AND league != ''"
        ).fetchone()[0]
        report.info(f"raw_player_stats: {teams} national teams, {leagues} leagues")

        if column_exists(conn, "raw_player_stats", "league_weight"):
            missing_lw = conn.execute(
                "SELECT COUNT(*) FROM raw_player_stats "
                "WHERE league_weight IS NULL OR league_weight = 0"
            ).fetchone()[0]
            if missing_lw == n and n > 0:
                report.warn("No league_weight values applied to player rows")
            elif missing_lw > n * 0.5:
                report.warn(
                    f"{missing_lw}/{n} players missing league_weight"
                )
            else:
                report.pass_(
                    f"league_weight applied ({n - missing_lw}/{n} players)"
                )

        if column_exists(conn, "raw_player_stats", "minutes"):
            zero_min = conn.execute(
                "SELECT COUNT(*) FROM raw_player_stats WHERE minutes = 0"
            ).fetchone()[0]
            if zero_min == n and n > 50:
                report.warn(
                    "All player minutes are zero — missing stats may be coded as weak"
                )

    if table_exists(conn, "team_squad_strength"):
        sq_n = conn.execute("SELECT COUNT(*) FROM team_squad_strength").fetchone()[0]
        if sq_n >= 40:
            report.pass_(f"team_squad_strength rows: {sq_n}")
        elif sq_n > 0:
            report.warn(f"team_squad_strength rows: {sq_n} (expected ~48)")
        else:
            report.warn("team_squad_strength empty — run trainer after collector")

        if column_exists(conn, "team_squad_strength", "coverage_tier"):
            tiers = conn.execute(
                "SELECT coverage_tier, COUNT(*) FROM team_squad_strength "
                "GROUP BY coverage_tier"
            ).fetchall()
            report.info("Coverage tiers: " + ", ".join(f"{t[0]}={t[1]}" for t in tiers))
            very_low = conn.execute(
                "SELECT team FROM team_squad_strength WHERE coverage_tier = 'Very Low'"
            ).fetchall()
            if very_low:
                report.info(
                    "Very Low coverage: " + ", ".join(r[0] for r in very_low[:12])
                    + ("..." if len(very_low) > 12 else "")
                )

        if column_exists(conn, "team_squad_strength", "final_squad_strength"):
            top = conn.execute(
                "SELECT team, final_squad_strength FROM team_squad_strength "
                "ORDER BY final_squad_strength DESC LIMIT 10"
            ).fetchall()
            report.info(
                "Top squad strength: "
                + ", ".join(f"{r[0]}={r[1]:.1f}" for r in top)
            )
            if column_exists(conn, "team_squad_strength", "data_quality_score"):
                bottom = conn.execute(
                    "SELECT team, data_quality_score FROM team_squad_strength "
                    "ORDER BY data_quality_score ASC LIMIT 10"
                ).fetchall()
                report.info(
                    "Lowest data quality: "
                    + ", ".join(f"{r[0]}={r[1]:.2f}" for r in bottom)
                )

    if table_exists(conn, "match_predictions"):
        adj_cols = [
            "squad_adjustment_applied",
            "squad_adjustment_amount",
            "adjusted_confidence",
            "base_team_a_proba",
        ]
        for col in adj_cols:
            if not column_exists(conn, "match_predictions", col):
                report.warn(f"match_predictions missing column: {col}")

        ps_n = (
            conn.execute("SELECT COUNT(*) FROM raw_player_stats").fetchone()[0]
            if table_exists(conn, "raw_player_stats")
            else 0
        )
        sqs_n = (
            conn.execute("SELECT COUNT(*) FROM team_squad_strength").fetchone()[0]
            if table_exists(conn, "team_squad_strength")
            else 0
        )
        report.info(f"raw_player_stats row count: {ps_n}")
        report.info(f"team_squad_strength row count: {sqs_n}")

        squad_pop = 0
        if column_exists(conn, "match_predictions", "squad_strength_a"):
            squad_pop = conn.execute(
                """
                SELECT COUNT(*) FROM match_predictions
                WHERE tournament_year = 2026
                  AND squad_strength_a IS NOT NULL
                  AND squad_strength_b IS NOT NULL
                """
            ).fetchone()[0]
            report.info(
                f"2026 predictions with squad_strength_a/b: {squad_pop}/"
                f"{EXPECTED_PREDICTIONS_2026}"
            )
            if squad_pop >= 50:
                report.pass_(
                    f"Squad columns populated on {squad_pop}/"
                    f"{EXPECTED_PREDICTIONS_2026} predictions"
                )
            elif squad_pop > 0:
                report.warn(
                    f"Only {squad_pop}/{EXPECTED_PREDICTIONS_2026} predictions "
                    "have squad strength values"
                )
            elif ps_n > 0 or sqs_n > 0:
                report.fail(
                    "Squad source tables have data but no 2026 predictions "
                    "have squad_strength_a/b — trainer INSERT may be broken"
                )
            else:
                report.warn(
                    "No squad values on 2026 predictions (player stats not loaded yet)"
                )

        if column_exists(conn, "match_predictions", "squad_adjustment_applied"):
            row = conn.execute(
                """
                SELECT COUNT(*) AS n,
                       SUM(squad_adjustment_applied) AS adj,
                       AVG(ABS(squad_adjustment_amount)) AS avg_adj
                FROM match_predictions
                WHERE tournament_year = 2026
                """
            ).fetchone()
            if row and row["n"]:
                adj_n = row["adj"] or 0
                report.info(
                    f"2026 predictions with squad_adjustment_applied=1: {adj_n}/{row['n']}"
                )
                report.info(
                    f"Average absolute squad adjustment: {(row['avg_adj'] or 0):.4f}"
                )
                if adj_n > 0:
                    report.pass_(
                        f"2026 predictions: {adj_n}/{row['n']} squad-adjusted, "
                        f"avg |adjustment|={(row['avg_adj'] or 0):.4f}"
                    )
                elif ps_n > 0 and sqs_n > 0:
                    report.fail(
                        "Squad tables populated but no predictions have "
                        "squad_adjustment_applied=1"
                    )
                else:
                    report.warn(
                        f"2026 predictions: 0/{row['n']} squad-adjusted "
                        "(expected after player stats load)"
                    )
                if row["adj"] and row["n"]:
                    big_adj = conn.execute(
                        """
                        SELECT COUNT(*) FROM match_predictions
                        WHERE tournament_year = 2026
                          AND squad_adjustment_applied = 1
                          AND ABS(squad_adjustment_amount) > 0.06
                          AND (
                            squad_coverage_tier_a = 'Very Low'
                            OR squad_coverage_tier_b = 'Very Low'
                          )
                        """
                    ).fetchone()[0]
                    if big_adj:
                        report.warn(
                            f"{big_adj} Very Low coverage match(es) with large adjustment"
                        )

        _check_2026_confidence_calibration(conn, report)


def _check_2026_confidence_calibration(conn: sqlite3.Connection, report: Report) -> None:
    if not table_exists(conn, "match_predictions"):
        return
    if not column_exists(conn, "match_predictions", "adjusted_confidence"):
        return

    over_cap = conn.execute(
        """
        SELECT COUNT(*) FROM match_predictions
        WHERE tournament_year = 2026 AND adjusted_confidence > 0.88
        """
    ).fetchone()[0]
    if over_cap:
        report.fail(
            f"{over_cap} group-stage prediction(s) exceed 88% final confidence"
        )
    else:
        report.pass_("No 2026 final confidence above 88% cap")

    hist_col = (
        "historical_ml_confidence"
        if column_exists(conn, "match_predictions", "historical_ml_confidence")
        else "raw_model_confidence"
    )
    if column_exists(conn, "match_predictions", hist_col):
        avg_row = conn.execute(
            f"""
            SELECT AVG({hist_col}) AS avg_ml,
                   AVG(adjusted_confidence) AS avg_final
            FROM match_predictions
            WHERE tournament_year = 2026
            """
        ).fetchone()
        if avg_row and avg_row["avg_ml"] is not None:
            report.info(
                f"Avg historical ML signal: {avg_row['avg_ml']:.1%} vs "
                f"avg final confidence: {avg_row['avg_final']:.1%}"
            )

    non_null_final = conn.execute(
        """
        SELECT COUNT(*) FROM match_predictions
        WHERE tournament_year = 2026 AND adjusted_confidence IS NOT NULL
        """
    ).fetchone()[0]
    if non_null_final >= EXPECTED_PREDICTIONS_2026:
        report.pass_(f"Final confidence populated on {non_null_final}/72 predictions")
    elif non_null_final > 0:
        report.warn(f"Only {non_null_final}/72 predictions have final confidence")
    else:
        report.fail("No final adjusted_confidence values on 2026 predictions")

    large_gap_flat = conn.execute(
        """
        SELECT COUNT(*) FROM match_predictions
        WHERE tournament_year = 2026
          AND squad_strength_diff IS NOT NULL
          AND ABS(squad_strength_diff) >= 25
          AND adjusted_confidence BETWEEN 0.55 AND 0.68
        """
    ).fetchone()[0]
    total_large = conn.execute(
        """
        SELECT COUNT(*) FROM match_predictions
        WHERE tournament_year = 2026
          AND squad_strength_diff IS NOT NULL
          AND ABS(squad_strength_diff) >= 25
        """
    ).fetchone()[0]
    if column_exists(conn, "match_predictions", "squad_strength_diff"):
        # rank gap proxy via squad + winner alignment — use large squad gap + low final
        strong_low = conn.execute(
            """
            SELECT COUNT(*) FROM match_predictions
            WHERE tournament_year = 2026
              AND ABS(squad_strength_diff) >= 25
              AND adjusted_confidence < 0.70
            """
        ).fetchone()[0]
        if strong_low >= 3:
            report.warn(
                f"{strong_low} large squad-gap matches have final confidence below 70%"
            )

    if total_large > 0 and large_gap_flat == total_large:
        report.warn(
            f"All {total_large} large squad-gap matches have final confidence "
            "between 55–68% — blend may be too conservative"
        )
    elif large_gap_flat > total_large // 2 and total_large >= 3:
        report.warn(
            f"{large_gap_flat}/{total_large} large squad-gap matches still have "
            "final confidence 55–68%"
        )

    top_gap = conn.execute(
        """
        SELECT team_a, team_b, squad_strength_diff, adjusted_confidence,
               predicted_winner
        FROM match_predictions
        WHERE tournament_year = 2026
          AND squad_strength_diff IS NOT NULL
        ORDER BY ABS(squad_strength_diff) DESC
        LIMIT 10
        """
    ).fetchall()
    if top_gap:
        report.info(
            "Top squad-gap matches: "
            + "; ".join(
                f"{r[0]} vs {r[1]} diff={r[2]:+.1f} adj={r[3]:.1%} ({r[4]})"
                for r in top_gap[:5]
            )
        )

    boost_sel = (
        "confidence_boost_amount"
        if column_exists(conn, "match_predictions", "confidence_boost_amount")
        else "squad_adjustment_amount"
    )
    hist_sel = (
        "historical_ml_confidence"
        if column_exists(conn, "match_predictions", "historical_ml_confidence")
        else "raw_model_confidence"
    )
    rank_sel = (
        "ranking_baseline_confidence"
        if column_exists(conn, "match_predictions", "ranking_baseline_confidence")
        else "NULL"
    )
    top_conf = conn.execute(
        f"""
        SELECT team_a, team_b, adjusted_confidence, {hist_sel},
               {rank_sel}
        FROM match_predictions
        WHERE tournament_year = 2026
        ORDER BY adjusted_confidence DESC
        LIMIT 10
        """
    ).fetchall()
    if top_conf:
        report.info(
            "Highest final confidence: "
            + "; ".join(
                f"{r[0]} vs {r[1]} final={r[2]:.1%} "
                f"(ML={r[3]:.1%}"
                + (f", rank={r[4]:.1%})" if r[4] is not None else ")")
                for r in top_conf[:5]
            )
        )

    gc_boost = (
        "confidence_boost_amount"
        if column_exists(conn, "match_predictions", "confidence_boost_amount")
        else "squad_adjustment_amount"
    )
    germany_curacao = conn.execute(
        f"""
        SELECT adjusted_confidence, raw_model_confidence, {gc_boost}
        FROM match_predictions
        WHERE tournament_year = 2026
          AND team_a = 'Germany' AND team_b = 'Curaçao'
        """
    ).fetchone()
    if germany_curacao:
        adj, raw, boost = germany_curacao
        report.info(
            f"Germany vs Curaçao: base={raw:.1%}, adjusted={adj:.1%}, "
            f"boost={boost or 0:.3f}"
        )
        if adj < 0.72:
            report.warn(
                f"Germany vs Curaçao final confidence {adj:.1%} below 72% "
                "(expected ~72–80% for strong blended signals)"
            )
        elif 0.72 <= adj <= 0.88:
            report.pass_(f"Germany vs Curaçao final confidence {adj:.1%} in target range")
        elif adj > 0.88:
            report.warn(
                f"Germany vs Curaçao adjusted confidence {adj:.1%} exceeds 88% cap"
            )


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
        check_squad_and_league(conn, report)
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
