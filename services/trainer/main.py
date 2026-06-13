import logging
import sqlite3
import time
from pathlib import Path

import schedule

from features import (
    build_feature_matrix,
    database_exists,
    load_raw_matches,
    load_raw_rankings,
    validate_training_data,
)
from model import (
    ensure_model_dir,
    save_metrics,
    save_placeholder_metrics,
    train_models,
)
from predictions import run_2026_predictions

logging.basicConfig(
    format="[TRAINER][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DB_PATH = "/app/data/worldcup.db"
COLLECTOR_READY_FLAG = Path("/app/data/collector_ready.flag")
PREDICTION_READY_FLAG = Path("/app/data/prediction_ready.flag")
PLAYER_STATS_MIN_ROWS = 100
WAIT_POLL_SECONDS = 10
WAIT_MAX_SECONDS = 300
RETRAIN_INTERVAL_DAYS = 7


def _player_stats_count() -> int:
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM raw_player_stats"
            ).fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def wait_for_player_stats() -> None:
    """
    Wait for collector phase-1 (football + player stats) before 2026 predictions.
    """
    deadline = time.time() + WAIT_MAX_SECONDS
    while time.time() < deadline:
        count = _player_stats_count()
        if count >= PLAYER_STATS_MIN_ROWS:
            logger.info("Player stats available in DB: %d rows.", count)
            return
        if PREDICTION_READY_FLAG.exists():
            logger.info(
                "prediction_ready flag set; waiting for player stats (rows=%d).",
                count,
            )
        time.sleep(WAIT_POLL_SECONDS)

    count = _player_stats_count()
    logger.info(
        "Proceeding with predictions (player stats rows=%d). "
        "Trainer will load wc_players_with_stats.csv directly if needed.",
        count,
    )


def run_training_cycle() -> None:
    logger.info("Starting training cycle")

    ensure_model_dir()

    if not database_exists():
        logger.warning("Database not found — skipping training.")
        save_placeholder_metrics("Database not found")
        return

    matches_df = load_raw_matches()

    if not validate_training_data(matches_df):
        save_placeholder_metrics("raw_matches table missing, empty, or invalid")
        return

    rankings_df = load_raw_rankings()
    X, y, years, sample_weight = build_feature_matrix(matches_df, rankings_df)

    metrics = train_models(X, y, years, sample_weight)
    save_metrics(metrics)

    wait_for_player_stats()
    run_2026_predictions()

    logger.info("Trainer cycle completed")


def main() -> None:
    logger.info("Trainer service started")

    run_training_cycle()

    schedule.every(RETRAIN_INTERVAL_DAYS).days.do(run_training_cycle)
    logger.info(
        "Trainer scheduled to rerun every %d day(s). Entering scheduler loop.",
        RETRAIN_INTERVAL_DAYS,
    )

    while True:
        schedule.run_pending()
        time.sleep(3600)


if __name__ == "__main__":
    main()
