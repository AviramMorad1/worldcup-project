import logging
import time

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

RETRAIN_INTERVAL_DAYS = 7


def run_training_cycle() -> None:
    logger.info("Starting collection run")

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
    X, y, years = build_feature_matrix(matches_df, rankings_df)

    metrics = train_models(X, y, years)
    save_metrics(metrics)

    run_2026_predictions()

    logger.info("Trainer cycle completed")


def main() -> None:
    logger.info("Trainer service started")

    # Run immediately on startup
    run_training_cycle()

    # Schedule weekly retraining
    schedule.every(RETRAIN_INTERVAL_DAYS).days.do(run_training_cycle)
    logger.info(
        "Trainer scheduled to rerun every %d day(s). Entering scheduler loop.",
        RETRAIN_INTERVAL_DAYS,
    )

    while True:
        schedule.run_pending()
        time.sleep(3600)  # wake up every hour to check the schedule


if __name__ == "__main__":
    main()
