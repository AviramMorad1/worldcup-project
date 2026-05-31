import logging
import time

from features import (
    build_feature_matrix,
    database_exists,
    load_raw_matches,
    validate_training_data,
)
from model import (
    ensure_model_dir,
    save_metrics,
    save_placeholder_metrics,
    train_models,
)

logging.basicConfig(
    format="[TRAINER][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_training_cycle() -> None:
    logger.info("Starting trainer cycle")

    ensure_model_dir()

    if not database_exists():
        logger.warning("Database not found — skipping training.")
        save_placeholder_metrics("Database not found")
        return

    matches_df = load_raw_matches()

    if not validate_training_data(matches_df):
        save_placeholder_metrics("raw_matches table missing, empty, or invalid")
        return

    X, y = build_feature_matrix(matches_df)

    metrics = train_models(X, y)
    save_metrics(metrics)

    logger.info("Trainer cycle completed")


def main() -> None:
    logger.info("Trainer service started")
    run_training_cycle()

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
