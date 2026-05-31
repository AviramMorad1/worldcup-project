import logging
import time
import os
import schedule
from reddit_collector import collect_reddit_data

logging.basicConfig(
    format="[COLLECTOR][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def collection_run() -> None:
    """
    Run one collector cycle.
    Currently collects real Reddit data through RSS.
    Football CSV loading will be added later.
    """
    logger.info("Starting collection run")

    try:
        collect_reddit_data()
        logger.info("Collection run completed successfully")

    except Exception:
        logger.exception("Collection run failed")


def main():
    logger.info("Collector service started")
    collection_run()

    schedule.every(7).days.do(collection_run)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
