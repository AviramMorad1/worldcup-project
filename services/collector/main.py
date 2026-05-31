import logging
import time
import schedule

logging.basicConfig(
    format="[COLLECTOR][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def collection_run():
    logger.info("Starting collection run placeholder")


def main():
    logger.info("Collector service started")
    collection_run()

    schedule.every(7).days.do(collection_run)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
