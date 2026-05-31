import logging
import time
import schedule

logging.basicConfig(
    format="[PREPROCESSOR][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def preprocessing_run():
    logger.info("Starting preprocessing placeholder")


def main():
    logger.info("Preprocessor service started")
    preprocessing_run()

    schedule.every(7).days.do(preprocessing_run)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
