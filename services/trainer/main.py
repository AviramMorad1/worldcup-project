import logging
import time
import schedule

logging.basicConfig(
    format="[TRAINER][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def training_run():
    logger.info("Starting training placeholder")


def main():
    logger.info("Trainer service started")
    training_run()

    schedule.every(7).days.do(training_run)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
