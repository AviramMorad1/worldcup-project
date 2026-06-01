"""
main.py
-------
Collector service entry point for the World Cup sentiment pipeline.

Each collection cycle:
  1. load_football_data()   — CSV → raw_matches / raw_rankings (skips if already loaded)
  2. collect_reddit_data()  — scrape Reddit → raw_reddit_posts
"""

import logging
import time

from football_loader import load_football_data
from reddit_collector import collect_reddit_data  # existing module — not modified

logging.basicConfig(
    format="[COLLECTOR][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CYCLE_INTERVAL_SECONDS = 3600  # 1 hour


def collection_run() -> None:
    logger.info("Collection cycle starting.")

    # --- Football CSV data (idempotent — skips tables that already have rows) ---
    try:
        load_football_data()
    except Exception as exc:
        logger.error("load_football_data raised an unexpected error: %s", exc)

    # --- Reddit posts ---
    try:
        collect_reddit_data()
    except Exception as exc:
        logger.error("collect_reddit_data raised an unexpected error: %s", exc)

    logger.info("Collection cycle complete.")


def main() -> None:
    logger.info("Collector service started.")
    collection_run()

    while True:
        time.sleep(CYCLE_INTERVAL_SECONDS)
        collection_run()


if __name__ == "__main__":
    main()