"""
main.py
-------
Collector service entry point for the World Cup sentiment pipeline.

Each collection cycle:
  1. load_football_data()   — CSV → raw_matches / raw_rankings (skips if already loaded)
  2. collect_reddit_data()  — RSS → raw_reddit_posts
  3. write ready flag       — signals preprocessor that first cycle finished
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

from football_loader import load_football_data
from reddit_collector import collect_reddit_data

logging.basicConfig(
    format="[COLLECTOR][%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

COLLECTOR_READY_FLAG = "/app/data/collector_ready.flag"
DEFAULT_INTERVAL_HOURS = 168


def _collection_interval_seconds() -> int:
    raw = os.environ.get("COLLECTION_INTERVAL_HOURS", str(DEFAULT_INTERVAL_HOURS))
    try:
        hours = max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid COLLECTION_INTERVAL_HOURS=%r — using default %d.",
            raw,
            DEFAULT_INTERVAL_HOURS,
        )
        hours = DEFAULT_INTERVAL_HOURS
    return hours * 3600


def write_collector_ready_flag(posts_collected: int) -> None:
    """Write a marker file so downstream services know collection finished."""
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "posts_collected_this_cycle": posts_collected,
    }
    with open(COLLECTOR_READY_FLAG, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    logger.info("Collector ready flag written to %s", COLLECTOR_READY_FLAG)


def collection_run() -> None:
    logger.info("Collection cycle starting.")

    posts_collected = 0

    try:
        load_football_data()
    except Exception as exc:
        logger.error("load_football_data raised an unexpected error: %s", exc)

    try:
        posts_collected = collect_reddit_data()
    except Exception as exc:
        logger.error("collect_reddit_data raised an unexpected error: %s", exc)

    try:
        write_collector_ready_flag(posts_collected)
    except Exception as exc:
        logger.error("Failed writing collector ready flag: %s", exc)

    logger.info("Collection cycle complete.")


def main() -> None:
    interval_seconds = _collection_interval_seconds()
    logger.info(
        "Collector service started. Collection interval: %d hour(s).",
        interval_seconds // 3600,
    )
    collection_run()

    while True:
        time.sleep(interval_seconds)
        collection_run()


if __name__ == "__main__":
    main()
