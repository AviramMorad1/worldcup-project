"""
main.py
-------
Collector service entry point for the World Cup sentiment pipeline.

Each collection cycle (order matters for trainer/dashboard):
  1. load_football_data()   — matches, rankings, player stats, squads (prediction inputs)
  2. write prediction-ready flag — trainer may run 2026 predictions after this step
  3. collect_reddit_data()  — RSS → raw_reddit_posts (sentiment; slower)
  4. write collector ready flag — full cycle done for preprocessor
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
PREDICTION_READY_FLAG = "/app/data/prediction_ready.flag"
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


def _write_flag(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    logger.info("Flag written to %s", path)


def write_prediction_ready_flag() -> None:
    """Signal that match/ranking/player data for predictions is loaded."""
    _write_flag(
        PREDICTION_READY_FLAG,
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "stage": "football_data_loaded",
        },
    )


def write_collector_ready_flag(posts_collected: int) -> None:
    """Signal full collector cycle finished (football + Reddit)."""
    _write_flag(
        COLLECTOR_READY_FLAG,
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "posts_collected_this_cycle": posts_collected,
            "stage": "full_cycle",
        },
    )


def collection_run() -> None:
    logger.info("Collection cycle starting.")

    posts_collected = 0

    # Phase 1 — prediction inputs first (fast CSV/DB loads)
    try:
        load_football_data()
        write_prediction_ready_flag()
        logger.info(
            "Football / player stats loaded — prediction_ready flag set "
            "(trainer can use squad data)."
        )
    except Exception as exc:
        logger.error("load_football_data raised an unexpected error: %s", exc)

    # Phase 2 — sentiment sources (RSS; can take minutes)
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
