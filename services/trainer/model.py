import json
import logging
from pathlib import Path

import pandas as pd

MODEL_DIR = Path("/app/data/models")
METRICS_PATH = MODEL_DIR / "metrics.json"

logger = logging.getLogger(__name__)


def ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Model directory ready: %s", MODEL_DIR)


def save_placeholder_metrics(reason: str) -> None:
    ensure_model_dir()
    metrics = {
        "status": "not_trained",
        "reason": reason,
        "accuracy": None,
        "f1_macro": None,
        "model_name": None,
    }
    save_metrics(metrics)


def train_models(X: pd.DataFrame, y: pd.Series) -> dict:
    if X.empty or y.empty:
        logger.warning("train_models called with empty data — skipping.")
        return {
            "status": "not_trained",
            "reason": "No valid training data available",
        }

    logger.info(
        "train_models called with %d rows and features %s. "
        "Real model training not implemented yet.",
        len(X),
        list(X.columns),
    )
    return {
        "status": "not_trained",
        "reason": "Trainer skeleton ready; real model training not implemented yet",
        "rows": len(X),
        "features": list(X.columns),
    }


def save_metrics(metrics: dict) -> None:
    ensure_model_dir()
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", METRICS_PATH)
