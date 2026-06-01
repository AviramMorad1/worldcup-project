import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

DB_PATH = "/app/data/worldcup.db"
MODEL_DIR = Path("/app/data/models")
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory + table setup
# ---------------------------------------------------------------------------

def ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Model directory ready: %s", MODEL_DIR)


def _open_db() -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and a 60-second busy timeout.

    Mirrors the same helper in features.py — keeps the trainer robust when
    the collector container is concurrently writing to the shared database.
    """
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def create_model_metrics_table() -> None:
    conn = _open_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at     TEXT,
                model_name TEXT,
                accuracy   REAL,
                f1_macro   REAL,
                notes      TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()
    logger.info("model_metrics table ready.")


def _insert_model_metrics(
    model_name: str,
    accuracy: float,
    f1_macro: float,
    notes: str = "",
) -> None:
    run_at = datetime.now(timezone.utc).isoformat()
    conn = _open_db()
    try:
        conn.execute(
            "INSERT INTO model_metrics (run_at, model_name, accuracy, f1_macro, notes)"
            " VALUES (?, ?, ?, ?, ?)",
            (run_at, model_name, accuracy, f1_macro, notes),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info(
        "model_metrics: inserted row for '%s' (accuracy=%.4f, f1_macro=%.4f).",
        model_name, accuracy, f1_macro,
    )


# ---------------------------------------------------------------------------
# JSON serialisation helper (handles numpy scalar types)
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """Split into train / test sets.

    Priority order:
      1. Year-based: train = years < 2022, test = year == 2022 (preferred)
      2. Stratified random 80/20
      3. Regular random 80/20
    """
    if 2022 in years.values:
        test_mask = years == 2022
        train_mask = years < 2022
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        if len(X_train) > 0 and len(X_test) > 0:
            logger.info(
                "Split method: year-based — train: years<2022 (%d rows), test: year==2022 (%d rows).",
                len(X_train), len(X_test),
            )
            return X_train, X_test, y_train, y_test, "year-based (train<2022, test=2022)"

    # Stratified random split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
        logger.info(
            "Split method: stratified random — train: %d rows, test: %d rows.",
            len(X_train), len(X_test),
        )
        return X_train, X_test, y_train, y_test, "stratified random 80/20"
    except ValueError as exc:
        logger.warning("Stratified split failed (%s) — falling back to regular split.", exc)

    # Regular random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    logger.info(
        "Split method: regular random — train: %d rows, test: %d rows.",
        len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test, "random 80/20"


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
) -> dict:
    """Train RandomForest and XGBoost classifiers; save best model and metrics.

    Safety guarantees:
    - Returns a not_trained dict (never raises) for any failure condition.
    - XGBoost failure is logged and skipped; RandomForest result is kept.
    - Writes model.pkl and calls _insert_model_metrics only on success.
    """
    if X.empty or y.empty:
        logger.warning("train_models called with empty data — skipping.")
        return {"status": "not_trained", "reason": "No valid training data available"}

    if len(X) < 10:
        logger.warning("Too few rows (%d) — need at least 10 to train.", len(X))
        return {"status": "not_trained", "reason": "Not enough valid training data"}

    if y.nunique() < 2:
        logger.warning("Only %d class(es) in target — need at least 2.", y.nunique())
        return {"status": "not_trained", "reason": "Not enough valid training data"}

    X_train, X_test, y_train, y_test, split_method = _split_data(X, y, years)
    logger.info("Training rows: %d | Test rows: %d", len(X_train), len(X_test))
    logger.info("Features: %s", list(X.columns))

    model_results: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    # --- RandomForestClassifier ---
    logger.info("Training RandomForestClassifier ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_metrics = _evaluate(rf, X_test, y_test)
    model_results["RandomForestClassifier"] = rf_metrics
    trained_models["RandomForestClassifier"] = rf
    logger.info(
        "RandomForestClassifier — accuracy=%.4f, f1_macro=%.4f",
        rf_metrics["accuracy"], rf_metrics["f1_macro"],
    )

    # --- XGBClassifier (optional — graceful fallback on any error) ---
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415

        logger.info("Training XGBClassifier ...")
        xgb = XGBClassifier(
            n_estimators=200,
            random_state=42,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        xgb_metrics = _evaluate(xgb, X_test, y_test)
        model_results["XGBClassifier"] = xgb_metrics
        trained_models["XGBClassifier"] = xgb
        logger.info(
            "XGBClassifier — accuracy=%.4f, f1_macro=%.4f",
            xgb_metrics["accuracy"], xgb_metrics["f1_macro"],
        )
    except Exception as exc:
        logger.warning("XGBClassifier training failed — skipping. Reason: %s", exc)

    # --- Select best model (highest f1_macro, accuracy as tie-breaker) ---
    best_name = max(
        model_results,
        key=lambda name: (
            model_results[name]["f1_macro"],
            model_results[name]["accuracy"],
        ),
    )
    best_metrics = model_results[best_name]
    best_model = trained_models[best_name]

    logger.info(
        "Best model: %s — accuracy=%.4f, f1_macro=%.4f",
        best_name, best_metrics["accuracy"], best_metrics["f1_macro"],
    )
    logger.info(
        "Confusion matrix (%s):\n%s",
        best_name, best_metrics["confusion_matrix"],
    )

    # --- Persist model ---
    ensure_model_dir()
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Best model saved to %s", MODEL_PATH)

    # --- Persist metrics to SQLite ---
    try:
        create_model_metrics_table()
        _insert_model_metrics(
            model_name=best_name,
            accuracy=best_metrics["accuracy"],
            f1_macro=best_metrics["f1_macro"],
            notes=f"split={split_method}; train_rows={len(X_train)}; test_rows={len(X_test)}",
        )
    except Exception as exc:
        logger.warning("Could not write to model_metrics table: %s", exc)

    return {
        "status": "trained",
        "best_model": best_name,
        "accuracy": best_metrics["accuracy"],
        "f1_macro": best_metrics["f1_macro"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "split_method": split_method,
        "features": list(X.columns),
        "models": model_results,
    }


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------

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


def save_metrics(metrics: dict) -> None:
    ensure_model_dir()
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, cls=_NumpyEncoder)
    logger.info("Metrics saved to %s", METRICS_PATH)
