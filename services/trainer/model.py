"""
model.py
--------
Model training, calibration, baseline evaluation, and persistence.

Trained with recency-weighted samples.
Probability calibration via CalibratedClassifierCV reduces overconfident scores.
A simple ranking-baseline model is evaluated alongside the ML models.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

DB_PATH    = "/app/data/worldcup.db"
MODEL_DIR  = Path("/app/data/models")
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
# JSON serialisation helper
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
    sample_weight: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
           pd.Series | None, pd.Series | None, str, list[int]]:
    """
    Split into train / test sets.

    Priority order:
      1. Year-based: train = all years before max year, test = max year
      2. Stratified random 80/20 (fallback — only when single year in data)
    """
    unique_years = sorted(years.unique().tolist())

    if len(unique_years) >= 2:
        # Use the latest year as test set, all prior as training
        test_year  = unique_years[-1]
        train_mask = years < test_year
        test_mask  = years == test_year
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        sw_train = sample_weight[train_mask] if sample_weight is not None else None
        sw_test  = sample_weight[test_mask]  if sample_weight is not None else None
        train_years = sorted([int(yr) for yr in years[train_mask].unique()])
        if len(X_train) > 0 and len(X_test) > 0:
            logger.info(
                "Split method: year-based — train: years %s (%d rows), "
                "test: year=%d (%d rows).",
                train_years, len(X_train), test_year, len(X_test),
            )
            method = f"year-based (train={train_years[0]}–{train_years[-1]}, test={test_year})"
            return X_train, X_test, y_train, y_test, sw_train, sw_test, method, train_years

    # Fallback: stratified random split
    logger.warning(
        "Only %d tournament year(s) in data — falling back to stratified random split. "
        "Evaluation metrics will be optimistic. Add more historical data to enable "
        "chronological evaluation.",
        len(unique_years),
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )
    sw_train = sample_weight.loc[X_train.index] if sample_weight is not None else None
    sw_test  = sample_weight.loc[X_test.index]  if sample_weight is not None else None
    method = "stratified random 80/20 (FALLBACK — single tournament year)"
    logger.info(
        "Split method: random — train: %d rows, test: %d rows.", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test, sw_train, sw_test, method, unique_years


# ---------------------------------------------------------------------------
# Baseline model: predict winner based on rank_diff sign
# ---------------------------------------------------------------------------


def _evaluate_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Simple ranking-baseline: if rank_diff < 0 → team_a wins (class 2),
    if rank_diff > 0 → team_b wins (class 0), else draw (class 1).

    rank_diff = rank_a - rank_b.  Lower FIFA rank number = better team.
    So if rank_diff < 0, team_a has a lower (better) rank → predict team_a wins.
    """
    if "rank_diff" not in X_test.columns:
        logger.warning("rank_diff not available — skipping baseline evaluation.")
        return {"accuracy": None, "f1_macro": None}

    def _baseline_predict(rd: float) -> int:
        if rd < -5:
            return 2   # team_a clearly better-ranked
        if rd > 5:
            return 0   # team_b clearly better-ranked
        return 1       # too close — predict draw

    y_pred_base = X_test["rank_diff"].apply(_baseline_predict)
    acc  = float(accuracy_score(y_test, y_pred_base))
    f1   = float(f1_score(y_test, y_pred_base, average="macro", zero_division=0))
    cm   = confusion_matrix(y_test, y_pred_base).tolist()
    logger.info(
        "Ranking baseline — accuracy=%.4f, f1_macro=%.4f", acc, f1
    )
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    acc  = float(accuracy_score(y_test, y_pred))
    f1   = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------


def _try_calibrate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sw_train: pd.Series | None,
) -> tuple[object, bool]:
    """
    Wrap model in CalibratedClassifierCV(method='sigmoid').

    Returns (calibrated_model, calibrated:bool).
    Falls back to the original model if calibration fails.
    """
    n_train = len(X_train)
    n_cv = 3 if n_train >= 60 else 2

    if n_train < 30:
        logger.warning(
            "Skipping calibration: only %d training rows (need ≥30).", n_train
        )
        return model, False

    try:
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method="sigmoid",
            cv=n_cv,
        )
        fit_params = {}
        if sw_train is not None:
            fit_params["sample_weight"] = sw_train.values
        calibrated.fit(X_train, y_train, **fit_params)
        logger.info(
            "Probability calibration applied (method=sigmoid, cv=%d, n_train=%d).",
            n_cv, n_train,
        )
        return calibrated, True
    except Exception as exc:
        logger.warning(
            "Calibration failed (%s) — using uncalibrated model.", exc
        )
        return model, False


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    sample_weight: pd.Series | None = None,
) -> dict:
    """
    Train RandomForest and XGBoost classifiers; calibrate; save best model.

    Returns metrics dict or not_trained dict on any failure.
    """
    if X.empty or y.empty:
        logger.warning("train_models called with empty data — skipping.")
        return {"status": "not_trained", "reason": "No valid training data available"}

    if len(X) < 10:
        logger.warning("Too few rows (%d) — need at least 10 to train.", len(X))
        return {"status": "not_trained", "reason": "Not enough valid training data"}

    if y.nunique() < 2:
        logger.warning(
            "Only %d class(es) in target — need at least 2.", y.nunique()
        )
        return {"status": "not_trained", "reason": "Not enough valid training data"}

    (X_train, X_test, y_train, y_test,
     sw_train, sw_test, split_method, train_years) = _split_data(
        X, y, years, sample_weight
    )
    logger.info("Training rows: %d | Test rows: %d", len(X_train), len(X_test))
    logger.info("Features: %s", list(X.columns))

    model_results: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    # ── Baseline evaluation ─────────────────────────────────────────────
    baseline_metrics = _evaluate_baseline(X_train, X_test, y_test)

    # ── RandomForestClassifier ──────────────────────────────────────────
    logger.info("Training RandomForestClassifier ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,        # limit depth to reduce overfitting on small data
        random_state=42,
        class_weight="balanced",
    )
    fit_kw: dict = {}
    if sw_train is not None:
        fit_kw["sample_weight"] = sw_train.values
    rf.fit(X_train, y_train, **fit_kw)

    rf_cal, rf_calibrated = _try_calibrate(rf, X_train, y_train, sw_train)
    rf_metrics = _evaluate(rf_cal, X_test, y_test)
    rf_metrics["calibrated"] = rf_calibrated
    model_results["RandomForestClassifier"] = rf_metrics
    trained_models["RandomForestClassifier"] = rf_cal
    logger.info(
        "RandomForestClassifier — accuracy=%.4f, f1_macro=%.4f%s",
        rf_metrics["accuracy"], rf_metrics["f1_macro"],
        " (calibrated)" if rf_calibrated else "",
    )

    # ── XGBClassifier (optional) ────────────────────────────────────────
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415

        logger.info("Training XGBClassifier ...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            verbosity=0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
        )
        xgb_fit_kw: dict = {}
        if sw_train is not None:
            xgb_fit_kw["sample_weight"] = sw_train.values
        xgb.fit(X_train, y_train, **xgb_fit_kw)

        xgb_cal, xgb_calibrated = _try_calibrate(xgb, X_train, y_train, sw_train)
        xgb_metrics = _evaluate(xgb_cal, X_test, y_test)
        xgb_metrics["calibrated"] = xgb_calibrated
        model_results["XGBClassifier"] = xgb_metrics
        trained_models["XGBClassifier"] = xgb_cal
        logger.info(
            "XGBClassifier — accuracy=%.4f, f1_macro=%.4f%s",
            xgb_metrics["accuracy"], xgb_metrics["f1_macro"],
            " (calibrated)" if xgb_calibrated else "",
        )
    except Exception as exc:
        logger.warning("XGBClassifier training failed — skipping. Reason: %s", exc)

    # ── Select best model (f1_macro, accuracy as tie-breaker) ───────────
    best_name = max(
        model_results,
        key=lambda name: (
            model_results[name]["f1_macro"],
            model_results[name]["accuracy"],
        ),
    )
    best_metrics = model_results[best_name]
    best_model   = trained_models[best_name]

    logger.info(
        "Best model: %s — accuracy=%.4f, f1_macro=%.4f",
        best_name, best_metrics["accuracy"], best_metrics["f1_macro"],
    )
    logger.info(
        "Ranking baseline — accuracy=%.4f, f1_macro=%s",
        baseline_metrics.get("accuracy") or 0,
        baseline_metrics.get("f1_macro") or "N/A",
    )
    if (baseline_metrics.get("accuracy") or 0) >= best_metrics["accuracy"]:
        logger.warning(
            "ML model does NOT outperform the ranking baseline "
            "(model=%.4f, baseline=%.4f). "
            "Consider adding more data or features.",
            best_metrics["accuracy"],
            baseline_metrics.get("accuracy") or 0,
        )
    logger.info(
        "Confusion matrix (%s):\n%s",
        best_name, best_metrics["confusion_matrix"],
    )

    # ── Persist model ────────────────────────────────────────────────────
    ensure_model_dir()
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Best model saved to %s", MODEL_PATH)

    # ── Persist metrics to SQLite ────────────────────────────────────────
    try:
        create_model_metrics_table()
        baseline_note = (
            f"baseline_acc={baseline_metrics.get('accuracy', 'N/A'):.4f}; "
            f"baseline_f1={baseline_metrics.get('f1_macro', 'N/A'):.4f}; "
        ) if baseline_metrics.get("accuracy") is not None else ""
        _insert_model_metrics(
            model_name=best_name,
            accuracy=best_metrics["accuracy"],
            f1_macro=best_metrics["f1_macro"],
            notes=(
                f"split={split_method}; "
                f"train_rows={len(X_train)}; test_rows={len(X_test)}; "
                f"calibrated={best_metrics.get('calibrated', False)}; "
                f"{baseline_note}"
            ),
        )
    except Exception as exc:
        logger.warning("Could not write to model_metrics table: %s", exc)

    return {
        "status": "trained",
        "best_model": best_name,
        "accuracy": best_metrics["accuracy"],
        "f1_macro": best_metrics["f1_macro"],
        "calibrated": best_metrics.get("calibrated", False),
        "baseline": baseline_metrics,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_years": train_years,
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
