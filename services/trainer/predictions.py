"""
predictions.py
--------------
Generates FIFA World Cup 2026 group stage predictions using the trained model.

All 72 group stage matchups (12 groups × 6 matches each) are hardcoded from
the official FIFA 2026 draw. Predictions are stored in the match_predictions
table with confidence scores, confidence bands, and human-readable explanations.

Confidence is the max probability from predict_proba on a (calibrated) model.
It represents model certainty given available features — not a guaranteed
win probability. Football has inherent variance that no model fully captures.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import json

import joblib
import numpy as np
import pandas as pd

from model import (
    DEFAULT_CLOSENESS_THRESHOLD,
    DEFAULT_DRAW_THRESHOLD,
    METRICS_PATH,
    apply_draw_decision_rule,
    align_proba_to_class_order,
)

DB_PATH    = "/app/data/worldcup.db"
MODEL_PATH = Path("/app/data/models/model.pkl")

logger = logging.getLogger(__name__)

ELO_DEFAULT  = 1500.0
RANK_DEFAULT = 150.0   # fallback for teams absent from raw_rankings

FEATURE_COLS = [
    "rank_diff",
    "elo_diff",
    "h2h_win_rate",
    "h2h_goal_diff",
    "stage_encoded",
    "is_host_a",
    "is_host_b",
    "current_rank_diff",
    "current_points_diff",
]

# ---------------------------------------------------------------------------
# Official FIFA World Cup 2026 group draw
# Source: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026
# 12 groups × 4 teams = 48 teams → 12 × C(4,2) = 72 group stage matches
# ---------------------------------------------------------------------------
GROUPS_2026: dict[str, list[str]] = {
    "A": ["Mexico",         "Korea Republic",        "South Africa",  "Czechia"],
    "B": ["Canada",         "Bosnia and Herzegovina","Qatar",          "Switzerland"],
    "C": ["Haiti",          "Brazil",                "Scotland",       "Morocco"],
    "D": ["United States",  "Australia",             "Paraguay",       "Türkiye"],
    "E": ["Côte d'Ivoire",  "Germany",               "Ecuador",        "Curaçao"],
    "F": ["Netherlands",    "Sweden",                "Japan",          "Tunisia"],
    "G": ["IR Iran",        "Belgium",               "New Zealand",    "Egypt"],
    "H": ["Saudi Arabia",   "Spain",                 "Uruguay",        "Cabo Verde"],
    "I": ["France",         "Iraq",                  "Senegal",        "Norway"],
    "J": ["Argentina",      "Austria",               "Algeria",        "Jordan"],
    "K": ["Portugal",       "Uzbekistan",            "Congo DR",       "Colombia"],
    "L": ["Ghana",          "England",               "Panama",         "Croatia"],
}

HOSTS_2026: set[str] = {"United States", "Canada", "Mexico"}

# Teams that never appeared in raw_matches (no WC history in our data)
_NO_HISTORY_TEAMS = {
    "South Africa", "Czechia", "Bosnia and Herzegovina", "Haiti", "Scotland",
    "Ecuador", "Curaçao", "Sweden", "New Zealand", "Cabo Verde",
    "Iraq", "Norway", "Austria", "Algeria", "Jordan",
    "Uzbekistan", "Congo DR", "Panama",
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _migrate_match_predictions(conn: sqlite3.Connection) -> None:
    """Safely add all expected columns to match_predictions if they are missing."""
    rows = conn.execute("PRAGMA table_info(match_predictions)").fetchall()
    existing = {row[1] for row in rows}
    migrations = [
        ("confidence_band", "TEXT"),
        ("explanation", "TEXT"),
        ("key_factors", "TEXT"),
        ("base_team_a_proba", "REAL"),
        ("base_draw_proba", "REAL"),
        ("base_team_b_proba", "REAL"),
        ("adjusted_team_a_proba", "REAL"),
        ("adjusted_draw_proba", "REAL"),
        ("adjusted_team_b_proba", "REAL"),
        ("squad_strength_a", "REAL"),
        ("squad_strength_b", "REAL"),
        ("squad_strength_diff", "REAL"),
        ("squad_coverage_a", "REAL"),
        ("squad_coverage_b", "REAL"),
        ("squad_coverage_tier_a", "TEXT"),
        ("squad_coverage_tier_b", "TEXT"),
        ("squad_adjustment_amount", "REAL"),
        ("squad_adjustment_applied", "INTEGER"),
        ("raw_model_confidence", "REAL"),
        ("adjusted_confidence", "REAL"),
        ("squad_adjusted_confidence", "REAL"),
        ("combined_strength_signal", "REAL"),
        ("confidence_boost_amount", "REAL"),
        ("confidence_cap_applied", "REAL"),
        ("ranking_baseline_confidence", "REAL"),
        ("historical_ml_confidence", "REAL"),
    ]
    for col, col_type in migrations:
        if col not in existing:
            conn.execute(
                f"ALTER TABLE match_predictions ADD COLUMN {col} {col_type}"
            )
            conn.commit()
            logger.info("match_predictions: added column '%s'.", col)


def _log_match_predictions_verification(conn: sqlite3.Connection) -> None:
    """Log squad column population after INSERT so dashboard issues are diagnosable."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN squad_strength_a IS NOT NULL
                      AND squad_strength_b IS NOT NULL THEN 1 ELSE 0 END) AS with_squad,
            SUM(CASE WHEN squad_adjustment_applied = 1 THEN 1 ELSE 0 END) AS adjusted,
            AVG(ABS(squad_adjustment_amount)) AS avg_abs_adj
        FROM match_predictions
        WHERE tournament_year = 2026
        """
    ).fetchone()
    if not row or row[0] == 0:
        logger.warning("match_predictions verification: no 2026 rows stored.")
        return
    total, with_squad, adjusted, avg_abs_adj = row
    logger.info(
        "match_predictions verification: %d rows, %d with squad values, "
        "%d adjusted, avg_abs_adjustment=%.4f",
        total,
        with_squad or 0,
        adjusted or 0,
        float(avg_abs_adj or 0.0),
    )
    sample = conn.execute(
        """
        SELECT team_a, team_b, squad_strength_a, squad_strength_b,
               adjusted_confidence, squad_adjusted_confidence
        FROM match_predictions
        WHERE tournament_year = 2026
          AND squad_strength_a IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    if sample:
        adj_conf = sample[4] if sample[4] is not None else sample[5]
        logger.info(
            "match_predictions sample: %s vs %s — squad_a=%.1f, squad_b=%.1f, "
            "adjusted_confidence=%.3f",
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            float(adj_conf or 0.0),
        )


def create_match_predictions_table() -> None:
    conn = _open_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS match_predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_year  INTEGER,
                group_name       TEXT,
                team_a           TEXT,
                team_b           TEXT,
                stage            TEXT,
                predicted_winner TEXT,
                confidence       REAL,
                confidence_band  TEXT,
                explanation      TEXT,
                key_factors      TEXT,
                run_at           TEXT
            )
        """)
        conn.commit()
        _migrate_match_predictions(conn)
    finally:
        conn.close()
    logger.info("match_predictions table ready.")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_latest_elo() -> dict[str, float]:
    conn = _open_db()
    try:
        df = pd.read_sql_query("SELECT team, year, elo_rating FROM team_elo", conn)
    except Exception as exc:
        logger.warning("Could not load team_elo: %s", exc)
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return {}
    idx = df.groupby("team")["year"].idxmax()
    return df.loc[idx].set_index("team")["elo_rating"].to_dict()


def _load_latest_rankings() -> pd.DataFrame:
    conn = _open_db()
    try:
        df = pd.read_sql_query("SELECT team, year, rank, points FROM raw_rankings", conn)
    except Exception as exc:
        logger.warning("Could not load raw_rankings: %s", exc)
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def _load_all_matches() -> pd.DataFrame:
    conn = _open_db()
    try:
        df = pd.read_sql_query(
            "SELECT team_a, team_b, score_a, score_b FROM raw_matches", conn
        )
    except Exception as exc:
        logger.warning("Could not load raw_matches: %s", exc)
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------------------
# Feature helpers for inference
# ---------------------------------------------------------------------------


def _get_rank(rankings_df: pd.DataFrame, team: str) -> float:
    if rankings_df.empty:
        return RANK_DEFAULT
    team_lower = team.strip().lower()
    rows = rankings_df[rankings_df["team"].str.strip().str.lower() == team_lower]
    if rows.empty:
        return RANK_DEFAULT
    return float(rows.loc[rows["year"].idxmax(), "rank"])


def _get_current_rank(rankings_df: pd.DataFrame, team: str) -> float:
    """Most recent ranking entry for a team (uses 2026 if available)."""
    return _get_rank(rankings_df, team)


def _get_current_points(rankings_df: pd.DataFrame, team: str) -> float:
    if rankings_df.empty or "points" not in rankings_df.columns:
        return 0.0
    team_lower = team.strip().lower()
    rows = rankings_df[rankings_df["team"].str.strip().str.lower() == team_lower]
    if rows.empty:
        return 0.0
    pts = rows.loc[rows["year"].idxmax(), "points"]
    try:
        return float(pts)
    except (ValueError, TypeError):
        return 0.0


def _compute_h2h(
    matches_df: pd.DataFrame, team_a: str, team_b: str
) -> tuple[float, float]:
    if matches_df.empty:
        return 0.5, 0.0
    mask = (
        ((matches_df["team_a"] == team_a) & (matches_df["team_b"] == team_b))
        | ((matches_df["team_a"] == team_b) & (matches_df["team_b"] == team_a))
    )
    h2h = matches_df[mask]
    if h2h.empty:
        return 0.5, 0.0

    wins_a = 0
    total_goal_diff = 0.0
    for _, r in h2h.iterrows():
        if r["team_a"] == team_a:
            diff = float(r["score_a"]) - float(r["score_b"])
            if r["score_a"] > r["score_b"]:
                wins_a += 1
        else:
            diff = float(r["score_b"]) - float(r["score_a"])
            if r["score_b"] > r["score_a"]:
                wins_a += 1
        total_goal_diff += diff
    return wins_a / len(h2h), total_goal_diff / len(h2h)


def _build_row(
    team_a: str,
    team_b: str,
    elo_map: dict[str, float],
    rankings_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> dict:
    elo_a = elo_map.get(team_a, ELO_DEFAULT)
    elo_b = elo_map.get(team_b, ELO_DEFAULT)

    rank_a = _get_rank(rankings_df, team_a)
    rank_b = _get_rank(rankings_df, team_b)

    curr_rank_a = _get_current_rank(rankings_df, team_a)
    curr_rank_b = _get_current_rank(rankings_df, team_b)

    curr_pts_a = _get_current_points(rankings_df, team_a)
    curr_pts_b = _get_current_points(rankings_df, team_b)

    h2h_win_rate, h2h_goal_diff = _compute_h2h(matches_df, team_a, team_b)

    return {
        "rank_diff":           rank_a - rank_b,
        "elo_diff":            elo_a - elo_b,
        "h2h_win_rate":        h2h_win_rate,
        "h2h_goal_diff":       h2h_goal_diff,
        "stage_encoded":       0,               # group stage
        "is_host_a":           int(team_a in HOSTS_2026),
        "is_host_b":           int(team_b in HOSTS_2026),
        "current_rank_diff":   curr_rank_a - curr_rank_b,
        "current_points_diff": curr_pts_a - curr_pts_b,
    }


# ---------------------------------------------------------------------------
# Confidence band
# ---------------------------------------------------------------------------


def _confidence_band(conf: float) -> str:
    if conf >= 0.70:
        return "High"
    if conf >= 0.55:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# 2026 final prediction blend (ML + ranking + ELO + squad — not training data)
# ---------------------------------------------------------------------------

_GROUP_STAGE_MAX_CONF = 0.88
_ORDINARY_WINNER_CAP = 0.82
_STRONG_AGREEMENT_CAP = 0.86
_MIN_DRAW_PROBA = 0.12
_PROBA_FLOOR = 0.01

# Final blend weights (sum = 1.0 when squad data available)
_W_RANK = 0.40
_W_ML = 0.30
_W_ELO_RANK = 0.10
_W_SQUAD = 0.20
# Fallback when squad unavailable
_W_RANK_NO_SQUAD = 0.45
_W_ML_NO_SQUAD = 0.35
_W_ELO_NO_SQUAD = 0.20


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _team_has_current_ranking(rankings_df: pd.DataFrame, team: str) -> bool:
    return _get_current_rank(rankings_df, team) < RANK_DEFAULT


def _combined_strength_signal(
    feat: dict,
    squad_diff: float,
    rankings_df: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> float:
    """Positive favors Team A; negative favors Team B."""
    squad_signal = _clip(squad_diff / 35.0, -1.0, 1.0)
    rank_signal = _clip((-feat["current_rank_diff"]) / 80.0, -1.0, 1.0)
    points_signal = _clip(feat["current_points_diff"] / 400.0, -1.0, 1.0)
    elo_signal = _clip(feat["elo_diff"] / 250.0, -1.0, 1.0)

    rank_ok = (
        _team_has_current_ranking(rankings_df, team_a)
        and _team_has_current_ranking(rankings_df, team_b)
    )
    pts_a = _get_current_points(rankings_df, team_a)
    pts_b = _get_current_points(rankings_df, team_b)
    points_ok = rank_ok and (pts_a > 0 or pts_b > 0)

    components: list[tuple[float, float]] = [(0.35, squad_signal)]
    if rank_ok:
        components.append((0.35, rank_signal))
    if points_ok:
        components.append((0.20, points_signal))
    components.append((0.10, elo_signal))

    total_w = sum(w for w, _ in components)
    if total_w <= 0:
        return 0.0
    return sum((w / total_w) * sig for w, sig in components)


def _coverage_boost_multiplier(
    tier_a: str,
    tier_b: str,
    predicted_winner: str,
    team_a: str,
    team_b: str,
) -> float:
    """
    Scale boost by data quality. Underdog Very Low coverage should not fully
    mute a boost when the predicted winner has solid squad data.
    """
    if predicted_winner == "Draw":
        return 0.0

    winner_tier = (
        tier_a if predicted_winner == team_a
        else tier_b if predicted_winner == team_b
        else "Unknown"
    )
    loser_tier = (
        tier_b if predicted_winner == team_a
        else tier_a if predicted_winner == team_b
        else "Unknown"
    )

    if winner_tier == "Very Low":
        return 0.0
    if winner_tier == "Low":
        return 0.35

    if winner_tier in ("High", "Medium"):
        if loser_tier == "Very Low":
            return 0.75
        if loser_tier == "Low":
            return 0.85
        if loser_tier in ("High", "Medium"):
            return 1.0
        return 0.60

    if loser_tier == "Very Low":
        return 0.35
    if "Low" in (winner_tier, loser_tier):
        return 0.60
    return 1.0


def _max_boost_from_favorite_signal(favorite_signal: float) -> float:
    fs = abs(favorite_signal)
    if fs >= 0.75:
        return 0.14
    if fs >= 0.50:
        return 0.10
    if fs >= 0.30:
        return 0.06
    return 0.03


def _signals_agree_with_winner(
    combined: float,
    predicted_winner: str,
    team_a: str,
    team_b: str,
) -> bool:
    if predicted_winner == "Draw":
        return False
    if predicted_winner == team_a:
        return combined > 0.08
    if predicted_winner == team_b:
        return combined < -0.08
    return False


def _winner_index(predicted_winner: str, team_a: str, team_b: str) -> int | None:
    if predicted_winner == team_a:
        return 2
    if predicted_winner == team_b:
        return 0
    if predicted_winner == "Draw":
        return 1
    return None


def _renormalize_proba(proba: list[float]) -> list[float]:
    p = [max(_PROBA_FLOOR, float(x)) for x in proba]
    total = sum(p)
    if total <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    return [x / total for x in p]


def _signal_to_proba(signal: float) -> list[float]:
    """
    Convert a directional strength signal to [team_b, draw, team_a] probabilities.
    signal > 0 favors team A; signal < 0 favors team B.
    """
    s = _clip(signal, -1.0, 1.0)
    if abs(s) < 0.08:
        return _renormalize_proba([0.28, 0.44, 0.28])

    strength = abs(s)
    if strength >= 0.70:
        fav = 0.58 + 0.24 * strength
        dog = max(_PROBA_FLOOR, 0.10 - 0.04 * strength)
        draw = max(_MIN_DRAW_PROBA, 1.0 - fav - dog)
        if s > 0:
            p_a, p_b = fav, dog
        else:
            p_b, p_a = fav, dog
    elif strength >= 0.30:
        fav = 0.34 + 0.38 * strength
        dog = max(_PROBA_FLOOR, 0.24 - 0.14 * strength)
        draw = max(_MIN_DRAW_PROBA, 1.0 - fav - dog)
        if s > 0:
            p_a, p_b = fav, dog
        else:
            p_b, p_a = fav, dog
    else:
        if s > 0:
            p_a = 0.32 + 0.28 * strength
            p_b = max(_PROBA_FLOOR, 0.28 - 0.12 * strength)
        else:
            p_b = 0.32 + 0.28 * strength
            p_a = max(_PROBA_FLOOR, 0.28 - 0.12 * strength)
        draw = max(_MIN_DRAW_PROBA, 1.0 - p_a - p_b)
    return _renormalize_proba([p_b, draw, p_a])


def _ranking_baseline_proba(feat: dict) -> list[float]:
    """FIFA current rank + points → soft outcome probabilities."""
    rank_signal = _clip((-feat["current_rank_diff"]) / 80.0, -1.0, 1.0)
    pts_signal = _clip(feat["current_points_diff"] / 400.0, -1.0, 1.0)
    combined = 0.70 * rank_signal + 0.30 * pts_signal
    return _signal_to_proba(combined)


def _elo_rank_agreement_proba(feat: dict) -> list[float]:
    """ELO and FIFA rank agreement → outcome probabilities."""
    rank_s = _clip((-feat["current_rank_diff"]) / 80.0, -1.0, 1.0)
    elo_s = _clip(feat["elo_diff"] / 250.0, -1.0, 1.0)

    if abs(rank_s) < 0.10 and abs(elo_s) < 0.10:
        return _renormalize_proba([0.30, 0.40, 0.30])

    if rank_s * elo_s > 0:
        signal = 0.55 * rank_s + 0.45 * elo_s
    else:
        signal = 0.35 * rank_s + 0.35 * elo_s
    return _signal_to_proba(signal)


def _squad_blend_coverage_scale(tier_a: str, tier_b: str) -> float:
    """Reduce squad influence when either team has sparse player stats."""
    tiers = (tier_a or "Unknown", tier_b or "Unknown")
    if "Very Low" in tiers:
        return 0.35
    if "Low" in tiers:
        return 0.60
    return 1.0


def _squad_strength_proba(
    squad_diff: float,
    tier_a: str,
    tier_b: str,
) -> list[float]:
    """Squad strength diff → soft probabilities (missing data handled via tiers)."""
    scale = _squad_blend_coverage_scale(tier_a, tier_b)
    signal = _clip(squad_diff / 35.0, -1.0, 1.0) * scale
    return _signal_to_proba(signal)


def _weighted_blend(
    components: list[tuple[float, list[float]]],
) -> list[float]:
    """Weighted sum of probability vectors, then renormalize."""
    total_w = sum(w for w, _ in components)
    if total_w <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    blended = [0.0, 0.0, 0.0]
    for w, proba in components:
        for i in range(3):
            blended[i] += (w / total_w) * proba[i]
    return _renormalize_proba(blended)


def _proba_to_winner(proba: list[float], team_a: str, team_b: str) -> str:
    idx = int(max(range(3), key=lambda i: proba[i]))
    if idx == 2:
        return team_a
    if idx == 0:
        return team_b
    return "Draw"


def _winner_confidence(proba: list[float], winner: str, team_a: str, team_b: str) -> float:
    idx = _winner_index(winner, team_a, team_b)
    if idx is None:
        return float(max(proba))
    return float(proba[idx])


def blend_final_2026_probabilities(
    ml_proba: list[float],
    feat: dict,
    team_a: str,
    team_b: str,
    squad_diff: float,
    tier_a: str,
    tier_b: str,
    rankings_df: pd.DataFrame,
    squad_data_ok: bool,
) -> tuple[list[float], dict]:
    """
    Final 2026 prediction blend:
      40% ranking baseline + 30% ML + 10% ELO/rank agreement + 20% squad
    (45/35/20 when squad unavailable), then confidence caps.

    proba order: [team_b, draw, team_a].
    """
    rank_proba = _ranking_baseline_proba(feat)
    elo_proba = _elo_rank_agreement_proba(feat)

    meta: dict = {
        "combined_strength_signal": 0.0,
        "confidence_boost_amount": 0.0,
        "confidence_cap_applied": _ORDINARY_WINNER_CAP,
        "squad_adjustment_amount": 0.0,
        "squad_adjustment_applied": 0,
        "signals_agree": False,
        "ranking_baseline_confidence": 0.0,
        "blend_weights": "",
    }

    if squad_data_ok:
        squad_proba = _squad_strength_proba(squad_diff, tier_a, tier_b)
        components = [
            (_W_RANK, rank_proba),
            (_W_ML, ml_proba),
            (_W_ELO_RANK, elo_proba),
            (_W_SQUAD, squad_proba),
        ]
        meta["blend_weights"] = (
            f"rank={_W_RANK}, ml={_W_ML}, elo={_W_ELO_RANK}, squad={_W_SQUAD}"
        )
        meta["squad_adjustment_applied"] = 1
    else:
        squad_proba = [1 / 3, 1 / 3, 1 / 3]
        components = [
            (_W_RANK_NO_SQUAD, rank_proba),
            (_W_ML_NO_SQUAD, ml_proba),
            (_W_ELO_NO_SQUAD, elo_proba),
        ]
        meta["blend_weights"] = (
            f"rank={_W_RANK_NO_SQUAD}, ml={_W_ML_NO_SQUAD}, "
            f"elo={_W_ELO_NO_SQUAD} (no squad)"
        )

    pre_cap = _weighted_blend(components)
    predicted_winner = _proba_to_winner(pre_cap, team_a, team_b)
    winner_idx = _winner_index(predicted_winner, team_a, team_b)
    initial_winner_p = (
        float(pre_cap[winner_idx]) if winner_idx is not None else 0.0
    )

    combined = _combined_strength_signal(
        feat, squad_diff if squad_data_ok else 0.0, rankings_df, team_a, team_b
    )
    meta["combined_strength_signal"] = round(combined, 4)
    meta["signals_agree"] = _signals_agree_with_winner(
        combined, predicted_winner, team_a, team_b
    )
    meta["ranking_baseline_confidence"] = round(
        _winner_confidence(rank_proba, predicted_winner, team_a, team_b), 6
    )

    if (
        winner_idx is not None
        and predicted_winner != "Draw"
        and meta["signals_agree"]
        and abs(combined) >= 0.35
    ):
        agree_boost = _max_boost_from_favorite_signal(abs(combined))
        cov_mult = (
            _coverage_boost_multiplier(
                tier_a, tier_b, predicted_winner, team_a, team_b
            )
            if squad_data_ok
            else 1.0
        )
        agree_boost *= cov_mult
        if agree_boost > 1e-6:
            before_boost = pre_cap[winner_idx]
            pre_cap = _apply_boost_to_winner(
                pre_cap, winner_idx, agree_boost, pre_cap[1]
            )
            meta["confidence_boost_amount"] = round(
                pre_cap[winner_idx] - before_boost, 6
            )
            meta["squad_adjustment_amount"] = meta["confidence_boost_amount"]

    if winner_idx is not None and predicted_winner != "Draw":
        cap = _winner_probability_cap(abs(combined), feat, squad_diff)
        meta["confidence_cap_applied"] = cap
        before = pre_cap[winner_idx]
        final_proba, applied_cap = _cap_winner_probability(pre_cap, winner_idx, cap)
        meta["confidence_cap_applied"] = applied_cap
        if max(final_proba) > _GROUP_STAGE_MAX_CONF:
            win_i = int(max(range(3), key=lambda i: final_proba[i]))
            final_proba, _ = _cap_winner_probability(
                final_proba, win_i, _GROUP_STAGE_MAX_CONF
            )
            meta["confidence_cap_applied"] = _GROUP_STAGE_MAX_CONF
    else:
        final_proba = pre_cap

    if winner_idx is not None:
        meta["confidence_boost_amount"] = round(
            final_proba[winner_idx] - initial_winner_p, 6
        )
        meta["squad_adjustment_amount"] = meta["confidence_boost_amount"]

    meta["pre_cap_confidence"] = round(float(max(pre_cap)), 6)
    return final_proba, meta


def _winner_probability_cap(
    combined_abs: float,
    feat: dict,
    squad_diff: float,
) -> float:
    rank_gap = abs(feat["current_rank_diff"])
    huge_agreement = (
        combined_abs >= 0.75
        and rank_gap >= 40
        and abs(squad_diff) >= 20
        and abs(feat["elo_diff"]) >= 80
    )
    cap = _STRONG_AGREEMENT_CAP if huge_agreement else _ORDINARY_WINNER_CAP
    return min(_GROUP_STAGE_MAX_CONF, cap)


def _apply_boost_to_winner(
    proba: list[float],
    winner_idx: int,
    boost: float,
    base_draw: float,
) -> list[float]:
    if boost <= 0 or winner_idx is None:
        return list(proba)

    p = [float(x) for x in proba]
    min_draw = base_draw if base_draw < _MIN_DRAW_PROBA else _MIN_DRAW_PROBA
    p[winner_idx] += boost
    remaining = boost

    for idx in (0, 1, 2):
        if idx == winner_idx or remaining <= 1e-9:
            continue
        if idx == 1:
            can_take = max(0.0, p[idx] - min_draw)
        else:
            can_take = max(0.0, p[idx] - _PROBA_FLOOR)
        take = min(can_take, remaining)
        p[idx] -= take
        remaining -= take

    if remaining > 1e-9:
        for idx in (0, 1, 2):
            if idx == winner_idx:
                continue
            can_take = max(0.0, p[idx] - (_PROBA_FLOOR if idx != 1 else min_draw))
            take = min(can_take, remaining)
            p[idx] -= take
            remaining -= take

    return _renormalize_proba(p)


def _cap_winner_probability(
    proba: list[float],
    winner_idx: int,
    cap: float,
) -> tuple[list[float], float]:
    p = list(proba)
    if p[winner_idx] <= cap:
        return p, cap
    excess = p[winner_idx] - cap
    p[winner_idx] = cap
    others = [i for i in range(3) if i != winner_idx]
    pool = sum(p[i] for i in others)
    if pool > 0:
        for i in others:
            p[i] += excess * (p[i] / pool)
    return _renormalize_proba(p), cap


def calibrate_2026_probabilities(
    base_proba: list[float],
    feat: dict,
    team_a: str,
    team_b: str,
    predicted_winner: str,
    squad_diff: float,
    tier_a: str,
    tier_b: str,
    rankings_df: pd.DataFrame,
    squad_data_ok: bool,
) -> tuple[list[float], dict]:
    """
    Post-model calibration: agreement-based boost from squad + FIFA + ELO signals.

    Returns (adjusted_proba, metadata dict).
    proba order: [team_b, draw, team_a].
    """
    meta: dict = {
        "combined_strength_signal": 0.0,
        "confidence_boost_amount": 0.0,
        "confidence_cap_applied": _ORDINARY_WINNER_CAP,
        "squad_adjustment_amount": 0.0,
        "squad_adjustment_applied": 0,
        "signals_agree": False,
    }
    if len(base_proba) != 3:
        return list(base_proba), meta

    proba = list(base_proba)
    base_draw = float(base_proba[1])
    winner_idx = _winner_index(predicted_winner, team_a, team_b)

    if not squad_data_ok:
        return proba, meta

    combined = _combined_strength_signal(
        feat, squad_diff, rankings_df, team_a, team_b
    )
    meta["combined_strength_signal"] = round(combined, 4)

    agrees = _signals_agree_with_winner(combined, predicted_winner, team_a, team_b)
    meta["signals_agree"] = agrees

    if not agrees or winner_idx is None or winner_idx == 1:
        return proba, meta

    favorite_signal = abs(combined)
    max_boost = _max_boost_from_favorite_signal(favorite_signal)
    cov_mult = _coverage_boost_multiplier(
        tier_a, tier_b, predicted_winner, team_a, team_b
    )
    boost = max_boost * cov_mult

    if boost < 1e-6:
        return proba, meta

    before_winner = proba[winner_idx]
    proba = _apply_boost_to_winner(proba, winner_idx, boost, base_draw)
    meta["confidence_boost_amount"] = round(proba[winner_idx] - before_winner, 6)
    meta["squad_adjustment_amount"] = meta["confidence_boost_amount"]
    meta["squad_adjustment_applied"] = 1

    cap = _winner_probability_cap(favorite_signal, feat, squad_diff)
    meta["confidence_cap_applied"] = cap
    proba, applied_cap = _cap_winner_probability(proba, winner_idx, cap)
    meta["confidence_cap_applied"] = applied_cap

    if max(proba) > _GROUP_STAGE_MAX_CONF:
        proba, _ = _cap_winner_probability(
            proba, int(max(range(3), key=lambda i: proba[i])), _GROUP_STAGE_MAX_CONF
        )
        meta["confidence_cap_applied"] = _GROUP_STAGE_MAX_CONF

    return proba, meta


# ---------------------------------------------------------------------------
# Human-readable explanation
# ---------------------------------------------------------------------------


def _build_explanation(
    team_a: str,
    team_b: str,
    feat: dict,
    predicted_winner: str,
    confidence: float,
    rankings_df: pd.DataFrame,
    elo_map: dict[str, float],
    matches_df: pd.DataFrame,
    squad_a: float | None = None,
    squad_b: float | None = None,
    tier_a: str = "",
    tier_b: str = "",
    adj_amount: float = 0.0,
    adj_applied: bool = False,
    combined_signal: float = 0.0,
    boost_amount: float = 0.0,
    cap_applied: float = _ORDINARY_WINNER_CAP,
    signals_agree: bool = False,
) -> tuple[str, str]:
    """Return (explanation, key_factors) as short human-readable strings."""
    rank_a = _get_current_rank(rankings_df, team_a)
    rank_b = _get_current_rank(rankings_df, team_b)
    elo_a  = elo_map.get(team_a, ELO_DEFAULT)
    elo_b  = elo_map.get(team_b, ELO_DEFAULT)

    a_missing = team_a in _NO_HISTORY_TEAMS or elo_a == ELO_DEFAULT
    b_missing = team_b in _NO_HISTORY_TEAMS or elo_b == ELO_DEFAULT

    factors: list[str] = []

    rank_a_str = f"#{int(rank_a)}" if rank_a < RANK_DEFAULT else "unranked"
    rank_b_str = f"#{int(rank_b)}" if rank_b < RANK_DEFAULT else "unranked"

    factors.append(f"FIFA ranking: {team_a} {rank_a_str} vs {team_b} {rank_b_str}")

    if abs(feat["elo_diff"]) > 10:
        stronger_elo = team_a if feat["elo_diff"] > 0 else team_b
        factors.append(f"ELO advantage: {stronger_elo} ({elo_a:.0f} vs {elo_b:.0f})")

    if feat["is_host_a"]:
        factors.append(f"Host nation: {team_a}")
    if feat["is_host_b"]:
        factors.append(f"Host nation: {team_b}")

    h2h_count = int(len(matches_df[
        ((matches_df["team_a"] == team_a) & (matches_df["team_b"] == team_b)) |
        ((matches_df["team_a"] == team_b) & (matches_df["team_b"] == team_a))
    ]))
    if h2h_count > 0:
        factors.append(
            f"H2H: {h2h_count} prior WC meeting(s), "
            f"{team_a} win rate {feat['h2h_win_rate']:.0%}"
        )
    else:
        factors.append("H2H: no prior World Cup meetings")

    if a_missing or b_missing:
        missing_names = [t for t, m in [(team_a, a_missing), (team_b, b_missing)] if m]
        factors.append(
            f"Limited WC history for {', '.join(missing_names)} — confidence less reliable"
        )

    if squad_a is not None and squad_b is not None and squad_a > 0:
        factors.append(
            f"Squad strength: {team_a} {squad_a:.0f}/100 vs {team_b} {squad_b:.0f}/100"
        )
        if tier_a in ("Low", "Very Low") or tier_b in ("Low", "Very Low"):
            low_teams = [
                t for t, tier in ((team_a, tier_a), (team_b, tier_b))
                if tier in ("Low", "Very Low")
            ]
            factors.append(
                f"Low player-stat coverage ({', '.join(low_teams)}); "
                "missing players treated as unknown"
            )
    factors.append(
        "Final blend: 40% FIFA ranking, 30% historical ML, "
        "10% ELO/rank agreement, 20% squad/player stats"
    )
    if adj_applied and abs(boost_amount) >= 0.005:
        factors.append(
            f"Post-blend cap adjustment: {boost_amount*100:+.1f}pp on final winner"
        )
    if abs(combined_signal) >= 0.05:
        favored_side = team_a if combined_signal > 0 else team_b
        factors.append(
            f"Combined strength signal: {combined_signal:+.2f} (favors {favored_side})"
        )
    if cap_applied < _GROUP_STAGE_MAX_CONF + 0.001:
        factors.append(f"Confidence cap: {cap_applied:.0%} max for group stage")

    key_factors = "; ".join(factors)

    favored = predicted_winner if predicted_winner != "Draw" else "neither team"
    conf_note = (
        "with high model confidence" if confidence >= 0.70
        else "with moderate model confidence" if confidence >= 0.55
        else "with low model confidence — outcome uncertain"
    )
    if predicted_winner == "Draw":
        expl = f"Model predicts a draw {conf_note}."
    else:
        diff = abs(feat["current_rank_diff"])
        if diff > 30:
            strength = "significant ranking gap"
        elif diff > 10:
            strength = "moderate ranking difference"
        else:
            strength = "similar current rankings"
        expl = (
            f"{favored} is favored ({conf_note}) based on {strength}. "
            + ("⚠️ Interpret carefully: limited historical data for one or both teams."
               if (a_missing or b_missing) else "")
        ).strip()
        if squad_a is not None and squad_b is not None:
            if tier_a in ("Low", "Very Low") or tier_b in ("Low", "Very Low"):
                expl += (
                    " Low player-stat coverage: squad influence was reduced and "
                    "FIFA-ranking fallback blended into squad scores."
                )
            elif abs(squad_a - squad_b) > 5:
                stronger = team_a if squad_a > squad_b else team_b
                expl += (
                    f" Current squad profile favors {stronger} "
                    f"(active players, league quality, production)."
                )
        if predicted_winner != "Draw" and signals_agree:
            expl += (
                f" Final confidence blends historical ML, FIFA ranking, ELO, and "
                f"current squad/player statistics — all favoring {predicted_winner}. "
                f"Capped at {cap_applied:.0%} to avoid overstatement."
            )
        elif predicted_winner != "Draw":
            expl += (
                " Final confidence combines ranking, ELO, historical ML, and squad "
                "signals; sources do not fully agree, so confidence stays moderate."
            )

    return expl, key_factors


# ---------------------------------------------------------------------------
# Draw decision rule (loaded from metrics.json after training)
# ---------------------------------------------------------------------------


def _load_draw_rule_params() -> tuple[float, float]:
    """Read tuned draw/closeness thresholds from metrics.json."""
    draw_th = DEFAULT_DRAW_THRESHOLD
    close_th = DEFAULT_CLOSENESS_THRESHOLD
    if not METRICS_PATH.exists():
        return draw_th, close_th
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
        rule = metrics.get("draw_decision_rule") or {}
        if rule.get("enabled"):
            draw_th = float(rule.get("draw_threshold", draw_th))
            close_th = float(rule.get("closeness_threshold", close_th))
            logger.info(
                "Using draw decision rule: draw_threshold=%.2f, "
                "closeness_threshold=%.2f",
                draw_th, close_th,
            )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not load draw rule from metrics.json (%s) — defaults "
            "draw=%.2f close=%.2f",
            exc, draw_th, close_th,
        )
    return draw_th, close_th


# ---------------------------------------------------------------------------
# Main prediction runner
# ---------------------------------------------------------------------------


def run_2026_predictions() -> None:
    """Predict outcomes for all 72 FIFA World Cup 2026 group stage matches."""
    if not MODEL_PATH.exists():
        logger.warning(
            "No trained model at %s — skipping 2026 predictions.", MODEL_PATH
        )
        return

    try:
        model = joblib.load(MODEL_PATH)
        if hasattr(model, "classes_"):
            logger.info(
                "Loaded model classes_: %s — predict_proba: "
                "[team_b=0, draw=1, team_a=2]",
                list(model.classes_),
            )
        else:
            logger.info(
                "Loaded model has no classes_ attribute; "
                "assuming proba order [team_b, draw, team_a]."
            )
    except Exception as exc:
        logger.warning(
            "Failed to load model: %s — skipping 2026 predictions.", exc
        )
        return

    logger.info(
        "Running 2026 group stage predictions (final blend: "
        "rank=%.0f%%, ml=%.0f%%, elo=%.0f%%, squad=%.0f%%) ...",
        _W_RANK * 100, _W_ML * 100, _W_ELO_RANK * 100, _W_SQUAD * 100,
    )

    elo_map     = _load_latest_elo()
    rankings_df = _load_latest_rankings()
    matches_df  = _load_all_matches()

    # Log missing current rankings
    all_2026_teams = {t for teams in GROUPS_2026.values() for t in teams}
    has_current_rank = {
        t for t in all_2026_teams
        if _get_current_rank(rankings_df, t) < RANK_DEFAULT
    }
    missing_rank = all_2026_teams - has_current_rank
    if missing_rank:
        logger.warning(
            "%d 2026 team(s) have no current ranking data "
            "(using fallback rank=%d): %s",
            len(missing_rank), int(RANK_DEFAULT), sorted(missing_rank),
        )

    create_match_predictions_table()

    # ── Squad strength setup (optional) ─────────────────────────────────────
    squad_available = False
    try:
        from squad_features import (  # noqa: PLC0415
            explain_squad_strength,
            get_squad_strength_features,
            setup_squad_features,
        )
        squad_available = setup_squad_features()
        if squad_available:
            logger.info("Squad strength features loaded — applying adjustment to predictions.")
        else:
            logger.warning(
                "Squad strength not available — predictions will show empty squad "
                "columns. Ensure collector loaded wc_players_with_stats.csv, then "
                "re-run trainer."
            )
    except ImportError as exc:
        logger.warning(
            "squad_features import failed (%s) — skipping squad adjustment.", exc
        )
    except Exception as exc:
        logger.warning("Squad features setup failed: %s — continuing without.", exc)

    draw_threshold, closeness_threshold = _load_draw_rule_params()

    run_at = datetime.now(timezone.utc).isoformat()
    prediction_rows = []

    for group_name, teams in GROUPS_2026.items():
        for team_a, team_b in combinations(teams, 2):
            feat = _build_row(team_a, team_b, elo_map, rankings_df, matches_df)
            X = pd.DataFrame([feat])[FEATURE_COLS]

            try:
                proba_raw = model.predict_proba(X)
                proba_aligned = align_proba_to_class_order(model, proba_raw)[0]
                proba = list(proba_aligned)
                predicted_class = apply_draw_decision_rule(
                    np.array(proba),
                    draw_threshold=draw_threshold,
                    closeness_threshold=closeness_threshold,
                )
                raw_confidence = float(proba[predicted_class])
            except Exception as exc:
                logger.warning(
                    "Prediction failed for %s vs %s: %s — using default.",
                    team_a, team_b, exc,
                )
                proba = [0.33, 0.33, 0.34]
                predicted_class, raw_confidence = 2, 0.34

            ml_proba = list(proba)
            ml_confidence = raw_confidence

            sq_a, sq_b, sq_diff = 0.0, 0.0, 0.0
            cov_a, cov_b = 0.0, 0.0
            tier_a, tier_b = "", ""
            sq_ok = False
            adj_applied = 0
            adj_amount = 0.0
            boost_amount = 0.0
            combined_signal = 0.0
            cap_applied = _ORDINARY_WINNER_CAP
            signals_agree = False
            ranking_conf = 0.0
            blend_meta: dict = {}

            if squad_available:
                try:
                    (
                        sq_a, sq_b, sq_diff, sq_ok,
                        tier_a, tier_b, cov_a, cov_b, _, _,
                    ) = get_squad_strength_features(team_a, team_b)
                except Exception as exc:
                    logger.warning(
                        "Squad features failed for %s vs %s: %s",
                        team_a, team_b, exc,
                    )

            try:
                proba, blend_meta = blend_final_2026_probabilities(
                    ml_proba,
                    feat,
                    team_a,
                    team_b,
                    sq_diff,
                    tier_a,
                    tier_b,
                    rankings_df,
                    squad_available and sq_ok,
                )
            except Exception as exc:
                logger.warning(
                    "Final blend failed for %s vs %s: %s — using ML only.",
                    team_a, team_b, exc,
                )
                proba = list(ml_proba)
                blend_meta = {}

            predicted_winner = _proba_to_winner(proba, team_a, team_b)
            adj_confidence = float(max(proba))
            historical_ml_conf = _winner_confidence(
                ml_proba, predicted_winner, team_a, team_b
            )
            adj_applied = blend_meta.get("squad_adjustment_applied", 0)
            adj_amount = blend_meta.get("squad_adjustment_amount", 0.0)
            boost_amount = blend_meta.get("confidence_boost_amount", 0.0)
            combined_signal = blend_meta.get("combined_strength_signal", 0.0)
            cap_applied = blend_meta.get(
                "confidence_cap_applied", _ORDINARY_WINNER_CAP
            )
            signals_agree = blend_meta.get("signals_agree", False)
            ranking_conf = blend_meta.get("ranking_baseline_confidence", 0.0)

            confidence = adj_confidence
            band = _confidence_band(confidence)

            expl, key_factors = _build_explanation(
                team_a, team_b, feat, predicted_winner, confidence,
                rankings_df, elo_map, matches_df,
                squad_a=sq_a if squad_available else None,
                squad_b=sq_b if squad_available else None,
                tier_a=tier_a,
                tier_b=tier_b,
                adj_amount=adj_amount,
                adj_applied=bool(adj_applied),
                combined_signal=combined_signal,
                boost_amount=boost_amount,
                cap_applied=cap_applied,
                signals_agree=signals_agree,
            )

            if squad_available:
                try:
                    sq_expl = explain_squad_strength(team_a, team_b, adj_amount)
                    if sq_expl:
                        expl = expl.rstrip(".") + ". " + sq_expl
                except Exception:
                    pass

            prediction_rows.append({
                "tournament_year": 2026,
                "group_name": group_name,
                "team_a": team_a,
                "team_b": team_b,
                "stage": "Group Stage",
                "predicted_winner": predicted_winner,
                "confidence": confidence,
                "confidence_band": band,
                "explanation": expl,
                "key_factors": key_factors,
                "run_at": run_at,
                "base_team_b_proba": round(ml_proba[0], 6),
                "base_draw_proba": round(ml_proba[1], 6),
                "base_team_a_proba": round(ml_proba[2], 6),
                "adjusted_team_b_proba": round(proba[0], 6),
                "adjusted_draw_proba": round(proba[1], 6),
                "adjusted_team_a_proba": round(proba[2], 6),
                "squad_strength_a": round(sq_a, 2) if squad_available else None,
                "squad_strength_b": round(sq_b, 2) if squad_available else None,
                "squad_strength_diff": round(sq_diff, 2) if squad_available else None,
                "squad_coverage_a": round(cov_a, 4) if squad_available else None,
                "squad_coverage_b": round(cov_b, 4) if squad_available else None,
                "squad_coverage_tier_a": tier_a or None,
                "squad_coverage_tier_b": tier_b or None,
                "squad_adjustment_amount": round(adj_amount, 6) if adj_applied else None,
                "squad_adjustment_applied": adj_applied,
                "raw_model_confidence": round(ml_confidence, 6),
                "historical_ml_confidence": round(historical_ml_conf, 6),
                "ranking_baseline_confidence": round(ranking_conf, 6),
                "adjusted_confidence": round(adj_confidence, 6),
                "squad_adjusted_confidence": round(adj_confidence, 6),
                "combined_strength_signal": round(combined_signal, 4),
                "confidence_boost_amount": round(boost_amount, 6) if adj_applied else None,
                "confidence_cap_applied": round(cap_applied, 4),
            })

    conn = _open_db()
    try:
        _migrate_match_predictions(conn)
        conn.execute(
            "DELETE FROM match_predictions WHERE tournament_year = ?", (2026,)
        )
        existing_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(match_predictions)").fetchall()
        }
        base_cols = [
            "tournament_year", "group_name", "team_a", "team_b", "stage",
            "predicted_winner", "confidence", "confidence_band", "explanation",
            "key_factors", "run_at",
        ]
        extra_cols = [
            "base_team_b_proba", "base_draw_proba", "base_team_a_proba",
            "adjusted_team_b_proba", "adjusted_draw_proba", "adjusted_team_a_proba",
            "squad_strength_a", "squad_strength_b", "squad_strength_diff",
            "squad_coverage_a", "squad_coverage_b",
            "squad_coverage_tier_a", "squad_coverage_tier_b",
            "squad_adjustment_amount", "squad_adjustment_applied",
            "raw_model_confidence", "adjusted_confidence", "squad_adjusted_confidence",
            "combined_strength_signal", "confidence_boost_amount",
            "confidence_cap_applied",
            "ranking_baseline_confidence", "historical_ml_confidence",
        ]
        insert_cols = base_cols + [c for c in extra_cols if c in existing_cols]
        placeholders = ", ".join("?" * len(insert_cols))
        sql = (
            f"INSERT INTO match_predictions ({', '.join(insert_cols)}) "
            f"VALUES ({placeholders})"
        )
        tuples = [
            tuple(row.get(c) for c in insert_cols)
            for row in prediction_rows
        ]
        conn.executemany(sql, tuples)
        conn.commit()
        _log_match_predictions_verification(conn)
    finally:
        conn.close()

    high = sum(1 for r in prediction_rows if r["confidence_band"] == "High")
    medium = sum(1 for r in prediction_rows if r["confidence_band"] == "Medium")
    low = sum(1 for r in prediction_rows if r["confidence_band"] == "Low")
    adj_count = sum(1 for r in prediction_rows if r.get("squad_adjustment_applied") == 1)
    logger.info(
        "Stored %d 2026 group stage predictions "
        "(High=%d, Medium=%d, Low=%d, squad_adj=%d).",
        len(prediction_rows), high, medium, low, adj_count,
    )
