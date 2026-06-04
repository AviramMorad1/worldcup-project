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

import joblib
import pandas as pd

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
        ("confidence_band",          "TEXT"),
        ("explanation",              "TEXT"),
        ("key_factors",              "TEXT"),
        # Squad-strength columns (always created; NULL when squad data not available)
        ("squad_strength_a",         "REAL"),
        ("squad_strength_b",         "REAL"),
        ("squad_strength_diff",      "REAL"),
        ("squad_adjustment_applied", "INTEGER"),
        ("raw_model_confidence",     "REAL"),
        ("squad_adjusted_confidence","REAL"),
    ]
    for col, col_type in migrations:
        if col not in existing:
            conn.execute(
                f"ALTER TABLE match_predictions ADD COLUMN {col} {col_type}"
            )
            conn.commit()
            logger.info("match_predictions: added column '%s'.", col)


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

    return expl, key_factors


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
    except Exception as exc:
        logger.warning(
            "Failed to load model: %s — skipping 2026 predictions.", exc
        )
        return

    logger.info("Running 2026 group stage predictions ...")

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
            apply_squad_adjustment,
            explain_squad_strength,
            get_squad_strength_features,
            setup_squad_features,
        )
        squad_available = setup_squad_features()
        if squad_available:
            logger.info("Squad strength features loaded — applying adjustment to predictions.")
        else:
            logger.info("Squad strength not available — predictions use ML model only.")
    except ImportError:
        logger.debug("squad_features module not found — skipping squad adjustment.")
    except Exception as exc:
        logger.warning("Squad features setup failed: %s — continuing without.", exc)

    run_at = datetime.now(timezone.utc).isoformat()
    prediction_rows = []

    for group_name, teams in GROUPS_2026.items():
        for team_a, team_b in combinations(teams, 2):
            feat = _build_row(team_a, team_b, elo_map, rankings_df, matches_df)
            X = pd.DataFrame([feat])[FEATURE_COLS]

            try:
                proba           = list(model.predict_proba(X)[0])
                predicted_class = int(max(range(len(proba)), key=lambda i: proba[i]))
                raw_confidence  = float(max(proba))
            except Exception as exc:
                logger.warning(
                    "Prediction failed for %s vs %s: %s — using default.",
                    team_a, team_b, exc,
                )
                proba = [0.33, 0.33, 0.34]
                predicted_class, raw_confidence = 2, 0.34

            # Squad adjustment
            sq_a, sq_b, sq_diff = 0.0, 0.0, 0.0
            adj_applied = 0
            adj_confidence = raw_confidence

            if squad_available:
                try:
                    sq_a, sq_b, sq_diff, sq_data_ok = get_squad_strength_features(
                        team_a, team_b
                    )
                    adj_proba, adj_applied_bool = apply_squad_adjustment(
                        proba, sq_a, sq_b, sq_data_ok
                    )
                    if adj_applied_bool:
                        proba = adj_proba
                        predicted_class = int(
                            max(range(len(proba)), key=lambda i: proba[i])
                        )
                        adj_confidence  = float(max(proba))
                        adj_applied     = 1
                except Exception as exc:
                    logger.warning(
                        "Squad adjustment failed for %s vs %s: %s",
                        team_a, team_b, exc,
                    )

            if predicted_class == 2:
                predicted_winner = team_a
            elif predicted_class == 0:
                predicted_winner = team_b
            else:
                predicted_winner = "Draw"

            confidence = adj_confidence
            band = _confidence_band(confidence)

            expl, key_factors = _build_explanation(
                team_a, team_b, feat, predicted_winner, confidence,
                rankings_df, elo_map, matches_df,
            )

            # Append squad explanation
            if squad_available and sq_a > 0:
                try:
                    sq_expl = explain_squad_strength(team_a, team_b)
                    if sq_expl:
                        expl = expl.rstrip(".") + " " + sq_expl
                except Exception:
                    pass

            prediction_rows.append((
                2026, group_name, team_a, team_b,
                "Group Stage", predicted_winner,
                confidence, band, expl, key_factors, run_at,
                round(sq_a, 2) if sq_a else None,
                round(sq_b, 2) if sq_b else None,
                round(sq_diff, 2) if sq_diff else None,
                adj_applied,
                round(raw_confidence, 6),
                round(adj_confidence, 6),
            ))

    conn = _open_db()
    try:
        conn.execute(
            "DELETE FROM match_predictions WHERE tournament_year = ?", (2026,)
        )
        conn.executemany(
            "INSERT INTO match_predictions"
            " (tournament_year, group_name, team_a, team_b, stage,"
            "  predicted_winner, confidence, confidence_band, explanation,"
            "  key_factors, run_at, squad_strength_a, squad_strength_b,"
            "  squad_strength_diff, squad_adjustment_applied,"
            "  raw_model_confidence, squad_adjusted_confidence)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            prediction_rows,
        )
        conn.commit()
    finally:
        conn.close()

    high   = sum(1 for r in prediction_rows if r[7] == "High")
    medium = sum(1 for r in prediction_rows if r[7] == "Medium")
    low    = sum(1 for r in prediction_rows if r[7] == "Low")
    adj_count = sum(1 for r in prediction_rows if r[14] == 1)
    logger.info(
        "Stored %d 2026 group stage predictions "
        "(High=%d, Medium=%d, Low=%d, squad_adj=%d).",
        len(prediction_rows), high, medium, low, adj_count,
    )
