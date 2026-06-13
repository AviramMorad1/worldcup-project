"""World Cup 2026 Analytics Platform — Streamlit Dashboard."""
from __future__ import annotations

import html
import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = "/app/data/worldcup.db"
METRICS_JSON_PATH = "/app/data/models/metrics.json"
HIGH_CONF_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Page config  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Goal.ML · World Cup 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

/* ──────────────────────────────────────────────────────────────────────
   PITCH PRECISION — light-mode design system for Goal.ML
   Pitch Green · Championship Gold · Stadium Navy
   ────────────────────────────────────────────────────────────────────── */
:root {
    --surface:        #f8f9fa;   /* page background            */
    --surface-card:   #ffffff;   /* card / container surface   */
    --surface-low:    #f3f4f5;   /* nested low layer           */
    --surface-high:   #e7e8e9;   /* chips, tracks              */
    --border:         #e1e3e4;   /* 1px component borders      */
    --border-soft:    #edeeef;
    --navy:           #0d1b2a;   /* Stadium Navy — headings    */
    --ink:            #191c1d;   /* primary body text          */
    --ink-soft:       #40493d;   /* secondary text             */
    --muted:          #707a6c;   /* outline / muted labels     */
    --green:          #2e7d32;   /* Pitch Green — primary      */
    --green-dark:     #0d631b;   /* hover / deep accent        */
    --green-tint:     #e8f1e9;   /* soft green fill            */
    --gold:           #d4af37;   /* Championship Gold — accent */
    --gold-bright:    #f9d45a;
    --gold-tint:      #fbf3da;
    --error:          #ba1a1a;
    --shadow-1:       0 1px 2px rgba(13,27,42,0.04);
    --shadow-2:       0 4px 20px rgba(13,27,42,0.06);
}

/* ── App background ───────────────────────────────────────────────── */
.stApp { background-color: var(--surface); }

/* ── Body font ────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

/* ── Page chrome ──────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}

/* ── Hero ─────────────────────────────────────────────────────────── */
.hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--green);
    margin-bottom: 0.5rem;
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
}
.hero-eyebrow::before {
    content: "";
    width: 7px; height: 7px;
    border-radius: 999px;
    background: var(--green);
    box-shadow: 0 0 0 3px rgba(46,125,50,0.18);
}
.hero-title {
    font-family: 'Montserrat', sans-serif;
    font-size: clamp(2.2rem, 4.4vw, 3.3rem);
    font-weight: 800;
    letter-spacing: -0.025em;
    color: var(--navy);
    line-height: 1.05;
    margin-bottom: 0.55rem;
}
.hero-title .accent { color: var(--green); }
.hero-sub {
    color: var(--ink-soft);
    font-size: 0.95rem;
    line-height: 1.6;
    margin-top: 0;
    max-width: 720px;
    font-family: 'Inter', sans-serif;
}

/* ── Match Day Panel (signature element) ─────────────────────────── */
.match-panel-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.3rem 0 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.match-panel-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
}
.match-panel {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 1rem;
    margin: 0 0 0.4rem;
}
.match-card {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-top: 4px solid var(--gold);
    border-radius: 0.5rem;
    padding: 1.05rem 1.15rem 1rem;
    font-family: 'Inter', sans-serif;
    box-shadow: var(--shadow-2);
    position: relative;
}
.match-card-group {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.match-card-group::before { content: "★"; font-size: 0.7rem; }
.match-card-teams {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.3rem;
    margin-bottom: 0.8rem;
}
.match-team {
    font-family: 'Montserrat', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--muted);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.match-team.home { text-align: left; }
.match-team.away { text-align: right; }
.match-team.is-winner { color: var(--navy); }
.match-vs {
    font-size: 0.58rem;
    font-weight: 700;
    color: var(--muted);
    letter-spacing: 0.1em;
    padding: 0 0.45rem;
    flex-shrink: 0;
}
.match-bar-track {
    background: var(--surface-high);
    border-radius: 999px;
    height: 6px;
    margin-bottom: 0.65rem;
    overflow: hidden;
}
.match-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--green) 0%, var(--green-dark) 100%);
    border-radius: 999px;
}
.match-result {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.match-winner-label {
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'Montserrat', sans-serif;
    color: var(--navy);
}
.match-conf-label {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.95rem;
    color: var(--green);
    font-weight: 800;
    letter-spacing: -0.01em;
}

/* ── Prediction cards (fixture grid) ──────────────────────────────── */
.pred-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(275px, 1fr));
    gap: 1rem;
    margin: 0.2rem 0 0.6rem;
}
.pred-card {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-top: 4px solid var(--muted);
    border-radius: 0.5rem;
    padding: 1rem 1.1rem 0.95rem;
    box-shadow: var(--shadow-1);
    transition: box-shadow 0.16s ease, transform 0.16s ease;
    display: flex;
    flex-direction: column;
}
.pred-card:hover { box-shadow: var(--shadow-2); transform: translateY(-2px); }
.pred-card.band-High   { border-top-color: var(--green); }
.pred-card.band-Medium { border-top-color: var(--gold); }
.pred-card.band-Low    { border-top-color: var(--error); }

.pred-top {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.8rem;
}
.pred-group {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--muted);
    background: var(--surface-low); border: 1px solid var(--border);
    padding: 0.16rem 0.55rem; border-radius: 999px;
}
.pred-pill {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.16rem 0.55rem; border-radius: 999px;
}
.pred-pill.band-High   { background: #e8f1e9; color: #1e6023; }
.pred-pill.band-Medium { background: #fbf3da; color: #8a6d12; }
.pred-pill.band-Low    { background: #fbe9e9; color: #ba1a1a; }

.pred-teams { margin-bottom: 0.85rem; }
.pred-team-row {
    display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;
    padding: 0.1rem 0;
}
.pred-team-name {
    font-family: 'Montserrat', sans-serif; font-weight: 700; font-size: 1rem;
    color: var(--muted); letter-spacing: -0.01em;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.pred-team-row.winner .pred-team-name { color: var(--navy); }
.pred-team-tag {
    font-size: 0.56rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--green); flex-shrink: 0;
    display: inline-flex; align-items: center; gap: 0.2rem;
}
.pred-divider {
    display: flex; align-items: center; gap: 0.55rem;
    color: var(--muted); font-size: 0.56rem; font-weight: 700; letter-spacing: 0.12em;
    margin: 0.25rem 0;
}
.pred-divider::before, .pred-divider::after {
    content: ""; flex: 1; height: 1px; background: var(--border);
}

.pred-conf-row {
    display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.4rem;
}
.pred-conf-label {
    font-size: 0.6rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted);
}
.pred-conf-val {
    font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 1.2rem;
    color: var(--navy); letter-spacing: -0.02em;
}
.pred-bar-track {
    background: var(--surface-high); border-radius: 999px; height: 6px;
    overflow: hidden; margin-bottom: 0.8rem;
}
.pred-bar-fill { height: 100%; border-radius: 999px; }
.pred-bar-fill.band-High   { background: linear-gradient(90deg, var(--green), var(--green-dark)); }
.pred-bar-fill.band-Medium { background: linear-gradient(90deg, #e3c04a, var(--gold)); }
.pred-bar-fill.band-Low    { background: linear-gradient(90deg, #d98a8a, var(--error)); }

.pred-meta { margin-top: auto; display: flex; flex-wrap: wrap; gap: 0.35rem; }
.pred-chip {
    font-size: 0.6rem; font-weight: 600; color: var(--ink-soft);
    background: var(--surface-low); border: 1px solid var(--border);
    padding: 0.13rem 0.45rem; border-radius: 0.25rem; white-space: nowrap;
}

/* ── Inline filter bar ────────────────────────────────────────────── */
.filter-bar-label {
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); margin: 0.2rem 0 0.1rem;
}

/* ── KPI / Feature card (4px left-border accent per spec) ──────────── */
.kpi-card {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--green);
    border-radius: 0.5rem;
    padding: 1.05rem 1.25rem 0.95rem;
    min-height: 112px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
    box-sizing: border-box;
    box-shadow: var(--shadow-1);
}
.kpi-value {
    font-family: 'Montserrat', sans-serif;
    font-size: clamp(1.5rem, 2.4vw, 2rem);
    font-weight: 800;
    color: var(--navy);
    line-height: 1.1;
    letter-spacing: -0.025em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}
.kpi-label {
    font-size: 0.66rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-top: 0.35rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: 'Inter', sans-serif;
}
.kpi-caption {
    font-size: 0.66rem;
    color: var(--muted);
    margin-top: 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: 'Inter', sans-serif;
}

/* ── Section header ───────────────────────────────────────────────── */
.sec-eyebrow {
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--green);
    margin-bottom: 0.25rem;
    font-family: 'Inter', sans-serif;
}
.sec-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--navy);
    margin-bottom: 0.15rem;
}
.sec-sub {
    font-size: 0.86rem;
    color: var(--ink-soft);
    line-height: 1.55;
    margin-bottom: 0.9rem;
    font-family: 'Inter', sans-serif;
}

/* ── Divider ──────────────────────────────────────────────────────── */
.div-line {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.2rem 0 1.4rem;
}

/* ── Empty-state box ──────────────────────────────────────────────── */
.empty-box {
    border: 1px dashed var(--muted);
    border-radius: 0.5rem;
    padding: 2.4rem 1rem;
    text-align: center;
    margin: 0.6rem 0 1rem;
    background: var(--surface-card);
}
.empty-icon { font-size: 2rem; }
.empty-msg  { font-size: 0.92rem; margin-top: 0.4rem; color: var(--ink-soft); font-weight: 500; font-family: 'Inter', sans-serif; }
.empty-hint { font-size: 0.8rem; margin-top: 0.4rem; font-style: italic; color: var(--muted); font-family: 'Inter', sans-serif; }

/* ── Sidebar ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--surface-card);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    font-size: 0.83rem;
    color: var(--ink-soft);
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.66rem;
    font-weight: 700;
    color: var(--navy);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 0.9rem 0 0.35rem;
    font-family: 'Inter', sans-serif;
    border-left: 3px solid var(--green);
    padding-left: 0.5rem;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 0.9rem;
    font-weight: 600;
    font-family: 'Montserrat', sans-serif;
    color: var(--muted);
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--green) !important;
}
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] ~ div[role="presentation"] {
    background-color: var(--green) !important;
}
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--border);
    gap: 0.4rem;
}

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--green);
    color: #ffffff;
    border: none;
    border-radius: 0.25rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: background 0.15s ease;
}
.stButton > button:hover {
    background: var(--green-dark);
    color: #ffffff;
}
.stButton > button:focus:not(:active) {
    color: #ffffff;
    box-shadow: 0 0 0 2px rgba(46,125,50,0.35);
}

/* ── Streamlit metric → feature-card numbers ──────────────────────── */
[data-testid="stMetric"],
[data-testid="metric-container"] {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--green);
    border-radius: 0.5rem;
    padding: 0.85rem 1rem 0.75rem;
    box-shadow: var(--shadow-1);
}
[data-testid="stMetricValue"] {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: var(--navy) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.66rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    color: var(--muted) !important;
}

/* ── Alerts / info banners ────────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 0.5rem;
    font-family: 'Inter', sans-serif;
}

/* ── Expanders & containers ───────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    background: var(--surface-card);
    box-shadow: var(--shadow-1);
}

/* keep dataframes readable */
div[data-testid="stDataFrame"] { overflow-x: auto; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def database_exists() -> bool:
    return os.path.exists(DB_PATH)


def _conn() -> sqlite3.Connection | None:
    if not database_exists():
        return None
    try:
        c = sqlite3.connect(DB_PATH)
        c.execute("PRAGMA journal_mode=WAL")
        return c
    except sqlite3.Error:
        return None


def table_exists(name: str) -> bool:
    c = _conn()
    if c is None:
        return False
    try:
        row = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None
    except Exception:
        return False
    finally:
        c.close()


@st.cache_data(ttl=300)
def load_table(table: str, sql: str | None = None) -> pd.DataFrame:
    if not table_exists(table):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as c:
            return pd.read_sql_query(sql or f"SELECT * FROM {table}", c)  # noqa: S608
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_match_predictions(cache_key: str) -> pd.DataFrame:
    """Reload when trainer run_at or squad-populated row count changes."""
    if not table_exists("match_predictions"):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as c:
            return pd.read_sql_query("SELECT * FROM match_predictions", c)
    except Exception:
        return pd.DataFrame()


def match_predictions_cache_key() -> str:
    run_at = scalar(
        "SELECT COALESCE(MAX(run_at), '') FROM match_predictions "
        "WHERE tournament_year = 2026",
        "",
    )
    squad_n = scalar(
        "SELECT COUNT(*) FROM match_predictions "
        "WHERE tournament_year = 2026 "
        "AND squad_strength_a IS NOT NULL AND squad_strength_b IS NOT NULL",
        0,
    )
    return f"{run_at}:{squad_n}"


def count_2026_predictions_with_squad() -> int:
    return int(
        scalar(
            "SELECT COUNT(*) FROM match_predictions "
            "WHERE tournament_year = 2026 "
            "AND squad_strength_a IS NOT NULL "
            "AND squad_strength_b IS NOT NULL",
            0,
        )
        or 0
    )


@st.cache_data(ttl=300)
def scalar(sql: str, default=None):
    c = _conn()
    if c is None:
        return default
    try:
        row = c.execute(sql).fetchone()
        return row[0] if row else default
    except Exception:
        return default
    finally:
        c.close()


@st.cache_data(ttl=300)
def load_metrics() -> dict:
    if not os.path.exists(METRICS_JSON_PATH):
        return {}
    try:
        with open(METRICS_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt(val, digits: int = 3) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def fmt_int(val) -> str:
    if val is None:
        return "—"
    try:
        return f"{int(val):,}"
    except (TypeError, ValueError):
        return str(val)


def shorten_model_name(name: str | None) -> str:
    if not name:
        return "—"
    aliases = {
        "XGBClassifier": "XGBoost",
        "RandomForestClassifier": "RandomForest",
        "GradientBoostingClassifier": "GradBoost",
        "LogisticRegression": "LogReg",
    }
    return aliases.get(name, name[:12] + ("…" if len(name) > 12 else ""))


def compute_date_range_label(start: str | None, end: str | None) -> tuple[str, str]:
    """Return (value_str, caption_str) for the date-range KPI card."""
    if not start or not end:
        return "—", ""
    try:
        d0 = datetime.strptime(start, "%Y-%m-%d")
        d1 = datetime.strptime(end, "%Y-%m-%d")
        days = (d1 - d0).days
        value = f"{days} days"
        caption = f"{start} → {end}"
        return value, caption
    except Exception:
        return "Available", f"{start} → {end}"


# ---------------------------------------------------------------------------
# KPI data
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def get_kpis() -> dict:
    k: dict = {
        "reddit_posts": scalar("SELECT COUNT(*) FROM raw_reddit_posts", 0),
        "processed_posts": scalar("SELECT COUNT(*) FROM processed_posts", 0),
        "teams_with_sentiment": scalar(
            "SELECT COUNT(DISTINCT team) FROM team_sentiment_daily", 0
        ),
        "predictions_2026": scalar(
            "SELECT COUNT(*) FROM match_predictions WHERE tournament_year=2026", 0
        ),
    }

    # Reddit date range from actual post timestamps
    k["reddit_min_date"] = scalar(
        "SELECT date(datetime(MIN(created_utc),'unixepoch')) FROM raw_reddit_posts"
    )
    k["reddit_max_date"] = scalar(
        "SELECT date(datetime(MAX(created_utc),'unixepoch')) FROM raw_reddit_posts"
    )

    m = load_metrics()
    k["accuracy"] = m.get("accuracy")
    k["f1_macro"] = m.get("f1_macro")
    k["best_model"] = shorten_model_name(m.get("best_model"))
    k["model_status"] = m.get("status", "unknown")
    k["run_at"] = str(m.get("run_at", "—"))[:16]
    k["best_model_full"] = m.get("best_model", "—")
    return k


# ---------------------------------------------------------------------------
# UI component helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def get_hero_predictions() -> list[dict]:
    """Return top 3 most confident 2026 group stage predictions for the hero panel."""
    if not table_exists("match_predictions"):
        return []
    try:
        with sqlite3.connect(DB_PATH) as c:
            cols = [r[1] for r in c.execute("PRAGMA table_info(match_predictions)").fetchall()]
            conf_col = (
                "adjusted_confidence" if "adjusted_confidence" in cols
                else "squad_adjusted_confidence" if "squad_adjusted_confidence" in cols
                else "confidence"
            )
            df = pd.read_sql_query(
                f"SELECT team_a, team_b, predicted_winner, {conf_col} AS confidence, "
                "group_name, confidence_band "
                "FROM match_predictions "
                "WHERE tournament_year = 2026 "
                f"AND {conf_col} IS NOT NULL "
                f"ORDER BY {conf_col} DESC LIMIT 3",
                c,
            )
            return df.to_dict("records") if not df.empty else []
    except Exception:
        return []


def render_page_header() -> None:
    head_l, head_r = st.columns([5, 1], gap="small")
    with head_l:
        st.markdown(
            """
<div class="hero-eyebrow">Goal.ML · FIFA World Cup 2026</div>
<div class="hero-title">Predicting the World Cup<br><span class="accent">with Machine Learning</span></div>
<div class="hero-sub">A blended prediction engine combining historical ML, FIFA rankings, ELO,
and current squad statistics — alongside live Reddit sentiment across all 48 nations.</div>
""",
            unsafe_allow_html=True,
        )
    with head_r:
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        if st.button("↺ Refresh", use_container_width=True, help="Clear cached data and reload"):
            st.cache_data.clear()
            st.rerun()

    if not database_exists():
        st.error(
            "**Database not found.** Start the full pipeline with:  \n"
            "`docker compose up -d`"
        )

    top_preds = get_hero_predictions()
    if top_preds:
        st.markdown(
            '<div class="match-panel-label">Most Confident Picks</div>',
            unsafe_allow_html=True,
        )
        cards_html = '<div class="match-panel">'
        for p in top_preds:
            team_a = str(p.get("team_a", "TBD"))
            team_b = str(p.get("team_b", "TBD"))
            winner = str(p.get("predicted_winner", ""))
            conf = p.get("confidence")
            group = str(p.get("group_name", "Group Stage") or "Group Stage")
            conf_pct = int(float(conf) * 100) if conf is not None else 0
            home_cls = "home is-winner" if winner == team_a else "home"
            away_cls = "away is-winner" if winner == team_b else "away"
            cards_html += (
                f'<div class="match-card">'
                f'<div class="match-card-group">{group}</div>'
                f'<div class="match-card-teams">'
                f'<div class="match-team {home_cls}">{team_a}</div>'
                f'<div class="match-vs">VS</div>'
                f'<div class="match-team {away_cls}">{team_b}</div>'
                f'</div>'
                f'<div class="match-bar-track"><div class="match-bar-fill" style="width:{conf_pct}%"></div></div>'
                f'<div class="match-result">'
                f'<span class="match-winner-label">{winner}</span>'
                f'<span class="match-conf-label">{conf_pct}%</span>'
                f'</div>'
                f'</div>'
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)


def kpi_card_html(label: str, value: str, caption: str = "") -> str:
    cap_html = (
        f'<div class="kpi-caption" title="{caption}">{caption}</div>'
        if caption
        else ""
    )
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-value" title="{value}">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f"{cap_html}"
        f"</div>"
    )


def render_kpi_grid(kpis: dict) -> None:
    """Two rows of 4 KPI cards each."""
    dr_val, dr_cap = compute_date_range_label(
        kpis.get("reddit_min_date"), kpis.get("reddit_max_date")
    )

    # Row 1 — volume metrics
    r1 = st.columns(4, gap="small")
    row1_cards = [
        ("Reddit Posts",  fmt_int(kpis.get("reddit_posts")),      ""),
        ("Processed",     fmt_int(kpis.get("processed_posts")),   ""),
        ("Teams",         fmt_int(kpis.get("teams_with_sentiment")), "with sentiment data"),
        ("Predictions",   fmt_int(kpis.get("predictions_2026")),  "2026 group stage"),
    ]
    for col, (lbl, val, cap) in zip(r1, row1_cards):
        col.markdown(kpi_card_html(lbl, val, cap), unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # Row 2 — model & time metrics
    r2 = st.columns(4, gap="small")
    row2_cards = [
        ("Accuracy",   fmt(kpis.get("accuracy")),  "overall"),
        ("F1 Macro",   fmt(kpis.get("f1_macro")),  "macro average"),
        ("Best Model", kpis.get("best_model", "—"), kpis.get("best_model_full", "")),
        ("Date Range", dr_val,                      dr_cap),
    ]
    for col, (lbl, val, cap) in zip(r2, row2_cards):
        col.markdown(kpi_card_html(lbl, val, cap), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = f'<div class="sec-eyebrow">{eyebrow}</div>' if eyebrow else ""
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'{eyebrow_html}<div class="sec-title">{title}</div>{sub}',
        unsafe_allow_html=True,
    )


def empty_state(msg: str, hint: str = "", icon: str = "📭") -> None:
    hint_html = f'<div class="empty-hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="empty-box">'
        f'<div class="empty-icon">{icon}</div>'
        f'<div class="empty-msg">{msg}</div>'
        f"{hint_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_footer(kpis: dict) -> None:
    """Global status + data health, formerly the sidebar (filters now live in tabs)."""
    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    fc1, fc2 = st.columns([1, 1.3], gap="large")

    with fc1:
        st.markdown(
            "<div class='filter-bar-label'>Model status</div>",
            unsafe_allow_html=True,
        )
        if kpis.get("model_status") == "trained":
            st.success(
                f"**{kpis.get('best_model', '—')}**  \n"
                f"Acc {fmt(kpis.get('accuracy'))} · F1 {fmt(kpis.get('f1_macro'))}  \n"
                f"Trained {kpis.get('run_at', '—')}"
            )
        else:
            st.warning("Model not yet trained.")

    with fc2:
        st.markdown(
            "<div class='filter-bar-label'>Data sources</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "**Football CSV** — Kaggle historical match & ranking files  \n"
            "**Reddit RSS** — public feeds, no API key required  \n"
            "**Database** — local SQLite at `/app/data/worldcup.db`"
        )

    with st.expander("🩺 Data health", expanded=False):
        tbls = [
            "raw_reddit_posts", "processed_posts", "team_sentiment_daily",
            "trending_words", "match_predictions", "model_metrics",
        ]
        hcols = st.columns(3)
        for i, t in enumerate(tbls):
            n = scalar(f"SELECT COUNT(*) FROM {t}", None)
            if n is None:
                line = f"⚠️ `{t}` — missing"
            elif n == 0:
                line = f"🔴 `{t}` — **empty**"
            else:
                line = f"🟢 `{t}` — {n:,}"
            hcols[i % 3].markdown(line)
        st.caption(
            "Integration check:  \n"
            "`docker compose run --rm preprocessor python /scripts/integration_check.py`"
        )


# ---------------------------------------------------------------------------
# Tab helpers — common chart settings
# ---------------------------------------------------------------------------

CHART_H = 380
TRANSPARENT = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

# Categorical palette tuned for legibility on a white surface,
# led by the brand trio (Pitch Green, Championship Gold, Stadium Navy).
WC_COLORS = [
    "#2e7d32", "#d4af37", "#0d1b2a", "#1b998b", "#c0392b",
    "#3a6ea5", "#6a4c93", "#8e6c1f", "#2a9d8f", "#a14a3a",
    "#4a6fa5", "#7d5ba6",
]


def _apply_clean_layout(fig, **extra):
    fig.update_layout(
        **TRANSPARENT,
        font=dict(family="Inter, sans-serif", color="#40493d"),
        **extra,
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 1 — Predictions
# ---------------------------------------------------------------------------


_BAND_COLORS = {
    "High":   ("background:#e8f1e9", "color:#1e6023"),
    "Medium": ("background:#fbf3da", "color:#8a6d12"),
    "Low":    ("background:#fbe9e9", "color:#ba1a1a"),
}


def _num(v):
    """Coerce a scalar to float, returning None for NaN / non-numeric."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if pd.notna(f) else None


def _band_for(conf, explicit=None) -> str:
    """Resolve a confidence band, preferring an explicit stored value."""
    if explicit is not None and str(explicit) in ("High", "Medium", "Low"):
        return str(explicit)
    c = _num(conf)
    if c is None:
        return "Low"
    if c >= 0.70:
        return "High"
    if c >= 0.55:
        return "Medium"
    return "Low"


def _prediction_card_html(
    r: pd.Series, conf_col: str, has_band: bool, has_squad_diff: bool, has_key: bool
) -> str:
    """Build the HTML for a single fixture card."""
    team_a_raw = str(r.get("team_a", "TBD"))
    team_b_raw = str(r.get("team_b", "TBD"))
    team_a = html.escape(team_a_raw)
    team_b = html.escape(team_b_raw)
    winner = str(r.get("predicted_winner", "") or "")
    group = html.escape(str(r.get("group_name", "") or "Group"))

    conf = _num(r.get(conf_col))
    conf_pct = f"{conf * 100:.0f}%" if conf is not None else "—"
    bar_w = max(0, min(100, int(round(conf * 100)))) if conf is not None else 0
    band = _band_for(conf, r.get("confidence_band") if has_band else None)

    is_draw = winner.strip().lower() == "draw"
    a_win = (not is_draw) and winner == team_a_raw
    b_win = (not is_draw) and winner == team_b_raw
    a_cls = "pred-team-row winner" if a_win else "pred-team-row"
    b_cls = "pred-team-row winner" if b_win else "pred-team-row"
    a_tag = '<span class="pred-team-tag">✓ Pick</span>' if a_win else ""
    b_tag = '<span class="pred-team-tag">✓ Pick</span>' if b_win else ""
    mid = "DRAW" if is_draw else "VS"

    chips: list[str] = []
    if has_squad_diff:
        sd = _num(r.get("squad_strength_diff"))
        if sd is not None:
            chips.append(f'<span class="pred-chip">Squad {sd:+.1f}</span>')
    if has_key:
        kf = str(r.get("key_factors") or "").strip()
        if kf and kf.lower() != "nan":
            short = kf[:44] + ("…" if len(kf) > 44 else "")
            chips.append(
                f'<span class="pred-chip" title="{html.escape(kf)}">'
                f"{html.escape(short)}</span>"
            )
    chips_html = f'<div class="pred-meta">{"".join(chips)}</div>' if chips else ""

    return (
        f'<div class="pred-card band-{band}">'
        f'<div class="pred-top">'
        f'<span class="pred-group">{group}</span>'
        f'<span class="pred-pill band-{band}">{band}</span>'
        f"</div>"
        f'<div class="pred-teams">'
        f'<div class="{a_cls}"><span class="pred-team-name">{team_a}</span>{a_tag}</div>'
        f'<div class="pred-divider">{mid}</div>'
        f'<div class="{b_cls}"><span class="pred-team-name">{team_b}</span>{b_tag}</div>'
        f"</div>"
        f'<div class="pred-conf-row">'
        f'<span class="pred-conf-label">Final Confidence</span>'
        f'<span class="pred-conf-val">{conf_pct}</span>'
        f"</div>"
        f'<div class="pred-bar-track">'
        f'<div class="pred-bar-fill band-{band}" style="width:{bar_w}%"></div></div>'
        f"{chips_html}"
        f"</div>"
    )


def render_prediction_cards(df: pd.DataFrame, conf_col: str, per_row: int = 3) -> None:
    """Lay fixtures out as a card grid using native columns (reliable render)."""
    has_band = "confidence_band" in df.columns
    has_squad_diff = "squad_strength_diff" in df.columns
    has_key = "key_factors" in df.columns

    records = [r for _, r in df.iterrows()]
    for i in range(0, len(records), per_row):
        chunk = records[i : i + per_row]
        cols = st.columns(per_row, gap="medium")
        for col, r in zip(cols, chunk):
            with col:
                st.markdown(
                    _prediction_card_html(
                        r, conf_col, has_band, has_squad_diff, has_key
                    ),
                    unsafe_allow_html=True,
                )


def tab_predictions() -> None:
    section_header(
        "2026 Group Stage Predictions",
        "Final blended predictions for all 72 group stage matches.",
        eyebrow="ML · Ranking · ELO · Squad",
    )

    df = load_match_predictions(match_predictions_cache_key())
    if df.empty:
        empty_state(
            "No predictions available yet.",
            "Run the trainer:  docker compose run --rm trainer",
            "🤖",
        )
        return

    df26 = (
        df[df["tournament_year"] == 2026].copy()
        if "tournament_year" in df.columns
        else df.copy()
    )

    # ── Inline filters (formerly the sidebar) ──────────────────────────────
    if "group_name" in df26.columns:
        all_groups = sorted(df26["group_name"].dropna().unique().tolist())
        if all_groups:
            fc1, fc2 = st.columns([3, 1], gap="medium")
            sel_groups = fc1.multiselect(
                "Filter by group", all_groups, default=all_groups,
                help="Show only matches from the selected groups",
            )
            band_choice = fc2.selectbox(
                "Confidence", ["All", "High", "Medium", "Low"],
                help="Show only matches in a confidence band",
            )
            if sel_groups:
                df26 = df26[df26["group_name"].isin(sel_groups)]
            if band_choice != "All" and "confidence_band" in df26.columns:
                df26 = df26[df26["confidence_band"] == band_choice]

    if df26.empty:
        empty_state("No predictions match the selected filters.")
        return

    has_band         = "confidence_band"            in df26.columns
    has_expl         = "explanation"                 in df26.columns
    has_squad        = "squad_strength_a"            in df26.columns
    has_adj_conf     = (
        "adjusted_confidence" in df26.columns
        or "squad_adjusted_confidence" in df26.columns
    )
    has_raw_conf     = "raw_model_confidence"        in df26.columns
    has_adj_applied  = "squad_adjustment_applied"    in df26.columns
    has_squad_diff   = "squad_strength_diff"         in df26.columns
    has_cov_a        = "squad_coverage_a"            in df26.columns
    has_cov_b        = "squad_coverage_b"            in df26.columns
    has_tier_a       = "squad_coverage_tier_a"       in df26.columns
    has_tier_b       = "squad_coverage_tier_b"       in df26.columns
    has_key_factors  = "key_factors"                 in df26.columns
    has_combined_sig = "combined_strength_signal"    in df26.columns
    squad_rows_in_db = count_2026_predictions_with_squad()
    has_expl_col     = has_expl

    conf_for_avg = (
        "adjusted_confidence"
        if "adjusted_confidence" in df26.columns
        else "squad_adjusted_confidence"
        if "squad_adjusted_confidence" in df26.columns
        else "confidence"
    )
    adj_conf_col = (
        "adjusted_confidence"
        if "adjusted_confidence" in df26.columns
        else "squad_adjusted_confidence"
    )
    hist_ml_col = (
        "historical_ml_confidence"
        if "historical_ml_confidence" in df26.columns
        else "raw_model_confidence"
        if has_raw_conf
        else None
    )

    # ── Summary cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("Matches", len(df26))
    if conf_for_avg in df26.columns:
        avg_conf = pd.to_numeric(df26[conf_for_avg], errors="coerce").mean()
        c2.metric("Avg Final Confidence", f"{avg_conf:.1%}")
        if has_band:
            high_count = (df26["confidence_band"] == "High").sum()
            c3.metric("High-confidence predictions", f"{high_count}/{len(df26)}")
        else:
            c3.metric("High-confidence (≥70%)",
                      f"{(df26['confidence'] >= 0.70).sum()}/{len(df26)}")
        if has_adj_applied:
            adj_ct = int(
                pd.to_numeric(df26["squad_adjustment_applied"], errors="coerce")
                .fillna(0)
                .sum()
            )
            c4.metric(
                "Blended predictions",
                f"{adj_ct}/{len(df26)}",
                help="Final predictions using ranking + ML + ELO + squad blend",
            )
        else:
            c4.metric("Prediction system", "Blended model + squad")

    # ── Methodology (collapsed to keep the cards front and centre) ─────────
    with st.expander("How Final Confidence is calculated", expanded=False):
        st.markdown(
            "**Final Confidence** combines historical ML (30%), FIFA ranking/points "
            "(40%), ELO/rank agreement (10%), and current squad/player statistics "
            "(20%), with caps to avoid overstatement. The standalone historical ML "
            "signal is only one component — see **Model Performance** for that "
            "diagnostic. Player stats cover ~900 of ~1200 expected World Cup players; "
            "missing stats are **missing data**, not weak players."
        )

    # ── Fixture cards (primary view) ───────────────────────────────────────
    card_conf_col = conf_for_avg if conf_for_avg in df26.columns else "confidence"
    if card_conf_col in df26.columns:
        df_cards = (
            df26.assign(_c=pd.to_numeric(df26[card_conf_col], errors="coerce"))
            .sort_values("_c", ascending=False)
            .drop(columns="_c")
        )
    else:
        df_cards = df26
    st.markdown(
        "<div class='match-panel-label'>Match Predictions</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Showing {len(df_cards)} matches · sorted by final confidence")
    render_prediction_cards(df_cards, card_conf_col)

    # ── Squad coverage notice (only when something needs attention) ────────
    if squad_rows_in_db == 0:
        st.warning(
            "Squad columns are empty in the database. The trainer likely ran before "
            "player stats were loaded. Run: `docker compose up --build` (collector then "
            "trainer), or `docker compose restart trainer` after collector finishes."
        )
    elif squad_rows_in_db < len(df26):
        st.info(
            f"Squad strength is populated for {squad_rows_in_db}/{len(df26)} "
            "2026 predictions. Re-run trainer after collector loads player stats "
            "to refresh all rows.",
            icon="ℹ️",
        )

    # ── Confidence distribution bar chart ─────────────────────────────────
    if has_band and "predicted_winner" in df26.columns:
        band_order = ["High", "Medium", "Low"]
        band_counts = (
            df26.groupby("confidence_band")
            .size()
            .reindex(band_order)
            .fillna(0)
            .reset_index()
        )
        band_counts.columns = ["Band", "Count"]
        col_l, col_r = st.columns([1, 2], gap="small")
        with col_l:
            st.markdown("**Confidence distribution**")
            fig_band = px.bar(
                band_counts, x="Band", y="Count", color="Band",
                color_discrete_map={"High": "#2e7d32", "Medium": "#d4af37", "Low": "#ba1a1a"},
                height=220,
            )
            _apply_clean_layout(fig_band, showlegend=False)
            st.plotly_chart(fig_band, use_container_width=True)
        with col_r:
            if {"team_a", "team_b", "confidence", "predicted_winner"}.issubset(df26.columns):
                st.markdown("**Model confidence by match (top 36)**")
                df_plot = df26.copy()
                df_plot["Match"] = df_plot["team_a"] + " vs " + df_plot["team_b"]
                fig = px.bar(
                    df_plot.sort_values("confidence", ascending=False).head(36),
                    x="Match", y="confidence",
                    color="confidence_band" if has_band else "predicted_winner",
                    color_discrete_map={"High": "#2e7d32", "Medium": "#d4af37", "Low": "#ba1a1a"}
                    if has_band else None,
                    labels={"confidence": "Model Confidence",
                            "confidence_band": "Band",
                            "predicted_winner": "Winner"},
                    height=CHART_H,
                )
                fig.add_hline(y=0.70, line_dash="dot", line_color="#2e7d32",
                              opacity=0.7, annotation_text="70% (High)")
                fig.add_hline(y=0.55, line_dash="dot", line_color="#b8941f",
                              opacity=0.7, annotation_text="55% (Medium)")
                _apply_clean_layout(fig, xaxis_tickangle=-50, margin=dict(b=110))
                st.plotly_chart(fig, use_container_width=True)
    elif {"team_a", "team_b", "confidence", "predicted_winner"}.issubset(df26.columns):
        st.markdown("**Model confidence by match**")
        df_plot = df26.copy()
        df_plot["Match"] = df_plot["team_a"] + " vs " + df_plot["team_b"]
        fig = px.bar(
            df_plot.sort_values("confidence", ascending=False).head(36),
            x="Match", y="confidence", color="predicted_winner",
            labels={"confidence": "Model Confidence", "predicted_winner": "Winner"},
            color_discrete_sequence=WC_COLORS,
            height=CHART_H,
        )
        fig.add_hline(y=HIGH_CONF_THRESHOLD, line_dash="dash",
                      line_color="#2e7d32", opacity=0.7,
                      annotation_text="55% threshold")
        _apply_clean_layout(fig, xaxis_tickangle=-50, margin=dict(b=110),
                            legend_title_text="Winner")
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-match explanations expander ───────────────────────────────────
    if has_expl and "explanation" in df26.columns:
        with st.expander("Match explanations (key factors per prediction)", expanded=False):
            expl_df = df26[
                ["group_name", "team_a", "team_b", "predicted_winner",
                 "confidence", "confidence_band", "explanation", "key_factors"]
            ].dropna(subset=["explanation"])
            expl_df = expl_df.rename(columns={
                "group_name": "Group",
                "team_a": "Team A", "team_b": "Team B",
                "predicted_winner": "Winner",
                "confidence": "Confidence",
                "confidence_band": "Band",
                "explanation": "Summary",
                "key_factors": "Key Factors",
            })
            expl_df["Confidence"] = expl_df["Confidence"].apply(
                lambda v: f"{v:.1%}" if isinstance(v, float) else v
            )
            st.dataframe(expl_df.reset_index(drop=True), use_container_width=True, height=380)

    # ═══════════════════════════════════════════════════════════════════════
    # Details & diagnostics (collapsed, below the cards)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='filter-bar-label'>Details &amp; diagnostics</div>",
        unsafe_allow_html=True,
    )

    # Build the wide table view
    col_map = {
        "group_name":       "Group",
        "team_a":           "Team A",
        "team_b":           "Team B",
        "predicted_winner": "Winner",
        adj_conf_col:       "Final Confidence",
        "confidence_band":  "Confidence Band",
        "squad_strength_a": "Squad A",
        "squad_strength_b": "Squad B",
        "squad_strength_diff": "Squad Diff",
        "squad_coverage_a": "Coverage A",
        "squad_coverage_b": "Coverage B",
        "squad_coverage_tier_a": "Tier A",
        "squad_coverage_tier_b": "Tier B",
        "key_factors":      "Key Factors",
        "explanation":      "Explanation",
    }
    display_cols = [c for c in col_map if c in df26.columns]
    df_show = df26[display_cols].rename(columns=col_map).copy()

    for col, num_fmt in (
        ("Squad A", "{:.1f}"),
        ("Squad B", "{:.1f}"),
        ("Squad Diff", "{:+.1f}"),
    ):
        if col in df_show.columns:
            nums = pd.to_numeric(df_show[col], errors="coerce")
            df_show[col] = nums.apply(
                lambda v: num_fmt.format(v) if pd.notna(v) else "—"
            )
    for cov_col, tier_col in (
        ("Coverage A", "Tier A"),
        ("Coverage B", "Tier B"),
    ):
        if cov_col not in df_show.columns:
            continue
        cov_nums = pd.to_numeric(df_show[cov_col], errors="coerce")
        tier_vals = (
            df_show[tier_col]
            if tier_col in df_show.columns
            else pd.Series([None] * len(df_show), index=df_show.index)
        )
        formatted = []
        for v, tier in zip(cov_nums, tier_vals):
            if pd.isna(v):
                formatted.append("—")
                continue
            tier_s = (
                str(tier).strip()
                if tier is not None and str(tier) not in ("", "nan", "None")
                else ""
            )
            formatted.append(f"{v:.0%} ({tier_s})" if tier_s else f"{v:.0%}")
        df_show[cov_col] = formatted
    df_show.drop(columns=["Tier A", "Tier B"], inplace=True, errors="ignore")

    if "Final Confidence" in df_show.columns:
        sort_vals = pd.to_numeric(df26[adj_conf_col], errors="coerce")
        df_show = df_show.assign(_sort=sort_vals.values).sort_values(
            "_sort", ascending=False
        ).drop(columns="_sort")

    def _cell_confidence(v):
        if not isinstance(v, (float, int)):
            return ""
        if v >= 0.70:
            return "background:#e8f1e9;color:#1e6023"
        if v >= 0.55:
            return "background:#fbf3da;color:#8a6d12"
        return "background:#fbe9e9;color:#ba1a1a"

    def _cell_band(v):
        if v == "High":
            return "color:#1e6023;font-weight:700"
        if v == "Medium":
            return "color:#8a6d12;font-weight:700"
        if v == "Low":
            return "color:#ba1a1a;font-weight:700"
        return ""

    with st.expander("Full data table", expanded=False):
        if "Final Confidence" in df_show.columns or "Confidence Band" in df_show.columns:
            conf_cols = [c for c in ("Final Confidence",) if c in df_show.columns]
            styled = df_show.style
            for cc in conf_cols:
                styled = styled.map(_cell_confidence, subset=[cc])
            if "Confidence Band" in df_show.columns:
                styled = styled.map(_cell_band, subset=["Confidence Band"])
            styled = styled.format({cc: "{:.1%}" for cc in conf_cols}, na_rep="—")
            st.dataframe(styled, use_container_width=True, height=420)
        else:
            st.dataframe(df_show, use_container_width=True, height=420)

    # Squad data health
    if has_squad or has_adj_applied:
        ps_count  = load_table("raw_player_stats")
        sq_count  = load_table("raw_national_squads")
        sqs_count = load_table("team_squad_strength")
        with st.expander("Squad data health", expanded=False):
            sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
            sc1.metric("Player stats loaded", len(ps_count) if not ps_count.empty else 0)
            sc2.metric("Teams with squad strength", len(sqs_count) if not sqs_count.empty else 0)
            if not sqs_count.empty and "coverage_tier" in sqs_count.columns:
                tier_dist = sqs_count["coverage_tier"].value_counts().to_dict()
                sc3.metric("Coverage tiers", str(tier_dist))
            else:
                sc3.metric("Squad rosters (optional)", len(sq_count) if not sq_count.empty else 0)
            adj_ct = (
                int(
                    pd.to_numeric(df26["squad_adjustment_applied"], errors="coerce")
                    .fillna(0)
                    .sum()
                )
                if has_adj_applied
                else 0
            )
            sc4.metric("Predictions adjusted", adj_ct)
            boost_col = (
                "confidence_boost_amount"
                if "confidence_boost_amount" in df26.columns
                else "squad_adjustment_amount"
            )
            if has_adj_applied and boost_col in df26.columns:
                adj_vals = pd.to_numeric(df26[boost_col], errors="coerce")
                avg_adj = adj_vals.abs().mean()
                if pd.notna(avg_adj):
                    st.caption(f"Average confidence boost: {avg_adj:.3f}")
            if has_combined_sig:
                sig_vals = pd.to_numeric(df26["combined_strength_signal"], errors="coerce")
                if sig_vals.notna().any():
                    st.caption(
                        f"Combined strength signal range: "
                        f"{sig_vals.min():+.2f} to {sig_vals.max():+.2f}"
                    )
            st.caption(
                "Missing players are uncertainty, not zero skill. Low-coverage teams "
                "blend FIFA fallback into squad scores; adjustment caps are tighter."
            )

    # Historical ML signal (debug)
    with st.expander("Historical ML signal (debug component)", expanded=False):
        st.caption(
            "The standalone historical ML classifier is only one input to the final "
            "prediction. It is shown here for transparency, not as the main model."
        )
        debug_map = {}
        if hist_ml_col and hist_ml_col in df26.columns:
            debug_map[hist_ml_col] = "Historical ML Signal"
        if "ranking_baseline_confidence" in df26.columns:
            debug_map["ranking_baseline_confidence"] = "Ranking Component"
        if debug_map:
            dbg = df26[list(debug_map.keys())].rename(columns=debug_map).copy()
            for dc in debug_map.values():
                if dc in dbg.columns:
                    nums = pd.to_numeric(dbg[dc], errors="coerce")
                    dbg[dc] = nums.apply(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
            st.dataframe(
                pd.concat(
                    [df26[["team_a", "team_b"]].reset_index(drop=True), dbg],
                    axis=1,
                ),
                use_container_width=True,
                height=280,
            )

    # Column glossary
    with st.expander("What do these columns mean?", expanded=False):
        st.markdown(
            """
| Column | Meaning |
|--------|---------|
| **Winner** | Final blended prediction (ranking + ML + ELO + squad/player stats) |
| **Final Confidence** | Probability for the predicted outcome after the full blend (capped at 88%) |
| **Confidence Band** | High / Medium / Low bucket from final confidence |
| **Squad A / B** | Current squad strength score (0–100) from club-season player stats |
| **Squad Diff** | Squad A minus Squad B (positive favors Team A) |
| **Coverage A / B** | Share of expected 26-man roster with stats in our database (0–100%) |
| **Key Factors** | Short summary: FIFA rank, ELO, squad, coverage warnings |
| **Explanation** | Full readable paragraph for this match |

**—** means squad data was not stored yet (re-run collector + trainer), not that the team has no players.
            """
        )


# ---------------------------------------------------------------------------
# Tab 2 — Sentiment
# ---------------------------------------------------------------------------

METRIC_DESC = {
    "avg_vader":    "VADER compound score — ranges −1 (negative) to +1 (positive).",
    "avg_textblob": "TextBlob polarity — ranges −1 to +1, secondary sentiment signal.",
    "hype_index":   "Hype index = (post volume / daily max) × max(0, avg VADER). Combines buzz and positivity.",
}


def tab_sentiment() -> None:
    section_header(
        "Reddit Sentiment Tracker",
        "Daily sentiment scores per team, derived from Reddit post titles and text.",
        eyebrow="VADER · TextBlob · Hype Index",
    )

    df = load_table("team_sentiment_daily")
    if df.empty:
        empty_state(
            "No sentiment data yet.",
            "Run the preprocessor:  docker compose run --rm preprocessor",
            "📉",
        )
        return

    required = {"team", "date", "avg_vader"}
    if not required.issubset(df.columns):
        st.warning(f"Expected columns {required}. Found: {list(df.columns)}")
        st.dataframe(df.head(20), use_container_width=True)
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["avg_vader", "avg_textblob", "hype_index", "post_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "avg_vader", "team"])

    # ── Inline filters (formerly the sidebar) ──────────────────────────────
    all_teams = sorted(df["team"].dropna().unique().tolist())
    default_teams = all_teams[:6] if len(all_teams) >= 6 else all_teams
    fc1, fc2 = st.columns([3, 1], gap="medium")
    sel_teams = fc1.multiselect("Teams", all_teams, default=default_teams)
    metric = fc2.selectbox(
        "Metric",
        ["avg_vader", "avg_textblob", "hype_index"],
        help=(
            "avg_vader: VADER −1→+1  |  "
            "avg_textblob: TextBlob −1→+1  |  "
            "hype_index: volume × positivity"
        ),
    )
    if metric not in df.columns:
        metric = "avg_vader"

    df_f = df[df["team"].isin(sel_teams)].copy() if sel_teams else df.copy()

    # Metric description
    if metric in METRIC_DESC:
        st.caption(METRIC_DESC[metric])

    if df_f.empty:
        empty_state("No data for the selected teams.")
        return

    metric_label = metric.replace("_", " ").title()
    unique_dates = df_f["date"].nunique()

    if unique_dates <= 1:
        st.info("Only one date available — showing bar chart instead of time series.")
        agg = df_f.groupby("team")[metric].mean().reset_index()
        fig = px.bar(
            agg, x="team", y=metric, color="team",
            labels={metric: metric_label, "team": "Team"},
            color_discrete_sequence=WC_COLORS,
            height=CHART_H,
        )
        _apply_clean_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(
            df_f.sort_values("date"), x="date", y=metric, color="team",
            markers=True,
            labels={metric: metric_label, "date": "Date"},
            color_discrete_sequence=WC_COLORS,
            height=CHART_H,
        )
        fig.update_traces(line_width=2, marker_size=6)
        fig.add_hline(y=0, line_dash="dot", line_color="#bfcaba", opacity=0.9)
        _apply_clean_layout(fig, hovermode="x unified", legend_title_text="Team")
        st.plotly_chart(fig, use_container_width=True)

    # Hype bar (only if not already showing it)
    if "hype_index" in df_f.columns and metric != "hype_index":
        dh = df_f.groupby("team", as_index=False)["hype_index"].mean().dropna()
        if not dh.empty and dh["hype_index"].sum() > 0:
            st.markdown("**Average Hype Index by team**")
            fig_h = px.bar(
                dh.sort_values("hype_index", ascending=False),
                x="team", y="hype_index", color="team",
                labels={"hype_index": "Hype Index", "team": "Team"},
                color_discrete_sequence=WC_COLORS,
                height=280,
            )
            _apply_clean_layout(fig_h, showlegend=False)
            st.plotly_chart(fig_h, use_container_width=True)

    # Recent data table
    with st.expander("Recent sentiment rows", expanded=False):
        show_cols = [c for c in ["team", "date", "avg_vader", "avg_textblob",
                                  "hype_index", "post_count"] if c in df_f.columns]
        st.dataframe(
            df_f[show_cols].sort_values("date", ascending=False).head(50)
            .reset_index(drop=True),
            use_container_width=True, height=300,
        )


# ---------------------------------------------------------------------------
# Tab 3 — Trending Topics
# ---------------------------------------------------------------------------

_CATEGORY_LABELS: dict[str, str] = {
    "player_mention":      "👤 Player Mentions",
    "sentiment_positive":  "✅ Positive Sentiment",
    "sentiment_negative":  "⚠️ Negative Sentiment",
    "injury_concern":      "🤕 Injury / Fitness",
    "squad_selection":     "📋 Squad / Lineup",
    "tactical_performance":"⚽ Tactics / Performance",
    "competition_context": "🏆 Competition Context",
    "other":               "🔤 Other Terms",
}

_ALL_CATEGORIES = ["all"] + list(_CATEGORY_LABELS.keys())


def tab_trending() -> None:
    section_header(
        "Trending Topics",
        "Player mentions, sentiment words, injuries, squad news and tactical discussion "
        "extracted from Reddit and Telegram posts. "
        "Generic boilerplate (football, world, cup, comments, link…) is filtered out.",
        eyebrow="NLP · Reddit RSS",
    )

    df = load_table("trending_words")
    if df.empty:
        empty_state(
            "No trending word data yet.",
            "Run the preprocessor:  docker compose run --rm preprocessor",
            "💬",
        )
        return

    required = {"team", "word", "frequency"}
    if not required.issubset(df.columns):
        st.warning(f"Expected columns {required}. Found: {list(df.columns)}")
        st.dataframe(df.head(20), use_container_width=True)
        return

    has_category   = "category"   in df.columns
    has_ngram_type = "ngram_type" in df.columns

    # ── Controls row ──────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 2, 2, 1], gap="small")

    teams = sorted(df["team"].dropna().unique().tolist())
    if not teams:
        empty_state("No team-tagged word data available.")
        return

    with ctrl1:
        sel_team = st.selectbox("Team", teams, key="tw_team")

    with ctrl2:
        if has_category:
            cat_options = _ALL_CATEGORIES
            cat_labels  = ["All categories"] + [_CATEGORY_LABELS.get(c, c) for c in cat_options[1:]]
            cat_idx     = st.selectbox(
                "Category",
                range(len(cat_options)),
                format_func=lambda i: cat_labels[i],
                key="tw_cat",
            )
            sel_category = cat_options[cat_idx]
        else:
            sel_category = "all"

    with ctrl3:
        if has_ngram_type:
            ngram_opts = ["all", "unigram", "bigram"]
            ngram_labels = ["All (words + phrases)", "Single words", "2-word phrases"]
            ng_idx = st.selectbox(
                "Type",
                range(len(ngram_opts)),
                format_func=lambda i: ngram_labels[i],
                key="tw_ng",
            )
            sel_ngram = ngram_opts[ng_idx]
        else:
            sel_ngram = "all"

    with ctrl4:
        top_n = st.slider("Top N", 5, 40, 20, key="tw_n")

    # ── Filter data ───────────────────────────────────────────────────────
    df_t = df[df["team"] == sel_team].copy()

    # Optional date filter
    if "date" in df_t.columns:
        dates = sorted(df_t["date"].dropna().unique().tolist())
        if len(dates) > 1:
            pick = st.multiselect(
                "Filter by date (optional)", dates, key="tw_dates",
                help="Leave blank to aggregate all available dates",
            )
            if pick:
                df_t = df_t[df_t["date"].isin(pick)]

    # Apply category filter
    if has_category and sel_category != "all":
        df_t = df_t[df_t["category"] == sel_category]

    # Apply ngram_type filter
    if has_ngram_type and sel_ngram != "all":
        df_t = df_t[df_t["ngram_type"] == sel_ngram]

    # Aggregate across dates
    group_cols = ["word"]
    if has_category:
        group_cols.append("category")
    if has_ngram_type:
        group_cols.append("ngram_type")

    agg_top = (
        df_t.groupby(group_cols, as_index=False)["frequency"]
        .sum()
        .nlargest(top_n, "frequency")
    )

    if agg_top.empty:
        empty_state(
            "No meaningful trending terms found for this filter.",
            "Try a different team, category, or date range — "
            "or collect more posts to build richer data.",
            "🔕",
        )
        return

    # ── Chart + table ─────────────────────────────────────────────────────
    # Colour by category when available, else a green frequency ramp
    if has_category and "category" in agg_top.columns:
        color_col   = "category"
        color_scale = None
        color_map   = {
            "player_mention":       "#0d1b2a",
            "sentiment_positive":   "#2e7d32",
            "sentiment_negative":   "#ba1a1a",
            "injury_concern":       "#d4af37",
            "squad_selection":      "#6a4c93",
            "tactical_performance": "#3a6ea5",
            "competition_context":  "#b8941f",
            "other":                "#707a6c",
        }
    else:
        color_col   = "frequency"
        color_scale = "Greens"
        color_map   = None

    chart_h = max(340, top_n * 22)

    left, right = st.columns([3, 2], gap="small")
    with left:
        if color_scale:
            fig = px.bar(
                agg_top.sort_values("frequency"),
                x="frequency", y="word", orientation="h",
                labels={"frequency": "Frequency", "word": "Word"},
                color="frequency", color_continuous_scale=color_scale,
                height=chart_h,
            )
            _apply_clean_layout(fig, coloraxis_showscale=False)
        else:
            fig = px.bar(
                agg_top.sort_values("frequency"),
                x="frequency", y="word", orientation="h",
                color="category",
                color_discrete_map=color_map,
                labels={"frequency": "Frequency", "word": "Word", "category": "Category"},
                height=chart_h,
            )
            fig.update_layout(
                legend=dict(
                    title="Category",
                    orientation="v",
                    x=1.01, y=1,
                    font=dict(size=10),
                )
            )
            _apply_clean_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        n_unique = df[df["team"] == sel_team]["word"].nunique()
        st.caption(
            f"Showing top {min(top_n, len(agg_top))} of "
            f"{n_unique} unique terms for **{sel_team}**"
        )
        # Rename columns for user-friendly display
        rename = {"word": "Word/Phrase", "frequency": "Count",
                  "category": "Category", "ngram_type": "Type"}
        display_cols = [c for c in ["word", "frequency", "category", "ngram_type"]
                        if c in agg_top.columns]
        st.dataframe(
            agg_top[display_cols]
            .sort_values("frequency", ascending=False)
            .rename(columns=rename)
            .reset_index(drop=True),
            use_container_width=True,
            height=chart_h,
        )

    # ── Category breakdown (only when showing "all") ──────────────────────
    if has_category and sel_category == "all" and "category" in df_t.columns:
        cat_agg = (
            df_t.groupby("category")["frequency"]
            .sum()
            .reset_index()
            .sort_values("frequency", ascending=False)
        )
        if not cat_agg.empty:
            with st.expander("Category breakdown", expanded=False):
                cat_agg["Category"] = cat_agg["category"].map(
                    lambda c: _CATEGORY_LABELS.get(c, c)
                )
                fig_cat = px.bar(
                    cat_agg,
                    x="Category", y="frequency",
                    labels={"frequency": "Total frequency"},
                    color="category",
                    color_discrete_map=color_map,
                    height=280,
                )
                _apply_clean_layout(fig_cat, showlegend=False)
                st.plotly_chart(fig_cat, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Historical Stats
# ---------------------------------------------------------------------------


def tab_historical() -> None:
    section_header(
        "Historical World Cup Data",
        "Match results from FIFA World Cup tournaments (1930–2022).",
        eyebrow="1930 – 2022 · 983 matches",
    )

    df = load_table("raw_matches")
    if df.empty:
        empty_state(
            "No historical data loaded.",
            "Place matches CSV in datasets/ and run the collector.",
            "📋",
        )
        return

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in ["score_a", "score_b"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Inline filters (formerly the sidebar) ──────────────────────────────
    fc1, fc2 = st.columns([2, 2], gap="medium")
    year_range = None
    if "year" in df.columns and not df["year"].isna().all():
        years = sorted(df["year"].dropna().astype(int).unique().tolist())
        if len(years) >= 2:
            year_range = fc1.slider(
                "Year range", years[0], years[-1], (years[0], years[-1])
            )
        elif years:
            fc1.caption(f"Only one year available: {years[0]}")
            year_range = (years[0], years[0])

    team_pool = sorted(
        set(df.get("team_a", pd.Series(dtype=str)).dropna().tolist())
        | set(df.get("team_b", pd.Series(dtype=str)).dropna().tolist())
    )
    ht = fc2.selectbox("Team", ["All teams"] + team_pool)

    if year_range and "year" in df.columns:
        df = df[df["year"].between(year_range[0], year_range[1])]
    if ht != "All teams" and "team_a" in df.columns and "team_b" in df.columns:
        df = df[(df["team_a"] == ht) | (df["team_b"] == ht)]

    # Summary row
    all_t = sorted(
        set(df.get("team_a", pd.Series()).dropna().tolist())
        | set(df.get("team_b", pd.Series()).dropna().tolist())
    )
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("Matches", len(df))
    if "year" in df.columns and not df["year"].isna().all():
        c2.metric("Year range",
                  f"{int(df['year'].min())} – {int(df['year'].max())}")
    c3.metric("Teams", len(all_t))
    if "stage" in df.columns:
        c4.metric("Stages", df["stage"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns(2, gap="small")
    with ch1:
        if "year" in df.columns:
            mpy = df.groupby("year").size().reset_index(name="Matches")
            fig = px.bar(mpy, x="year", y="Matches",
                         labels={"year": "Year"},
                         title="Matches per Tournament",
                         color="Matches", color_continuous_scale="Greens",
                         height=300)
            _apply_clean_layout(fig, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with ch2:
        if ht != "All teams" and "winner" in df.columns and not df.empty:
            wins   = (df["winner"] == ht).sum()
            draws  = (df["winner"] == "Draw").sum()
            losses = max(0, len(df) - wins - draws)
            fig_p = px.pie(
                values=[max(0, wins), max(0, draws), losses],
                names=["Wins", "Draws", "Losses"],
                title=f"{ht} — Record",
                color_discrete_sequence=["#2e7d32", "#d4af37", "#ba1a1a"],
                hole=0.42,
                height=300,
            )
            _apply_clean_layout(fig_p)
            st.plotly_chart(fig_p, use_container_width=True)
        elif "score_a" in df.columns and "score_b" in df.columns:
            avg = (df["score_a"].fillna(0) + df["score_b"].fillna(0)).mean()
            st.metric("Avg goals / match", f"{avg:.2f}")
            st.metric("Total goals", fmt_int(
                int(df["score_a"].fillna(0).sum() + df["score_b"].fillna(0).sum())
            ))

    # Match table — polished subset
    result_cols = [c for c in ["year", "stage", "team_a", "score_a", "score_b", "team_b", "winner"]
                   if c in df.columns]
    rename_hist = {
        "year": "Year", "stage": "Stage",
        "team_a": "Team A", "score_a": "A",
        "score_b": "B", "team_b": "Team B", "winner": "Winner",
    }
    with st.expander("Match results table", expanded=True):
        st.dataframe(
            df[result_cols].rename(columns=rename_hist)
            .sort_values("Year", ascending=False).head(200)
            .reset_index(drop=True),
            use_container_width=True,
            height=360,
        )


# ---------------------------------------------------------------------------
# Tab 5 — Model Performance
# ---------------------------------------------------------------------------


def tab_model() -> None:
    section_header(
        "Historical ML Component",
        "Diagnostic metrics for the standalone historical classifier only. "
        "Final 2026 predictions blend ranking, ELO, ML, and squad/player stats.",
        eyebrow="XGBoost · RandomForest · 2022 test set",
    )

    st.info(
        "This tab evaluates **only the historical ML component** trained on past World Cup "
        "matches. The **final 2026 prediction system** also blends FIFA ranking, ELO, and "
        "current squad/player statistics. See the Predictions tab for final confidence.",
        icon="ℹ️",
    )

    m = load_metrics()
    df_hist = load_table("model_metrics")

    if not m and df_hist.empty:
        empty_state(
            "No metrics yet.",
            "Run the trainer:  docker compose run --rm trainer",
            "📊",
        )
        return

    if m.get("status") == "trained":
        c1, c2, c3, c4, c5 = st.columns(5, gap="small")
        c1.metric("ML component", shorten_model_name(m.get("best_model")))
        c2.metric("ML accuracy (2022 test)", fmt(m.get("accuracy")))
        c3.metric("ML F1 macro", fmt(m.get("f1_macro")))
        c4.metric("Train rows",  m.get("train_rows", "—"))
        c5.metric("Test rows",   m.get("test_rows", "—"))

        if m.get("run_at"):
            st.caption(f"Last trained: {str(m['run_at'])[:19]}")

        baseline = m.get("baseline") or {}
        base_acc = baseline.get("accuracy")
        base_f1 = baseline.get("f1_macro")
        if base_acc is not None and base_f1 is not None:
            beats = m.get("ml_beats_baseline_f1", False)
            st.caption(
                f"Ranking baseline (same test set): accuracy {float(base_acc):.1%}, "
                f"F1 macro {float(base_f1):.3f}"
                + (
                    " — ranking baseline outperforms standalone ML"
                    if not beats
                    else " — ML beats ranking baseline on F1"
                )
            )

        if m.get("single_class_collapse"):
            st.warning(
                "The ML model predicts only one outcome class on the 2022 test set. "
                "Retrain after symmetric augmentation fixes (see trainer logs)."
            )
        elif m.get("zero_draw_predictions"):
            st.warning(
                "The historical ML component predicts **zero draws** on the 2022 test set. "
                "A draw decision rule is applied for 2026 final predictions; see trainer logs."
            )
        elif m.get("predicted_class_distribution"):
            st.caption(
                "Test predicted classes (draw-rule evaluation): "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in m["predicted_class_distribution"].items()
                )
            )

        draw_rule = m.get("draw_decision_rule") or {}
        if draw_rule.get("enabled"):
            st.caption(
                f"Draw decision rule: predict Draw when draw proba ≥ "
                f"{float(draw_rule.get('draw_threshold', 0.3)):.2f} and "
                f"|P(A)−P(B)| ≤ {float(draw_rule.get('closeness_threshold', 0.1)):.2f}"
            )
        if m.get("draw_recall") is not None:
            st.caption(
                f"Draw recall (2022 test): {float(m['draw_recall']):.3f}"
                + (
                    f" | Balanced accuracy: {float(m['balanced_accuracy']):.3f}"
                    if m.get("balanced_accuracy") is not None
                    else ""
                )
            )
        argmax_cmp = m.get("argmax_comparison") or {}
        if argmax_cmp:
            st.caption(
                "Argmax-only comparison — "
                f"accuracy {float(argmax_cmp.get('accuracy', 0)):.1%}, "
                f"F1 macro {float(argmax_cmp.get('f1_macro', 0)):.3f}, "
                f"draw recall {float(argmax_cmp.get('draw_recall', 0)):.3f}"
            )

        if m.get("proba_mapping"):
            st.caption(f"Probability order: {m['proba_mapping']}")

        acc = m.get("accuracy")
        if acc is not None and float(acc) < 0.55:
            st.info(
                "Accuracy below 55% is expected for this small dataset. "
                "Random baseline (3 classes) = 33 %."
            )
    elif m:
        st.warning(f"Model not trained yet. Reason: {m.get('reason', 'unknown')}")

    # Confusion matrix
    bm = m.get("best_model")
    if bm:
        cm = m.get("models", {}).get(bm, {}).get("confusion_matrix")
        if cm:
            st.markdown("**Confusion matrix**")
            st.caption(
                "This confusion matrix evaluates **only the historical ML component** on the "
                "2022 test set (with draw decision rule). The **final 2026 prediction system** "
                "is a blended model that also uses FIFA ranking, ELO, and squad/player statistics."
            )
            labels = ["B wins", "Draw", "A wins"]
            fig_cm = px.imshow(
                cm, x=labels, y=labels,
                text_auto=True, color_continuous_scale="Greens",
                title=f"{shorten_model_name(bm)} — rows=actual, cols=predicted (draw-rule)",
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                aspect="auto", height=340,
            )
            _apply_clean_layout(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

            # Feature importance
        fi = m.get("feature_importances")
        if fi:
            st.markdown("**Feature importance**")
            fi_df = pd.DataFrame(
                sorted(fi.items(), key=lambda x: x[1]),
                columns=["Feature", "Importance"]
            )
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                title="What drives the model's predictions",
                color="Importance", color_continuous_scale="Greens",
                height=340,
            )
            _apply_clean_layout(fig_fi)
            st.plotly_chart(fig_fi, use_container_width=True)

    # Model comparison
    models_d = m.get("models", {})
    if len(models_d) > 1:
        st.markdown("**Model comparison**")
        cmp = [
            {"Model": shorten_model_name(n),
             "Accuracy": v.get("accuracy") or 0,
             "F1 Macro": v.get("f1_macro") or 0}
            for n, v in models_d.items()
        ]
        fig_cmp = px.bar(
            pd.DataFrame(cmp), x="Model", y=["Accuracy", "F1 Macro"],
            barmode="group",
            color_discrete_sequence=["#2e7d32", "#d4af37"],
            height=300,
        )
        _apply_clean_layout(fig_cmp)
        st.plotly_chart(fig_cmp, use_container_width=True)

    # Training history
    if not df_hist.empty:
        with st.expander("Training history", expanded=False):
            st.dataframe(df_hist, use_container_width=True)
            if {"accuracy", "f1_macro", "run_at"}.issubset(df_hist.columns):
                fig_h = px.line(
                    df_hist.sort_values("run_at"), x="run_at",
                    y=["accuracy", "f1_macro"], markers=True,
                    labels={"run_at": "Run", "value": "Score", "variable": "Metric"},
                    color_discrete_sequence=["#2e7d32", "#d4af37"],
                    height=280,
                )
                _apply_clean_layout(fig_h)
                st.plotly_chart(fig_h, use_container_width=True)

    # Features
    feats = m.get("features")
    if feats:
        with st.expander("Features used", expanded=False):
            st.markdown(
                " ".join(
                    f'<code style="background:#e8f1e9;color:#2e7d32;'
                    f'border:1px solid #cfe3d0;border-radius:4px;padding:2px 7px;'
                    f'font-size:0.78rem;font-weight:600;font-family:Inter,sans-serif">{f}</code>'
                    for f in feats
                ),
                unsafe_allow_html=True,
            )

    # Raw JSON
    if m:
        with st.expander("Raw metrics.json", expanded=False):
            st.json(m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    render_page_header()
    _, btn_col = st.columns([8, 1])
    with btn_col:
        if st.button("↻ Refresh", use_container_width=True):
            try:
                with open("/app/data/preprocess_trigger.flag", "w") as f:
                    f.write("")
            except Exception:
                pass
            st.cache_data.clear()
            st.rerun()



    kpis = get_kpis()

    render_kpi_grid(kpis)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Predictions", "Sentiment", "Trends", "History", "Model"]
    )

    with tab1:
        tab_predictions()
    with tab2:
        tab_sentiment()
    with tab3:
        tab_trending()
    with tab4:
        tab_historical()
    with tab5:
        tab_model()

    render_footer(kpis)


main()
