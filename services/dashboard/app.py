"""World Cup 2026 Analytics Platform — Streamlit Dashboard."""
from __future__ import annotations

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
    page_title="World Cup 2026 Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ── page chrome ─────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── KPI card ────────────────────────────────────────────────────── */
.kpi-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem 0.9rem;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
    box-sizing: border-box;
}
.kpi-value {
    font-size: clamp(1.35rem, 2vw, 1.9rem);
    font-weight: 700;
    color: #60c9f8;
    line-height: 1.25;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}
.kpi-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-caption {
    font-size: 0.72rem;
    color: #4b5563;
    margin-top: 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── hero ─────────────────────────────────────────────────────────── */
.hero-title {
    font-size: clamp(1.8rem, 3vw, 2.4rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f9fafb;
    line-height: 1.2;
}
.hero-sub {
    color: #9ca3af;
    font-size: 1rem;
    margin-top: 0.3rem;
}
.hero-pills {
    margin-top: 0.6rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.pill {
    background: #1e3a5f;
    color: #93c5fd;
    border-radius: 999px;
    padding: 0.18rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
}

/* ── section header ─────────────────────────────────────────────── */
.sec-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f3f4f6;
    margin-bottom: 0.15rem;
}
.sec-sub {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 0.8rem;
}

/* ── divider ─────────────────────────────────────────────────────── */
.div-line {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 1rem 0 1.2rem;
}

/* ── empty-state box ─────────────────────────────────────────────── */
.empty-box {
    border: 1px dashed #374151;
    border-radius: 10px;
    padding: 2.2rem 1rem;
    text-align: center;
    color: #4b5563;
    margin: 0.6rem 0 1rem;
}
.empty-icon { font-size: 2rem; }
.empty-msg  { font-size: 0.9rem; margin-top: 0.4rem; }
.empty-hint { font-size: 0.78rem; margin-top: 0.4rem; font-style: italic; color: #374151; }

/* ── sidebar ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0d1117;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    font-size: 0.83rem;
    color: #9ca3af;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.82rem;
    font-weight: 700;
    color: #60c9f8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.8rem 0 0.3rem;
}

/* ── tab label font ─────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 0.9rem;
    font-weight: 600;
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


def render_page_header() -> None:
    st.markdown(
        """
<div class="hero-title">⚽ World Cup 2026 Analytics</div>
<div class="hero-sub">Match predictions · Reddit sentiment · Tournament insights</div>
<div class="hero-pills">
  <span class="pill">Reddit RSS</span>
  <span class="pill">VADER NLP</span>
  <span class="pill">XGBoost</span>
  <span class="pill">SQLite</span>
  <span class="pill">Streamlit</span>
</div>
""",
        unsafe_allow_html=True,
    )

    if not database_exists():
        st.error(
            "**Database not found.** Start the full pipeline with:  \n"
            "`docker compose up -d`"
        )

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


def section_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sec-title">{title}</div>{sub}',
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


def render_sidebar(kpis: dict, sent_teams: list[str]) -> dict:
    sb = st.sidebar

    # ── Refresh ──
    sb.markdown("### 🔄 Controls")
    if sb.button("↺ Refresh data", use_container_width=True, help="Clear all cached data"):
        st.cache_data.clear()
        st.rerun()

    # ── Sentiment filters ──
    sb.markdown("### 📊 Sentiment")
    default_teams = sent_teams[:6] if len(sent_teams) >= 6 else sent_teams
    sel_sent_teams = sb.multiselect("Teams", sent_teams, default=default_teams)
    sentiment_metric = sb.selectbox(
        "Metric",
        ["avg_vader", "avg_textblob", "hype_index"],
        help=(
            "avg_vader: VADER −1→+1  |  "
            "avg_textblob: TextBlob −1→+1  |  "
            "hype_index: volume × positivity"
        ),
    )

    # ── Predictions filter ──
    sb.markdown("### 🔮 Predictions")
    sel_groups: list[str] = []
    if table_exists("match_predictions"):
        df_g = load_table(
            "match_predictions",
            "SELECT DISTINCT group_name FROM match_predictions "
            "WHERE tournament_year=2026 AND group_name IS NOT NULL ORDER BY group_name",
        )
        if not df_g.empty and "group_name" in df_g.columns:
            all_groups = sorted(df_g["group_name"].dropna().unique().tolist())
            sel_groups = sb.multiselect("Groups", all_groups, default=all_groups)

    # ── Historical filter ──
    sb.markdown("### 📜 History")
    year_range = None
    hist_team = "All teams"

    if table_exists("raw_matches"):
        df_y = load_table("raw_matches", "SELECT DISTINCT year FROM raw_matches ORDER BY year")
        if not df_y.empty and "year" in df_y.columns:
            years = sorted(df_y["year"].dropna().astype(int).unique().tolist())
            if len(years) >= 2:
                year_range = sb.slider("Year range", years[0], years[-1], (years[0], years[-1]))
            elif years:
                sb.caption(f"Only one year available: {years[0]}")
                year_range = (years[0], years[0])

        df_t = load_table(
            "raw_matches",
            "SELECT DISTINCT team_a AS t FROM raw_matches "
            "UNION SELECT DISTINCT team_b FROM raw_matches ORDER BY t",
        )
        all_teams: list[str] = (
            sorted(df_t["t"].dropna().unique().tolist()) if not df_t.empty and "t" in df_t.columns else []
        )
        hist_team = sb.selectbox("Team", ["All teams"] + all_teams)

    # ── Model status ──
    sb.markdown("### 🤖 Model")
    if kpis.get("model_status") == "trained":
        sb.success(
            f"**{kpis.get('best_model', '—')}**  \n"
            f"Acc {fmt(kpis.get('accuracy'))}  ·  F1 {fmt(kpis.get('f1_macro'))}  \n"
            f"Trained {kpis.get('run_at', '—')}"
        )
    else:
        sb.warning("Not yet trained.")

    # ── Data sources ──
    sb.markdown("### ℹ️ Data Sources")
    sb.caption(
        "**Football CSV** — Kaggle historical match & ranking files  \n"
        "**Reddit RSS** — public feeds, no API key required  \n"
        "**Database** — local SQLite at `/app/data/worldcup.db`"
    )

    # ── Data health ──
    with sb.expander("🩺 Data Health", expanded=False):
        tbls = [
            "raw_reddit_posts", "processed_posts", "team_sentiment_daily",
            "trending_words", "match_predictions", "model_metrics",
        ]
        for t in tbls:
            n = scalar(f"SELECT COUNT(*) FROM {t}", None)
            if n is None:
                sb.markdown(f"⚠️ `{t}` — missing")
            elif n == 0:
                sb.markdown(f"🔴 `{t}` — **empty**")
            else:
                sb.markdown(f"🟢 `{t}` — {n:,}")
        sb.caption(
            "Integration check:  \n"
            "`docker compose run --rm preprocessor python /scripts/integration_check.py`"
        )

    return {
        "sent_teams": sel_sent_teams,
        "sentiment_metric": sentiment_metric,
        "sel_groups": sel_groups,
        "year_range": year_range,
        "hist_team": hist_team,
    }


# ---------------------------------------------------------------------------
# Tab helpers — common chart settings
# ---------------------------------------------------------------------------

CHART_H = 380
TRANSPARENT = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")


def _apply_clean_layout(fig, **extra):
    fig.update_layout(**TRANSPARENT, **extra)
    return fig


# ---------------------------------------------------------------------------
# Tab 1 — Predictions
# ---------------------------------------------------------------------------


def tab_predictions(filters: dict) -> None:
    section_header(
        "🔮 2026 Group Stage Predictions",
        "ML predictions for all 72 group stage matches. "
        "Confidence ≥ 55% = high, below = uncertain.",
    )

    df = load_table("match_predictions")
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

    sel_groups = filters.get("sel_groups")
    if sel_groups and "group_name" in df26.columns:
        df26 = df26[df26["group_name"].isin(sel_groups)]

    if df26.empty:
        empty_state("No predictions match the selected groups.")
        return

    # Summary cards
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("Matches", len(df26))
    if "confidence" in df26.columns:
        best = df26.loc[df26["confidence"].idxmax()]
        c2.metric("Highest confidence",
                  f"{best.get('team_a','?')} vs {best.get('team_b','?')}")
        c3.metric("Max confidence", f"{best['confidence']:.1%}")
        c4.metric("Avg confidence", f"{df26['confidence'].mean():.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Table
    col_map = {
        "group_name": "Group", "team_a": "Team A", "team_b": "Team B",
        "predicted_winner": "Winner", "confidence": "Confidence", "stage": "Stage",
    }
    display_cols = [c for c in col_map if c in df26.columns]
    df_show = df26[display_cols].rename(columns=col_map).copy()

    if "Confidence" in df_show.columns:
        df_show = df_show.sort_values("Confidence", ascending=False)

        def _cell(v):
            if isinstance(v, (float, int)):
                return (
                    "background:#1a3a2a;color:#4caf50"
                    if v >= HIGH_CONF_THRESHOLD
                    else "background:#3a2a10;color:#ffa726"
                )
            return ""

        styled = df_show.style.map(_cell, subset=["Confidence"]).format(
            {"Confidence": "{:.1%}"}
        )
        with st.expander("Predictions table", expanded=True):
            st.dataframe(styled, use_container_width=True, height=400)
    else:
        with st.expander("Predictions table", expanded=True):
            st.dataframe(df_show, use_container_width=True, height=400)

    # Bar chart
    if {"team_a", "team_b", "confidence", "predicted_winner"}.issubset(df26.columns):
        st.markdown("**Confidence by match**")
        df_plot = df26.copy()
        df_plot["Match"] = df_plot["team_a"] + " vs " + df_plot["team_b"]
        fig = px.bar(
            df_plot.sort_values("confidence", ascending=False).head(36),
            x="Match", y="confidence", color="predicted_winner",
            labels={"confidence": "Confidence", "predicted_winner": "Winner"},
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=CHART_H,
        )
        fig.add_hline(y=HIGH_CONF_THRESHOLD, line_dash="dash",
                      line_color="#4caf50", opacity=0.6,
                      annotation_text="55% threshold")
        _apply_clean_layout(fig, xaxis_tickangle=-50, margin=dict(b=110),
                            legend_title_text="Winner")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Sentiment
# ---------------------------------------------------------------------------

METRIC_DESC = {
    "avg_vader":    "VADER compound score — ranges −1 (negative) to +1 (positive).",
    "avg_textblob": "TextBlob polarity — ranges −1 to +1, secondary sentiment signal.",
    "hype_index":   "Hype index = (post volume / daily max) × max(0, avg VADER). Combines buzz and positivity.",
}


def tab_sentiment(filters: dict) -> None:
    section_header(
        "📊 Reddit Sentiment Tracker",
        "Daily sentiment scores per team, derived from Reddit post titles and text.",
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

    metric = filters.get("sentiment_metric", "avg_vader")
    if metric not in df.columns:
        st.info(f"Metric '{metric}' not available. Falling back to avg_vader.")
        metric = "avg_vader"

    sel_teams = filters.get("sent_teams") or []
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
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=CHART_H,
        )
        _apply_clean_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(
            df_f.sort_values("date"), x="date", y=metric, color="team",
            markers=True,
            labels={metric: metric_label, "date": "Date"},
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=CHART_H,
        )
        fig.update_traces(line_width=2, marker_size=6)
        fig.add_hline(y=0, line_dash="dot", line_color="#374151", opacity=0.6)
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
                color_discrete_sequence=px.colors.qualitative.Safe,
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


def tab_trending() -> None:
    section_header(
        "🔤 Trending Topics",
        "Most frequent words extracted from Reddit posts after stopword removal.",
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

    teams = sorted(df["team"].dropna().unique().tolist())
    if not teams:
        empty_state("No team-tagged word data available.")
        return

    col_sel, col_n = st.columns([3, 1], gap="small")
    with col_sel:
        sel_team = st.selectbox("Team", teams, key="tw_team")
    with col_n:
        top_n = st.slider("Top N", 5, 30, 15, key="tw_n")

    df_t = df[df["team"] == sel_team].copy()

    if "date" in df_t.columns:
        dates = sorted(df_t["date"].dropna().unique().tolist())
        if len(dates) > 1:
            pick = st.multiselect("Filter by date (optional)", dates, key="tw_dates",
                                  help="Leave blank to aggregate all dates")
            if pick:
                df_t = df_t[df_t["date"].isin(pick)]

    top = df_t.groupby("word", as_index=False)["frequency"].sum().nlargest(top_n, "frequency")

    if top.empty:
        empty_state(f"No word data for {sel_team}.")
        return

    left, right = st.columns([3, 2], gap="small")
    with left:
        fig = px.bar(
            top.sort_values("frequency"),
            x="frequency", y="word", orientation="h",
            labels={"frequency": "Frequency", "word": "Word"},
            color="frequency",
            color_continuous_scale="Blues",
            height=max(320, top_n * 23),
        )
        _apply_clean_layout(fig, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.caption(f"{len(df_t)} unique words for **{sel_team}**")
        st.dataframe(
            top.sort_values("frequency", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=max(320, top_n * 23),
        )


# ---------------------------------------------------------------------------
# Tab 4 — Historical Stats
# ---------------------------------------------------------------------------


def tab_historical(filters: dict) -> None:
    section_header(
        "📜 Historical World Cup Data",
        "Match results from FIFA World Cup tournaments (1930–2022).",
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

    # Apply sidebar filters
    yr = filters.get("year_range")
    if yr and "year" in df.columns:
        df = df[df["year"].between(yr[0], yr[1])]

    ht = filters.get("hist_team", "All teams")
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
                         color="Matches", color_continuous_scale="Blues",
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
                color_discrete_sequence=["#4caf50", "#ffa726", "#ef5350"],
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
        "🤖 Model Performance",
        "Trained on 1990–2018 match data; tested on 2022. "
        "Dataset is small (~64 matches) — metrics are indicative.",
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
        c1.metric("Best model",  shorten_model_name(m.get("best_model")))
        c2.metric("Accuracy",    fmt(m.get("accuracy")))
        c3.metric("F1 Macro",    fmt(m.get("f1_macro")))
        c4.metric("Train rows",  m.get("train_rows", "—"))
        c5.metric("Test rows",   m.get("test_rows", "—"))

        if m.get("run_at"):
            st.caption(f"Last trained: {str(m['run_at'])[:19]}")

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
            labels = ["B wins", "Draw", "A wins"]
            fig_cm = px.imshow(
                cm, x=labels, y=labels,
                text_auto=True, color_continuous_scale="Blues",
                title=f"{shorten_model_name(bm)} — rows=actual, cols=predicted",
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                aspect="auto", height=340,
            )
            _apply_clean_layout(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

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
            color_discrete_sequence=["#60c9f8", "#f97316"],
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
                    color_discrete_sequence=["#60c9f8", "#f97316"],
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
                    f'<code style="background:#1e3a5f;color:#93c5fd;'
                    f'border-radius:4px;padding:2px 6px;font-size:0.8rem">{f}</code>'
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

    kpis = get_kpis()

    # Collect team list for sidebar
    df_st = load_table(
        "team_sentiment_daily",
        "SELECT DISTINCT team FROM team_sentiment_daily ORDER BY team",
    )
    sent_teams = (
        df_st["team"].dropna().tolist()
        if not df_st.empty and "team" in df_st.columns
        else []
    )

    filters = render_sidebar(kpis, sent_teams)

    render_kpi_grid(kpis)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Predictions", "Sentiment", "Trends", "History", "Model"]
    )

    with tab1:
        tab_predictions(filters)
    with tab2:
        tab_sentiment(filters)
    with tab3:
        tab_trending()
    with tab4:
        tab_historical(filters)
    with tab5:
        tab_model()


main()
