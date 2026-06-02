"""World Cup 2026 Analytics Platform — Streamlit Dashboard."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

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
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="World Cup 2026 Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS  — cards, layout, typography
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── global page ──────────────────────────────────────── */
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }
    div[data-testid="stTabContent"] { padding-top: 1rem; }

    /* ── KPI card ──────────────────────────────────────────── */
    .kpi-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 0.9rem 1rem 0.7rem;
        text-align: center;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
        box-sizing: border-box;
    }
    .kpi-card .kv {
        font-size: clamp(1.3rem, 2.2vw, 1.9rem);
        font-weight: 700;
        color: #58a6ff;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    .kpi-card .kl {
        font-size: 0.68rem;
        color: #8b949e;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-card .kc {
        font-size: 0.65rem;
        color: #484f58;
        margin-top: 0.15rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── section header ────────────────────────────────────── */
    .sec-header {
        border-left: 3px solid #58a6ff;
        padding-left: 0.65rem;
        margin: 0.6rem 0 0.25rem;
    }
    .sec-header .sh-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e6edf3;
        line-height: 1.2;
    }
    .sec-header .sh-sub {
        font-size: 0.78rem;
        color: #8b949e;
        margin-top: 0.1rem;
    }

    /* ── hero ──────────────────────────────────────────────── */
    .hero-wrap { margin-bottom: 0.5rem; }
    .hero-title {
        font-size: clamp(1.7rem, 3.5vw, 2.5rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #e6edf3;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #8b949e;
        margin-top: 0.25rem;
    }
    .pill {
        display: inline-block;
        border: 1px solid #21262d;
        background: #161b22;
        color: #58a6ff;
        border-radius: 20px;
        padding: 0.12rem 0.6rem;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.35rem 0.15rem 0 0;
    }

    /* ── empty-state box ───────────────────────────────────── */
    .empty-box {
        padding: 2rem 1.5rem;
        border: 1px dashed #21262d;
        border-radius: 10px;
        text-align: center;
        color: #484f58;
        margin: 0.8rem 0;
    }
    .empty-box .ei { font-size: 1.6rem; }
    .empty-box .em { font-size: 0.88rem; margin-top: 0.4rem; }
    .empty-box code { color: #58a6ff; background: #0d1117; padding: 0.1rem 0.35rem; border-radius: 4px; }

    /* ── sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] { background-color: #0d1117; }
    section[data-testid="stSidebar"] hr { border-color: #21262d; margin: 0.6rem 0; }

    /* ── divider ───────────────────────────────────────────── */
    .hdiv { border: none; border-top: 1px solid #21262d; margin: 0.9rem 0; }
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
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error:
        return None


def table_exists(name: str) -> bool:
    c = _conn()
    if c is None:
        return False
    try:
        row = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None
    except Exception:
        return False
    finally:
        c.close()


@st.cache_data(ttl=300)
def load_table(table: str, query: str | None = None) -> pd.DataFrame:
    if not table_exists(table):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as c:
            return pd.read_sql_query(query or f"SELECT * FROM {table}", c)  # noqa: S608
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def scalar(sql: str, default=None):
    c = _conn()
    if c is None:
        return default
    try:
        r = c.execute(sql).fetchone()
        return r[0] if r else default
    except Exception:
        return default
    finally:
        c.close()


@st.cache_data(ttl=300)
def load_metrics_json() -> dict:
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
    """Format a numeric value or return '—'."""
    if val is None:
        return "—"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


_MODEL_ABBREV = {
    "XGBClassifier": "XGBoost",
    "RandomForestClassifier": "RandomForest",
    "GradientBoostingClassifier": "GradBoost",
    "LogisticRegression": "LogReg",
    "SVC": "SVC",
    "DecisionTreeClassifier": "DecTree",
}


def fmt_model(name: str | None) -> str:
    if not name:
        return "—"
    short = _MODEL_ABBREV.get(name, name)
    return short[:14] + "…" if len(short) > 14 else short


def fmt_date_range(min_d: str | None, max_d: str | None) -> tuple[str, str]:
    """Return (value, caption) for the date-range KPI card."""
    if not min_d or not max_d:
        return "—", ""
    caption = f"{min_d} → {max_d}"
    try:
        start = datetime.strptime(min_d, "%Y-%m-%d")
        end = datetime.strptime(max_d, "%Y-%m-%d")
        days = (end - start).days
        return f"{days}d", caption
    except ValueError:
        return "Available", caption


# ---------------------------------------------------------------------------
# KPI data
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def get_kpis() -> dict:
    k: dict = {}
    k["reddit_posts"] = scalar("SELECT COUNT(*) FROM raw_reddit_posts", 0)
    k["processed_posts"] = scalar("SELECT COUNT(*) FROM processed_posts", 0)
    k["telegram_posts"] = scalar("SELECT COUNT(*) FROM raw_telegram_posts", 0)
    k["teams"] = scalar("SELECT COUNT(DISTINCT team) FROM team_sentiment_daily", 0)
    k["predictions"] = scalar(
        "SELECT COUNT(*) FROM match_predictions WHERE tournament_year = 2026", 0
    )
    min_d = scalar(
        "SELECT date(datetime(MIN(created_utc),'unixepoch')) FROM raw_reddit_posts"
    )
    max_d = scalar(
        "SELECT date(datetime(MAX(created_utc),'unixepoch')) FROM raw_reddit_posts"
    )
    k["date_val"], k["date_cap"] = fmt_date_range(min_d, max_d)

    m = load_metrics_json()
    k["accuracy"] = m.get("accuracy")
    k["f1_macro"] = m.get("f1_macro")
    k["best_model"] = m.get("best_model")
    k["model_status"] = m.get("status", "unknown")
    k["run_at"] = m.get("run_at", "—")
    return k


# ---------------------------------------------------------------------------
# UI component helpers
# ---------------------------------------------------------------------------


def kpi_card(value: str, label: str, caption: str = "") -> str:
    cap_html = f'<div class="kc">{caption}</div>' if caption else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kv">{value}</div>'
        f'<div class="kl">{label}</div>'
        f"{cap_html}"
        f"</div>"
    )


def section_header(title: str, subtitle: str = "") -> None:
    sub_html = f'<div class="sh-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sec-header"><div class="sh-title">{title}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def empty_state(msg: str, hint: str = "") -> None:
    hint_html = f'<div class="em">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="empty-box"><div class="ei">🗂️</div>'
        f'<div class="em">{msg}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def hdiv() -> None:
    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def render_header() -> None:
    st.markdown(
        '<div class="hero-wrap">'
        '<div class="hero-title">⚽ World Cup 2026 Analytics</div>'
        '<div class="hero-sub">Match predictions · Reddit sentiment · Tournament insights</div>'
        '<span class="pill">RSS Feed</span>'
        '<span class="pill">VADER NLP</span>'
        '<span class="pill">XGBoost</span>'
        '<span class="pill">SQLite</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    if not database_exists():
        st.error(
            "**Database not found.** Start the pipeline with:  `docker compose up -d`"
        )
    hdiv()


# ---------------------------------------------------------------------------
# KPI grid  —  2 rows × 4 cards
# ---------------------------------------------------------------------------


def render_kpis(k: dict) -> None:
    # Row 1: volume metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(str(k.get("reddit_posts") or 0), "Reddit Posts"), unsafe_allow_html=True)
    c2.markdown(kpi_card(str(k.get("processed_posts") or 0), "Processed"), unsafe_allow_html=True)

    # Show Telegram count or teams depending on what's available
    tg = k.get("telegram_posts") or 0
    if tg:
        c3.markdown(kpi_card(str(tg), "Telegram Posts"), unsafe_allow_html=True)
    else:
        c3.markdown(kpi_card(str(k.get("teams") or 0), "Teams"), unsafe_allow_html=True)
    c4.markdown(kpi_card(str(k.get("predictions") or 0), "Predictions"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

    # Row 2: model + date
    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(kpi_card(fmt(k.get("accuracy")), "Accuracy"), unsafe_allow_html=True)
    c6.markdown(kpi_card(fmt(k.get("f1_macro")), "F1 Macro"), unsafe_allow_html=True)
    c7.markdown(
        kpi_card(fmt_model(k.get("best_model")), "Best Model"),
        unsafe_allow_html=True,
    )
    c8.markdown(
        kpi_card(k.get("date_val", "—"), "Date Range", k.get("date_cap", "")),
        unsafe_allow_html=True,
    )

    hdiv()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(k: dict, sent_teams: list[str]) -> dict:
    # Refresh
    st.sidebar.markdown("**🔄 Controls**")
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.markdown("---")

    # Sentiment filters
    st.sidebar.markdown("**📊 Sentiment**")
    default_teams = sent_teams[:6] if len(sent_teams) >= 6 else sent_teams
    sel_teams = st.sidebar.multiselect("Teams", sent_teams, default=default_teams)
    metric = st.sidebar.selectbox(
        "Metric",
        ["avg_vader", "hype_index", "avg_textblob"],
        help=(
            "avg_vader: VADER compound (−1 to 1)\n"
            "hype_index: volume × positivity\n"
            "avg_textblob: TextBlob polarity (−1 to 1)"
        ),
    )
    st.sidebar.markdown("---")

    # Predictions filter
    st.sidebar.markdown("**🔮 Predictions**")
    sel_groups: list[str] = []
    if table_exists("match_predictions"):
        df_g = load_table(
            "match_predictions",
            "SELECT DISTINCT group_name FROM match_predictions "
            "WHERE tournament_year=2026 ORDER BY group_name",
        )
        if not df_g.empty and "group_name" in df_g.columns:
            all_groups = sorted(df_g["group_name"].dropna().unique().tolist())
            sel_groups = st.sidebar.multiselect("Groups", all_groups, default=all_groups)
    st.sidebar.markdown("---")

    # Historical filters
    st.sidebar.markdown("**📜 Historical**")
    year_range = None
    hist_team = "All teams"
    if table_exists("raw_matches"):
        df_y = load_table("raw_matches", "SELECT DISTINCT year FROM raw_matches ORDER BY year")
        if not df_y.empty and "year" in df_y.columns:
            years = sorted(df_y["year"].dropna().astype(int).unique().tolist())
            if len(years) >= 2:
                year_range = st.sidebar.slider(
                    "Year range", years[0], years[-1], (years[0], years[-1])
                )
            elif years:
                st.sidebar.info(f"Only one year: {years[0]}")
                year_range = (years[0], years[0])
        df_t = load_table(
            "raw_matches",
            "SELECT DISTINCT team_a AS t FROM raw_matches "
            "UNION SELECT DISTINCT team_b FROM raw_matches ORDER BY t",
        )
        all_teams = (
            sorted(df_t["t"].dropna().unique().tolist())
            if not df_t.empty and "t" in df_t.columns
            else []
        )
        if all_teams:
            hist_team = st.sidebar.selectbox("Team", ["All teams"] + all_teams)
    st.sidebar.markdown("---")

    # Model status
    st.sidebar.markdown("**🤖 Model**")
    if k.get("model_status") == "trained":
        st.sidebar.success(
            f"{fmt_model(k.get('best_model'))}  \n"
            f"Acc: {fmt(k.get('accuracy'))}  F1: {fmt(k.get('f1_macro'))}  \n"
            f"{str(k.get('run_at', ''))[:16]}"
        )
    else:
        st.sidebar.warning("Model not trained yet.")
    st.sidebar.markdown("---")

    # Data sources
    st.sidebar.markdown("**ℹ️ Data Sources**")
    st.sidebar.caption(
        "Football stats from CSV files.  \n"
        "Reddit posts from public RSS feeds.  \n"
        "Telegram from public channels (optional).  \n"
        "Storage: local SQLite database."
    )

    # Data health
    with st.sidebar.expander("🩺 Data Health", expanded=False):
        for tbl in [
            "raw_reddit_posts", "raw_telegram_posts", "processed_posts",
            "team_sentiment_daily", "match_predictions",
        ]:
            n = scalar(f"SELECT COUNT(*) FROM {tbl}", None)
            icon = "✅" if n and n > 0 else ("⚠️" if n == 0 else "—")
            st.markdown(f"{icon} `{tbl}`: {n if n is not None else 'missing'}")
        st.caption(
            "Run pipeline check:  \n"
            "`docker compose run --rm preprocessor "
            "python /scripts/integration_check.py`"
        )

    return {
        "sent_teams": sel_teams,
        "metric": metric,
        "sel_groups": sel_groups,
        "hist_team": hist_team,
        "year_range": year_range,
    }


# ---------------------------------------------------------------------------
# Tab: Predictions
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "avg_vader": "VADER compound score (−1 to +1). Negative = pessimistic, positive = optimistic.",
    "hype_index": "Hype index = (post volume / daily max) × max(0, avg VADER). Captures volume and positivity.",
    "avg_textblob": "TextBlob polarity (−1 to +1). Secondary sentiment signal complementing VADER.",
}


def render_predictions_tab(f: dict) -> None:
    section_header(
        "2026 Group Stage Predictions",
        "Generated by the trained model using ELO ratings, FIFA rankings, and historical head-to-head stats.",
    )

    df = load_table("match_predictions")
    if df.empty:
        empty_state(
            "No predictions yet.",
            "Run: <code>docker compose run --rm trainer</code>",
        )
        return

    df26 = (
        df[df["tournament_year"] == 2026].copy()
        if "tournament_year" in df.columns
        else df.copy()
    )

    sel_groups = f.get("sel_groups")
    if sel_groups and "group_name" in df26.columns:
        df26 = df26[df26["group_name"].isin(sel_groups)]

    if df26.empty:
        empty_state("No predictions match the selected groups.")
        return

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions", len(df26))
    if "confidence" in df26.columns:
        best = df26.loc[df26["confidence"].idxmax()]
        c2.metric("Most Confident", f"{best.get('team_a','?')} vs {best.get('team_b','?')}")
        c3.metric("Max Confidence", f"{best['confidence']:.0%}")
        c4.metric("Avg Confidence", f"{df26['confidence'].mean():.0%}")

    hdiv()

    # Table
    display_cols = [
        c for c in ["group_name", "team_a", "team_b", "predicted_winner", "confidence", "stage"]
        if c in df26.columns
    ]
    rename = {
        "group_name": "Group", "team_a": "Team A", "team_b": "Team B",
        "predicted_winner": "Winner", "confidence": "Confidence", "stage": "Stage",
    }
    df_show = df26[display_cols].rename(columns=rename).copy()
    if "Confidence" in df_show.columns:
        df_show = df_show.sort_values("Confidence", ascending=False)

        def _cc(v):
            if isinstance(v, (float, int)):
                return ("background-color:#1a3a2a;color:#4caf50"
                        if v >= HIGH_CONF_THRESHOLD
                        else "background-color:#3a2010;color:#e67c1b")
            return ""

        styled = df_show.style.map(_cc, subset=["Confidence"]).format({"Confidence": "{:.0%}"})
        st.dataframe(styled, use_container_width=True, height=400)
    else:
        st.dataframe(df_show, use_container_width=True, height=400)

    # Chart
    if {"team_a", "team_b", "confidence", "predicted_winner"}.issubset(df26.columns):
        st.markdown("#### Confidence by Match")
        df_chart = df26.copy()
        df_chart["Match"] = df_chart["team_a"] + " vs " + df_chart["team_b"]
        fig = px.bar(
            df_chart.sort_values("confidence", ascending=False).head(40),
            x="Match",
            y="confidence",
            color="predicted_winner",
            labels={"confidence": "Confidence", "predicted_winner": "Predicted Winner"},
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=380,
        )
        fig.add_hline(
            y=HIGH_CONF_THRESHOLD, line_dash="dash", line_color="#4caf50",
            annotation_text="High-confidence threshold",
        )
        fig.update_layout(
            xaxis_tickangle=-50, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(b=110),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Sentiment Tracker
# ---------------------------------------------------------------------------


def render_sentiment_tab(f: dict) -> None:
    section_header(
        "Reddit Sentiment Tracker",
        "Daily sentiment scores and hype index computed from Reddit post titles and bodies using VADER and TextBlob.",
    )

    df = load_table("team_sentiment_daily")
    if df.empty:
        empty_state(
            "No sentiment data yet.",
            "Run: <code>docker compose run --rm preprocessor</code>",
        )
        return

    required = {"team", "date", "avg_vader"}
    if not required.issubset(df.columns):
        st.warning(f"Expected columns {required}. Found: {list(df.columns)}")
        st.dataframe(df.head(20), use_container_width=True)
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["avg_vader"] = pd.to_numeric(df["avg_vader"], errors="coerce")
    for col in ["avg_textblob", "hype_index", "post_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "avg_vader", "team"])

    metric = f.get("metric", "avg_vader")
    sel_teams = f.get("sent_teams", [])

    if metric not in df.columns:
        st.info(f"Metric '{metric}' not in data yet — showing avg_vader.")
        metric = "avg_vader"

    if metric in METRIC_LABELS:
        st.caption(f"📌 {METRIC_LABELS[metric]}")

    df_f = df[df["team"].isin(sel_teams)] if sel_teams else df.copy()
    if df_f.empty:
        empty_state("No data for the selected teams.")
        return

    unique_dates = df_f["date"].nunique()
    metric_label = metric.replace("_", " ").title()

    if unique_dates <= 1:
        st.info("Only one date available — bar chart shown instead of time series.")
        agg = df_f.groupby("team")[metric].mean().reset_index()
        fig = px.bar(
            agg, x="team", y=metric, color="team",
            labels={metric: metric_label, "team": "Team"},
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=340,
        )
    else:
        fig = px.line(
            df_f.sort_values("date"), x="date", y=metric, color="team",
            markers=True,
            labels={metric: metric_label, "date": "Date"},
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=380,
        )
        fig.update_traces(line_width=2, marker_size=5)
        fig.add_hline(y=0, line_dash="dot", line_color="#444", opacity=0.6)

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        showlegend=(unique_dates > 1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hype bar (only when metric is not hype_index)
    if "hype_index" in df_f.columns and metric != "hype_index":
        agg_h = df_f.groupby("team", as_index=False)["hype_index"].mean().dropna()
        if not agg_h.empty and agg_h["hype_index"].sum() > 0:
            st.markdown("##### Avg Hype Index by Team")
            fig_h = px.bar(
                agg_h.sort_values("hype_index", ascending=False),
                x="team", y="hype_index", color="team",
                labels={"hype_index": "Hype Index", "team": "Team"},
                color_discrete_sequence=px.colors.qualitative.Safe,
                height=280,
            )
            fig_h.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # Recent data table
    hdiv()
    st.markdown("##### Recent sentiment rows")
    preview = [c for c in ["team", "date", "avg_vader", "avg_textblob", "hype_index", "post_count"]
               if c in df_f.columns]
    st.dataframe(
        df_f[preview].sort_values("date", ascending=False).head(30).reset_index(drop=True),
        use_container_width=True,
        height=280,
    )


# ---------------------------------------------------------------------------
# Tab: Trending Topics
# ---------------------------------------------------------------------------


def render_trending_tab() -> None:
    section_header(
        "Trending Topics",
        "Most frequent words extracted from cleaned Reddit post text per team. "
        "Common English stopwords are removed.",
    )

    df = load_table("trending_words")
    if df.empty:
        empty_state(
            "No trending word data yet.",
            "Run: <code>docker compose run --rm preprocessor</code>",
        )
        return

    required = {"team", "word", "frequency"}
    if not required.issubset(df.columns):
        st.warning(f"Columns {required} not found. Available: {list(df.columns)}")
        st.dataframe(df.head(20), use_container_width=True)
        return

    teams = sorted(df["team"].dropna().unique().tolist())
    if not teams:
        empty_state("No team-tagged word data available.")
        return

    col_a, col_b = st.columns([3, 1])
    with col_a:
        sel_team = st.selectbox("Team", teams, key="tr_team")
    with col_b:
        top_n = st.slider("Top N", 5, 30, 15, key="tr_n")

    df_t = df[df["team"] == sel_team].copy()

    # Optional date filter
    if "date" in df_t.columns:
        dates = sorted(df_t["date"].dropna().unique().tolist())
        if len(dates) > 1:
            sel_d = st.multiselect("Date (optional)", dates, default=[], key="tr_d",
                                   help="Leave empty to aggregate all dates.")
            if sel_d:
                df_t = df_t[df_t["date"].isin(sel_d)]

    top = df_t.groupby("word", as_index=False)["frequency"].sum().nlargest(top_n, "frequency")
    if top.empty:
        empty_state(f"No words found for {sel_team}.")
        return

    col_chart, col_tbl = st.columns([3, 2])
    with col_chart:
        fig = px.bar(
            top.sort_values("frequency"),
            x="frequency", y="word", orientation="h",
            labels={"frequency": "Frequency", "word": "Word"},
            color="frequency", color_continuous_scale="Blues",
            height=max(320, top_n * 22),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_tbl:
        st.markdown(f"**Top words — {sel_team}**")
        st.dataframe(
            top.sort_values("frequency", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=max(320, top_n * 22),
        )


# ---------------------------------------------------------------------------
# Tab: Historical Stats
# ---------------------------------------------------------------------------


def render_historical_tab(f: dict) -> None:
    section_header(
        "Historical World Cup Data",
        "Match results loaded from local CSV files covering FIFA World Cup tournaments.",
    )

    df = load_table("raw_matches")
    if df.empty:
        empty_state(
            "No historical data loaded.",
            "Place CSV files in datasets/ and run the collector.",
        )
        return

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in ["score_a", "score_b"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    yr = f.get("year_range")
    if yr and "year" in df.columns:
        df = df[df["year"].between(yr[0], yr[1])]

    ht = f.get("hist_team", "All teams")
    if ht != "All teams":
        mask = (df.get("team_a", pd.Series()) == ht) | (df.get("team_b", pd.Series()) == ht)
        df = df[mask]

    all_teams_v = sorted(
        set(df.get("team_a", pd.Series()).dropna().tolist())
        | set(df.get("team_b", pd.Series()).dropna().tolist())
    )

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", len(df))
    if "year" in df.columns and not df["year"].isna().all():
        c2.metric("Years", f"{int(df['year'].min())} – {int(df['year'].max())}")
    c3.metric("Teams", len(all_teams_v))
    if "stage" in df.columns:
        c4.metric("Stages", df["stage"].nunique())

    hdiv()

    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        if "year" in df.columns:
            mpy = df.groupby("year").size().reset_index(name="Matches")
            fig_y = px.bar(
                mpy, x="year", y="Matches", title="Matches per Tournament",
                color="Matches", color_continuous_scale="Blues",
                labels={"year": "Year"},
            )
            fig_y.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, height=300,
            )
            st.plotly_chart(fig_y, use_container_width=True)

    with ch2:
        if ht != "All teams" and "winner" in df.columns and not df.empty:
            wins = (df["winner"] == ht).sum()
            draws = (df["winner"] == "Draw").sum()
            losses = len(df) - wins - draws
            fig_pie = px.pie(
                values=[max(0, wins), max(0, draws), max(0, losses)],
                names=["Wins", "Draws", "Losses"],
                title=f"{ht} — Record",
                color_discrete_sequence=["#4caf50", "#e67c1b", "#ef5350"],
                hole=0.4,
                height=300,
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)
        elif "score_a" in df.columns:
            avg_g = (df["score_a"].fillna(0) + df["score_b"].fillna(0)).mean()
            st.metric("Avg goals / match", f"{avg_g:.2f}")
            st.caption("Select a specific team in the sidebar to see their win/draw/loss record.")

    # Results table
    hdiv()
    st.markdown("##### Match Results")
    result_cols = [c for c in ["year", "stage", "team_a", "score_a", "score_b", "team_b", "winner"]
                   if c in df.columns]
    st.dataframe(
        df[result_cols].sort_values("year", ascending=False).head(200).reset_index(drop=True),
        use_container_width=True,
        height=360,
    )


# ---------------------------------------------------------------------------
# Tab: Model Performance
# ---------------------------------------------------------------------------


def render_model_tab() -> None:
    section_header(
        "Model Performance",
        "Trained on FIFA World Cup 1990–2022 data (~64 matches). "
        "Dataset is small — metrics are indicative, not production-grade.",
    )

    m = load_metrics_json()
    df_hist = load_table("model_metrics")

    if not m and df_hist.empty:
        empty_state(
            "No model metrics yet.",
            "Run: <code>docker compose run --rm trainer</code>",
        )
        return

    if m.get("status") == "trained":
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best Model", fmt_model(m.get("best_model")))
        c2.metric("Accuracy", fmt(m.get("accuracy")))
        c3.metric("F1 Macro", fmt(m.get("f1_macro")))
        c4.metric("Train Rows", m.get("train_rows", "—"))
        c5.metric("Test Rows", m.get("test_rows", "—"))
        if m.get("run_at"):
            st.caption(f"Last trained: {str(m['run_at'])[:19]}")

        acc = m.get("accuracy")
        if acc is not None and float(acc) < 0.55:
            st.info(
                "Accuracy below 55% is expected for this small dataset. "
                "Random baseline for a 3-class problem is ~33%, so any higher score is meaningful."
            )
    elif m:
        st.warning(f"Model not trained yet. Reason: {m.get('reason', 'unknown')}")

    hdiv()

    # Confusion matrix
    best = m.get("best_model")
    if best:
        cm = m.get("models", {}).get(best, {}).get("confusion_matrix")
        if cm:
            st.markdown("##### Confusion Matrix")
            labels = ["B wins", "Draw", "A wins"]
            fig_cm = px.imshow(
                cm, x=labels, y=labels, text_auto=True,
                color_continuous_scale="Blues",
                title=f"{fmt_model(best)} — rows = actual, cols = predicted",
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                aspect="auto", height=340,
            )
            fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cm, use_container_width=True)

    # Model comparison
    models_d = m.get("models", {})
    if len(models_d) > 1:
        st.markdown("##### Model Comparison")
        cmp = [
            {"Model": fmt_model(n), "Accuracy": v.get("accuracy") or 0, "F1 Macro": v.get("f1_macro") or 0}
            for n, v in models_d.items()
        ]
        fig_c = px.bar(
            pd.DataFrame(cmp), x="Model", y=["Accuracy", "F1 Macro"],
            barmode="group",
            color_discrete_sequence=["#58a6ff", "#f0883e"],
            height=300,
        )
        fig_c.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_c, use_container_width=True)

    # Training history
    if not df_hist.empty:
        hdiv()
        st.markdown("##### Training History")
        st.dataframe(df_hist, use_container_width=True)
        if {"accuracy", "f1_macro", "run_at"}.issubset(df_hist.columns):
            fig_h = px.line(
                df_hist.sort_values("run_at"), x="run_at", y=["accuracy", "f1_macro"],
                markers=True, labels={"run_at": "Run", "value": "Score", "variable": "Metric"},
                color_discrete_sequence=["#58a6ff", "#f0883e"], height=280,
            )
            fig_h.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # Features & raw JSON
    feats = m.get("features")
    if feats:
        hdiv()
        st.markdown("##### Features Used")
        st.caption("  ".join(f"`{f}`" for f in feats))

    if m:
        with st.expander("Raw metrics.json", expanded=False):
            st.json(m)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    render_header()

    k = get_kpis()

    df_st = load_table(
        "team_sentiment_daily",
        "SELECT DISTINCT team FROM team_sentiment_daily ORDER BY team",
    )
    sent_teams = (
        df_st["team"].dropna().tolist()
        if not df_st.empty and "team" in df_st.columns
        else []
    )

    filters = render_sidebar(k, sent_teams)
    render_kpis(k)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔮 Predictions", "📊 Sentiment", "🔤 Trends", "📜 History", "🤖 Model"]
    )
    with tab1:
        render_predictions_tab(filters)
    with tab2:
        render_sentiment_tab(filters)
    with tab3:
        render_trending_tab()
    with tab4:
        render_historical_tab(filters)
    with tab5:
        render_model_tab()


main()
