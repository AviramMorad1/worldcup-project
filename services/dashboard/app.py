import json
import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = "/app/data/worldcup.db"
METRICS_JSON_PATH = "/app/data/models/metrics.json"

st.set_page_config(
    page_title="World Cup Analytics",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def database_exists() -> bool:
    return os.path.exists(DB_PATH)


def table_exists(table_name: str) -> bool:
    if not database_exists():
        return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


@st.cache_data(ttl=300)
def load_table(table_name: str) -> pd.DataFrame:
    if not table_exists(table_name):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)  # noqa: S608
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_metrics_json() -> dict:
    if not os.path.exists(METRICS_JSON_PATH):
        return {}
    try:
        with open(METRICS_JSON_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("⚽ World Cup 2026 Analytics Platform")

if not database_exists():
    st.warning(
        "Database not found at `%s`. Start the collector service to initialise it." % DB_PATH
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Match Predictions", "Sentiment Tracker", "Trending Words", "Historical Stats", "Model Performance"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Match Predictions
# ---------------------------------------------------------------------------
with tab1:
    st.header("Match Predictions")
    df_pred = load_table("match_predictions")

    if df_pred.empty:
        st.info("No match predictions available yet. Run the trainer service first.")
    else:
        st.dataframe(df_pred, use_container_width=True)

        if "confidence" in df_pred.columns and "team_a" in df_pred.columns and "team_b" in df_pred.columns:
            df_pred["matchup"] = df_pred["team_a"] + " vs " + df_pred["team_b"]
            fig = px.bar(
                df_pred,
                x="matchup",
                y="confidence",
                color="predicted_winner" if "predicted_winner" in df_pred.columns else None,
                title="Prediction Confidence by Match",
                labels={"matchup": "Match", "confidence": "Confidence"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2 — Sentiment Tracker
# ---------------------------------------------------------------------------
with tab2:
    st.header("Sentiment Tracker")
    df_sent = load_table("team_sentiment_daily")

    if df_sent.empty:
        st.info("No sentiment data available yet. Run the preprocessor service first.")
    else:
        required_cols = {"team", "date", "avg_vader"}
        if required_cols.issubset(df_sent.columns):
            teams = sorted(df_sent["team"].dropna().unique().tolist())
            selected_teams = st.sidebar.multiselect(
                "Filter by team (Sentiment)", teams, default=teams[:5] if len(teams) >= 5 else teams
            )
            df_filtered = df_sent[df_sent["team"].isin(selected_teams)] if selected_teams else df_sent

            fig_line = px.line(
                df_filtered,
                x="date",
                y="avg_vader",
                color="team",
                title="Average VADER Sentiment Over Time",
                labels={"avg_vader": "Avg VADER Score", "date": "Date"},
            )
            st.plotly_chart(fig_line, use_container_width=True)

            if "hype_index" in df_sent.columns:
                df_hype = (
                    df_filtered.groupby("team", as_index=False)["hype_index"].mean()
                )
                fig_hype = px.bar(
                    df_hype,
                    x="team",
                    y="hype_index",
                    title="Average Hype Index by Team",
                    labels={"hype_index": "Hype Index"},
                )
                st.plotly_chart(fig_hype, use_container_width=True)
        else:
            st.dataframe(df_sent, use_container_width=True)
            st.info("Expected columns (team, date, avg_vader) not yet present in the table.")

# ---------------------------------------------------------------------------
# Tab 3 — Trending Words
# ---------------------------------------------------------------------------
with tab3:
    st.header("Trending Words")
    df_words = load_table("trending_words")

    if df_words.empty:
        st.info("No trending word data available yet. Run the preprocessor service first.")
    else:
        required_cols = {"team", "word", "frequency"}
        if required_cols.issubset(df_words.columns):
            word_teams = sorted(df_words["team"].dropna().unique().tolist())
            selected_word_team = st.sidebar.selectbox(
                "Filter by team (Trending Words)",
                word_teams,
            )
            df_team_words = df_words[df_words["team"] == selected_word_team]
            top10 = df_team_words.nlargest(10, "frequency")

            fig_words = px.bar(
                top10,
                x="word",
                y="frequency",
                title=f"Top 10 Words — {selected_word_team}",
                labels={"word": "Word", "frequency": "Frequency"},
            )
            st.plotly_chart(fig_words, use_container_width=True)
        else:
            st.dataframe(df_words, use_container_width=True)
            st.info("Expected columns (team, word, frequency) not yet present in the table.")

# ---------------------------------------------------------------------------
# Tab 4 — Historical Stats
# ---------------------------------------------------------------------------
with tab4:
    st.header("Historical Stats")
    df_matches = load_table("raw_matches")

    if df_matches.empty:
        st.info("No historical match data loaded yet. Run the collector service first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Matches", len(df_matches))

        if "year" in df_matches.columns:
            col2.metric(
                "Year Range",
                f"{int(df_matches['year'].min())} – {int(df_matches['year'].max())}",
            )

        teams_count = 0
        for col in ("team_a", "home_team", "Home Team Name"):
            if col in df_matches.columns:
                teams_count = df_matches[col].nunique()
                break
        if teams_count:
            col3.metric("Unique Teams", teams_count)

        st.subheader("Data Preview")
        st.dataframe(df_matches.head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5 — Model Performance
# ---------------------------------------------------------------------------
with tab5:
    st.header("Model Performance")

    metrics_json = load_metrics_json()
    df_metrics = load_table("model_metrics")

    has_data = bool(metrics_json) or not df_metrics.empty

    if not has_data:
        st.info("No model metrics available yet. Run the trainer service first.")
    else:
        if metrics_json:
            st.subheader("Latest Model Metrics (metrics.json)")
            m1, m2 = st.columns(2)
            if "accuracy" in metrics_json:
                m1.metric("Accuracy", f"{metrics_json['accuracy']:.3f}")
            if "f1_macro" in metrics_json:
                m2.metric("F1 Macro", f"{metrics_json['f1_macro']:.3f}")
            if "run_at" in metrics_json:
                st.caption(f"Last trained: {metrics_json['run_at']}")

        if not df_metrics.empty:
            st.subheader("Training History")
            st.dataframe(df_metrics, use_container_width=True)

            if {"accuracy", "f1_macro", "run_at"}.issubset(df_metrics.columns):
                fig_metrics = px.line(
                    df_metrics.sort_values("run_at"),
                    x="run_at",
                    y=["accuracy", "f1_macro"],
                    title="Model Metrics Over Time",
                    labels={"run_at": "Run At", "value": "Score", "variable": "Metric"},
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
