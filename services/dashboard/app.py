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
    st.header("2026 Group Stage Predictions")
    df_pred = load_table("match_predictions")

    if df_pred.empty:
        st.info("No match predictions available yet. Run the trainer service first.")
    else:
        df_2026 = df_pred[df_pred["tournament_year"] == 2026].copy() if "tournament_year" in df_pred.columns else df_pred.copy()

        # Group filter in sidebar
        if "group_name" in df_2026.columns:
            all_groups = sorted(df_2026["group_name"].dropna().unique().tolist())
            selected_groups = st.sidebar.multiselect("Filter by Group", all_groups, default=all_groups)
            df_2026 = df_2026[df_2026["group_name"].isin(selected_groups)] if selected_groups else df_2026

        # Color-code confidence: green ≥ 0.55, yellow < 0.55
        def _confidence_color(val):
            if isinstance(val, float):
                color = "#28a745" if val >= 0.55 else "#ffc107"
                return f"background-color: {color}22; color: {'#155724' if val >= 0.55 else '#856404'}"
            return ""

        display_cols = [c for c in ["group_name", "team_a", "team_b", "predicted_winner", "confidence"] if c in df_2026.columns]
        styled = df_2026[display_cols].style.map(_confidence_color, subset=["confidence"] if "confidence" in display_cols else [])
        st.dataframe(styled, use_container_width=True)

        # Confidence bar chart per group
        if {"team_a", "team_b", "confidence", "predicted_winner"}.issubset(df_2026.columns):
            df_2026["matchup"] = df_2026["team_a"] + " vs " + df_2026["team_b"]
            fig = px.bar(
                df_2026.sort_values("confidence", ascending=False),
                x="matchup",
                y="confidence",
                color="predicted_winner",
                title="Prediction Confidence by Match (green ≥ 0.55 = high confidence)",
                labels={"matchup": "Match", "confidence": "Confidence", "predicted_winner": "Predicted Winner"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.add_hline(y=0.55, line_dash="dash", line_color="green", annotation_text="High confidence threshold")
            fig.update_layout(xaxis_tickangle=-45, height=500)
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
            # Coerce types so the chart actually renders
            df_sent = df_sent.copy()
            df_sent["date"] = pd.to_datetime(df_sent["date"], errors="coerce")
            df_sent["avg_vader"] = pd.to_numeric(df_sent["avg_vader"], errors="coerce")
            df_sent = df_sent.dropna(subset=["date", "avg_vader", "team"])

            teams = sorted(df_sent["team"].dropna().unique().tolist())
            selected_teams = st.sidebar.multiselect(
                "Filter by team (Sentiment)",
                teams,
                default=teams[:5] if len(teams) >= 5 else teams,
            )
            df_filtered = df_sent[df_sent["team"].isin(selected_teams)] if selected_teams else df_sent.copy()

            if df_filtered.empty:
                st.info("No sentiment rows match the current filters.")
            else:
                unique_dates = df_filtered["date"].nunique()
                if unique_dates <= 1:
                    st.info(
                        "Only one sentiment date is available, so a bar chart is shown "
                        "instead of a time-series line."
                    )
                    fig_sent = px.bar(
                        df_filtered,
                        x="team",
                        y="avg_vader",
                        color="team",
                        title="Average VADER Sentiment by Team",
                        labels={"avg_vader": "Avg VADER Score", "team": "Team"},
                    )
                    st.plotly_chart(fig_sent, use_container_width=True)
                    st.dataframe(df_filtered[["team", "date", "avg_vader"]].reset_index(drop=True),
                                 use_container_width=True)
                else:
                    fig_line = px.line(
                        df_filtered.sort_values("date"),
                        x="date",
                        y="avg_vader",
                        color="team",
                        markers=True,
                        title="Average VADER Sentiment Over Time",
                        labels={"avg_vader": "Avg VADER Score", "date": "Date"},
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

            # Hype index chart
            if "hype_index" in df_sent.columns and not df_filtered.empty:
                df_sent["hype_index"] = pd.to_numeric(df_sent["hype_index"], errors="coerce")
                df_hype = df_filtered.groupby("team", as_index=False)["hype_index"].mean()
                df_hype = df_hype.dropna(subset=["hype_index"])
                if df_hype.empty or df_hype["hype_index"].sum() == 0:
                    st.info("Hype index data is not yet available.")
                else:
                    fig_hype = px.bar(
                        df_hype.sort_values("hype_index", ascending=False),
                        x="team",
                        y="hype_index",
                        title="Average Hype Index by Team",
                        labels={"hype_index": "Hype Index", "team": "Team"},
                        color="team",
                    )
                    st.plotly_chart(fig_hype, use_container_width=True)
            elif "hype_index" not in df_sent.columns:
                st.info("Hype index data is not yet available.")
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
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Matches", len(df_matches))

        if "year" in df_matches.columns:
            df_matches["year"] = pd.to_numeric(df_matches["year"], errors="coerce")
            col2.metric(
                "Year Range",
                f"{int(df_matches['year'].min())} – {int(df_matches['year'].max())}",
            )

        all_teams = sorted(set(df_matches.get("team_a", pd.Series()).dropna().tolist() +
                               df_matches.get("team_b", pd.Series()).dropna().tolist()))
        col3.metric("Unique Teams", len(all_teams))

        # Sidebar filters
        if "year" in df_matches.columns:
            year_min = int(df_matches["year"].min())
            year_max = int(df_matches["year"].max())
            if year_min == year_max:
                # Only one year of data — slider would crash with min == max
                st.sidebar.info(f"Only one tournament year available: {year_min}")
                year_range = (year_min, year_max)
            else:
                year_range = st.sidebar.slider(
                    "Year range (Historical)", year_min, year_max, (year_min, year_max)
                )
            df_matches = df_matches[df_matches["year"].between(year_range[0], year_range[1])]

        if all_teams:
            selected_team = st.sidebar.selectbox("Filter by team (Historical)", ["All teams"] + all_teams)
            if selected_team != "All teams":
                mask = (df_matches["team_a"] == selected_team) | (df_matches["team_b"] == selected_team)
                df_matches = df_matches[mask]

        st.subheader("Match Results")
        st.dataframe(df_matches[["year", "stage", "team_a", "score_a", "score_b", "team_b", "winner"]
                                  if all(c in df_matches.columns for c in ["stage", "winner"])
                                  else df_matches.columns.tolist()].head(100),
                     use_container_width=True)

        # Win rate chart for selected team
        if all_teams and "selected_team" in dir() and selected_team != "All teams" and not df_matches.empty:
            wins = (df_matches["winner"] == selected_team).sum() if "winner" in df_matches.columns else 0
            draws = (df_matches["winner"] == "Draw").sum() if "winner" in df_matches.columns else 0
            losses = len(df_matches) - wins - draws
            fig_wr = px.pie(
                values=[wins, draws, losses],
                names=["Wins", "Draws", "Losses"],
                title=f"{selected_team} — Win/Draw/Loss breakdown",
                color_discrete_sequence=["#28a745", "#ffc107", "#dc3545"],
            )
            st.plotly_chart(fig_wr, use_container_width=True)

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
        if metrics_json and metrics_json.get("status") == "trained":
            best_model = metrics_json.get("best_model", "—")
            st.subheader(f"Best model: {best_model}")

            # Key metrics row
            c1, c2, c3, c4 = st.columns(4)
            if "accuracy" in metrics_json:
                c1.metric("Accuracy", f"{metrics_json['accuracy']:.3f}")
            if "f1_macro" in metrics_json:
                c2.metric("F1 Macro", f"{metrics_json['f1_macro']:.3f}")
            if "train_rows" in metrics_json:
                c3.metric("Train rows", metrics_json["train_rows"])
            if "test_rows" in metrics_json:
                c4.metric("Test rows", metrics_json["test_rows"])

            if "run_at" in metrics_json:
                st.caption(f"Last trained: {metrics_json['run_at']}")

            # Confusion matrix heatmap
            cm_data = (
                metrics_json.get("models", {})
                .get(best_model, {})
                .get("confusion_matrix")
            )
            if cm_data:
                st.subheader("Confusion Matrix")
                labels = ["team_b wins", "Draw", "team_a wins"]
                fig_cm = px.imshow(
                    cm_data,
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title=f"Confusion Matrix — {best_model} (rows=actual, cols=predicted)",
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                    aspect="auto",
                )
                fig_cm.update_layout(height=400)
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix not available in metrics.json.")

            # Per-model comparison if multiple models were trained
            models_dict = metrics_json.get("models", {})
            if len(models_dict) > 1:
                st.subheader("Model Comparison")
                comparison = [
                    {"Model": name, "Accuracy": v.get("accuracy", 0), "F1 Macro": v.get("f1_macro", 0)}
                    for name, v in models_dict.items()
                ]
                fig_cmp = px.bar(
                    pd.DataFrame(comparison),
                    x="Model",
                    y=["Accuracy", "F1 Macro"],
                    barmode="group",
                    title="Accuracy vs F1 Macro by Model",
                    color_discrete_sequence=["#4c78a8", "#f58518"],
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

        elif metrics_json:
            st.warning(f"Model not yet trained. Reason: {metrics_json.get('reason', 'unknown')}")

        if not df_metrics.empty:
            st.subheader("Training History")
            st.dataframe(df_metrics, use_container_width=True)

            if {"accuracy", "f1_macro", "run_at"}.issubset(df_metrics.columns):
                fig_metrics = px.line(
                    df_metrics.sort_values("run_at"),
                    x="run_at",
                    y=["accuracy", "f1_macro"],
                    markers=True,
                    title="Model Metrics Over Time",
                    labels={"run_at": "Run At", "value": "Score", "variable": "Metric"},
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
