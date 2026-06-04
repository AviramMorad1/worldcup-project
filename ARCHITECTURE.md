# World Cup Analytics Platform — Architecture

## Project Overview

A microservices-based data pipeline and dashboard for World Cup analytics.
Four Docker containers communicate via a shared SQLite database on a Docker volume.
The system collects Reddit posts and historical match data, runs NLP sentiment analysis,
trains an ML match prediction model, and displays everything in a live Streamlit dashboard.

---

## Repository Structure

```
worldcup-project/
├── docker-compose.yml
├── ARCHITECTURE.md
├── CLAUDE.md
├── TASKS.md
├── .env.example
├── shared/                     # Shared utilities imported by multiple services
│   ├── __init__.py
│   └── db.py                   # SQLite connection helpers
├── data/                       # Docker volume mount point (gitignored)
│   └── worldcup.db             # SQLite database (auto-created)
├── services/
│   ├── collector/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py             # Entry point + scheduler
│   │   ├── reddit_collector.py # Reddit RSS collector (no API auth)
│   │   └── football_loader.py  # Historical CSV loader
│   ├── preprocessor/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py             # Entry point + scheduler
│   │   ├── text_cleaner.py     # Reddit post cleaning
│   │   └── sentiment.py        # VADER + TextBlob sentiment scoring
│   ├── trainer/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py             # Entry point + scheduler
│   │   ├── features.py         # Feature engineering
│   │   ├── model.py            # XGBoost training + evaluation
│   │   └── models/             # Saved .pkl files (gitignored)
│   └── dashboard/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app.py              # Streamlit app
├── datasets/
│   ├── matches.csv             # Historical World Cup matches (1990–2022)
│   └── rankings.csv            # FIFA rankings per year
└── tests/
    ├── test_collector.py
    ├── test_preprocessor.py
    └── test_trainer.py
```

---

## Services

### 1. collector

**Purpose:** Ingests raw data from two sources into SQLite.

**Runs:** On startup, then every `COLLECTION_INTERVAL_HOURS` (default 168 = 7 days).

**Data sources:**
- `reddit_collector.py` — Fetches public RSS feeds from football subreddits:
  `r/worldcup`, `r/soccer`, `r/FIFA`, `r/football`, `r/PremierLeague`,
  `r/ChampionsLeague`, `r/LaLiga`, `r/MLS`.
  Up to 25 posts per subreddit per cycle (~200 posts max). Uses `INSERT OR IGNORE`
  for deduplication. Comment RSS collection is disabled (downstream uses posts only).
- `football_loader.py` — One-time load of local CSV files from `datasets/` into SQLite on first run.
  Checks if data already loaded before re-importing.

**Output tables:**
- `raw_reddit_posts` (id, subreddit, title, body, author, score, created_utc, collected_at)
- `raw_reddit_comments` (schema retained for compatibility; not populated currently)
- `raw_matches` (id, year, stage, team_a, team_b, score_a, score_b, winner)
- `raw_rankings` (team, year, rank, points)

**Key libraries:** `requests`, `feedparser`, `sqlite3`

**Environment variables needed:**
- `COLLECTION_INTERVAL_HOURS` — hours between Reddit RSS collection cycles (default: 168)

**Coordination:** Writes `/app/data/collector_ready.flag` after each collection cycle so the
preprocessor can avoid starting before posts are inserted.

---

### 2. preprocessor

**Purpose:** Cleans raw data and enriches it with NLP features.

**Runs:** On startup (after waiting for collector readiness), then every
`PREPROCESS_INTERVAL_MINUTES` (default 60). On first start, polls for
`collector_ready.flag` or non-empty `raw_reddit_posts` (up to 30 min).
Each cycle processes only unprocessed posts, then recomputes aggregations.
Failed cycles are logged and the service continues on the next interval.

**Logic:**
- `text_cleaner.py` — Lowercase, remove URLs, remove special characters, tokenize.
  Uses `nltk` (punkt tokenizer, stopwords). Store cleaned text back to processed table.
- `sentiment.py` — Run VADER (`vaderSentiment`) and TextBlob on cleaned text.
  Store both scores. VADER is primary for Reddit slang; TextBlob is secondary.
- Aggregate per team per day: average sentiment, post volume, "hype index"
  (hype = normalized_volume × positive_sentiment_ratio).

**Output tables:**
- `processed_posts` (id, post_id, cleaned_text, tokens_json, vader_compound, vader_pos,
  vader_neg, vader_neu, textblob_polarity, textblob_subjectivity, processed_at)
- `team_sentiment_daily` (team, date, avg_vader, avg_textblob, post_count, hype_index)
- `trending_words` (team, date, word, frequency) — top 20 words per team per run

**Key libraries:** `nltk`, `vaderSentiment`, `textblob`

**Environment variables needed:**
- `PREPROCESS_INTERVAL_MINUTES` — minutes between preprocessing cycles (default: 60)

---

### 3. trainer

**Purpose:** Trains and saves a match outcome prediction model.

**Runs:** Once per week (after preprocessor finishes). Saves model artifact to shared volume.

**Features used (from `raw_matches` + `raw_rankings`):**
- `rank_a`, `rank_b` — FIFA ranking at time of tournament
- `rank_diff` — rank_a minus rank_b
- `elo_a`, `elo_b` — ELO rating (computed from match history, stored in `team_elo` table)
- `elo_diff`
- `h2h_win_rate_a` — historical win rate of team_a vs team_b
- `h2h_goal_diff` — average goal difference in past meetings
- `stage_encoded` — label encoded (group=0, r16=1, quarter=2, semi=3, final=4)
- `is_host_a`, `is_host_b` — boolean

**Target:** `outcome` — 0=team_b wins, 1=draw, 2=team_a wins

**Model pipeline:**
1. Load features from SQLite
2. Train/test split (by tournament year — last 2 tournaments as test set)
3. Train `XGBClassifier` with basic hyperparameters
4. Evaluate: accuracy, F1 per class, confusion matrix
5. Save model as `models/model.pkl` and metadata as `models/metrics.json`
6. Also train a simpler `RandomForestClassifier` for comparison

**Output tables:**
- `team_elo` (team, year, elo_rating) — computed and stored for feature reuse
- `model_metrics` (run_at, model_name, accuracy, f1_macro, notes)
- `match_predictions` — 2026 group stage predictions with baseline and adjusted probabilities,
  squad strength metadata, confidence bands, and explanations
- `raw_player_stats` — current-season club stats for WC players (collector; not used in historical training)
- `team_squad_strength` — per-team 2026 squad scores with coverage tiers and FIFA fallback blend
- `datasets/league_strengths.csv` — league quality weights from `datasets/global_national_league_rankings.csv` (global ranking) via `scripts/build_league_strengths.py`, with Wikipedia/static fallback

**2026 prediction flow (trainer):**
1. Baseline `predict_proba` from trained model (rank/ELO/H2H/host/current FIFA only).
2. `squad_features.compute_team_squad_strength()` from `raw_player_stats` (2026 adjustment only).
3. `apply_squad_adjustment()` conservatively shifts probabilities; stores base vs adjusted columns.

**Key libraries:** `scikit-learn`, `xgboost`, `pandas`, `joblib`

---

### 4. dashboard

**Purpose:** Streamlit app that reads from SQLite and displays analytics.

**Runs:** Continuously on port 8501. Streamlit's `st.rerun()` or `time.sleep` loop refreshes data.

**Pages / sections:**

1. **Match Predictions** — Table of 2026 predicted outcomes with confidence bars.
   Color-coded: green = high confidence, yellow = uncertain.

2. **Sentiment Tracker** — Line chart (Plotly) of `team_sentiment_daily.avg_vader`
   over time, filterable by team. Hype index bar chart.

3. **Trending Words** — Word frequency table per team. Top 10 words as styled table.
   Optional: simple bar chart of top words.

4. **Historical Stats** — Win rates, average goals, head-to-head records from `raw_matches`.
   Filterable by team and year range.

5. **Model Performance** — Confusion matrix heatmap, accuracy/F1 from `model_metrics`.
   Shows last training time.

**Key libraries:** `streamlit`, `plotly`, `pandas`, `sqlite3`

---

## Shared Infrastructure

### Docker Compose

All four services defined in `docker-compose.yml`.
All mount the same named volume `worldcup_data` at `/app/data`.
The SQLite file lives at `/app/data/worldcup.db` inside every container.

Services start in dependency order:
`collector` → `preprocessor` → `trainer` → `dashboard`
Use `depends_on` with `condition: service_started`.

Dashboard exposes port `8501:8501`.
No other ports exposed externally.

### Networking

All services on a single Docker bridge network `worldcup_net`.
Services reference each other by service name if needed (currently only via shared SQLite).

### Environment Variables

Defined in `.env` file (gitignored). `.env.example` committed to repo.
Loaded via `env_file: .env` in docker-compose.yml.

```
COLLECTION_INTERVAL_HOURS=168
PREPROCESS_INTERVAL_MINUTES=60
```

Reddit posts are collected via public RSS feeds. No Reddit API credentials are required.

---

## Data Flow

```
[Reddit RSS]          [datasets/*.csv]
      │                      │
      ▼                      ▼
┌─────────────┐       loads on first run
│  collector  │ ─────────────────────────▶ raw_reddit_posts
│             │                           raw_matches
└─────────────┘                           raw_rankings
                                               │
                                               ▼
                                       ┌──────────────┐
                                       │ preprocessor │ ─▶ processed_posts
                                       │              │ ─▶ team_sentiment_daily
                                       └──────────────┘ ─▶ trending_words
                                               │
                                               ▼
                                       ┌──────────────┐
                                       │   trainer    │ ─▶ team_elo
                                       │              │ ─▶ model_metrics
                                       │              │ ─▶ match_predictions
                                       └──────────────┘ ─▶ models/model.pkl
                                               │
                                               ▼
                                       ┌──────────────┐
                                       │  dashboard   │ reads all tables
                                       │  :8501       │ reads model.pkl
                                       └──────────────┘
```

---

## Database Schema (SQLite)

All tables created by the service that owns them, on first run.
Use `CREATE TABLE IF NOT EXISTS` everywhere.
No foreign key enforcement (SQLite default) — join on string IDs.

---

## Coding Conventions

- Python 3.11 in all containers
- All services structured as: `main.py` (entry + scheduler) + focused module files
- Logging via Python `logging` module, level INFO, format: `[SERVICE][LEVEL] message`
- No print statements — use logger
- Database access always via context manager (`with sqlite3.connect(...) as conn`)
- All SQL in dedicated functions, never inline in business logic
- Requirements pinned to exact versions in `requirements.txt`
- No shared code via pip — copy `shared/db.py` into each container's build context via Dockerfile COPY

---

## Datasets to Download

Before running the project, place these files in `datasets/`:

1. **matches.csv** — from Kaggle: "FIFA World Cup" dataset by abecklas
   URL: https://www.kaggle.com/datasets/abecklas/fifa-world-cup
   Columns needed: Year, Stage, Home Team Name, Away Team Name, Home Team Goals, Away Team Goals

2. **rankings.csv** — from Kaggle: "FIFA World Rankings 1992-2024"
   URL: https://www.kaggle.com/datasets/cashncarry/fifaworldranking
   Columns needed: rank_date, country_full, rank, total_points

Both are free with a Kaggle account.
