# World Cup 2026 Analytics Platform

A 4-service Docker microservices pipeline for World Cup analytics.

```
collector → preprocessor → trainer → dashboard
```

All services share one SQLite database on a Docker named volume.

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd worldcup-project

# 2. Copy environment config (optional — defaults work for RSS collection)
cp .env.example .env

# 3. Place CSV datasets in datasets/
#    matches.csv   — https://www.kaggle.com/datasets/abecklas/fifa-world-cup
#    rankings.csv  — https://www.kaggle.com/datasets/cashncarry/fifaworldranking

# 4. Build and run everything
docker-compose up --build

# 5. Open the dashboard
# http://localhost:8501

# 6. (Optional) Verify the pipeline is healthy
docker compose run --rm preprocessor python /scripts/integration_check.py
```

To run a single service:
```bash
docker-compose up --build collector
```

To rebuild after code changes:
```bash
docker-compose up --build --force-recreate <service_name>
```

---

## Services

| Service | Description | Schedule |
|---|---|---|
| `collector` | Loads CSV data + collects Reddit posts via RSS | On startup, then every `COLLECTION_INTERVAL_HOURS` (default 7 days) |
| `preprocessor` | Cleans text, runs VADER/TextBlob sentiment | On startup, then every `PREPROCESS_INTERVAL_MINUTES` (default 60) |
| `trainer` | Trains XGBoost/RandomForest model, generates 2026 predictions | On startup, then every 7 days |
| `dashboard` | Streamlit app — predictions, sentiment, historical stats | Continuous on port 8501 |

---

## Environment Variables

Copy `.env.example` to `.env` if you want to override defaults:

| Variable | Description |
|---|---|
| `COLLECTION_INTERVAL_HOURS` | Hours between Reddit RSS collection cycles (default: 168 = 7 days) |
| `PREPROCESS_INTERVAL_MINUTES` | Minutes between preprocessing cycles (default: 60) |

Reddit data is collected from public RSS feeds. No Reddit API app or credentials are required.

---

## Integration validation

`scripts/integration_check.py` is a lightweight health check for the full pipeline. Run it after `docker compose up` (or after a collection/preprocessing/training cycle) to confirm the database, sentiment data, model artifacts, and 2026 predictions look correct.

### What it checks

| Check | Critical? | Description |
|---|---|---|
| Database file | Yes | Opens `worldcup.db` from env or common paths |
| Core tables | Yes | `raw_matches`, `raw_rankings`, `raw_reddit_posts`, `processed_posts`, `team_sentiment_daily`, `trending_words`, `match_predictions`, `model_metrics` |
| Row counts | Yes | All critical tables must contain data |
| Low volume | Warn | Fewer than 50 Reddit/processed posts |
| Sentiment ranges | Yes | `vader_compound` and `textblob_polarity` in `[-1, 1]` |
| Reddit timestamps | Yes | `created_utc` populated; multiple publish dates when post count > 50 |
| 2026 predictions | Yes | Rows in `match_predictions` for `tournament_year = 2026` (expects 72) |
| Model artifacts | Yes | `model.pkl` and `metrics.json` with `accuracy` + `f1_macro` |
| Model quality | Warn | Accuracy or F1 below 50% |
| Dashboard source | Warn | `services/dashboard/app.py` exists |

Exit codes:
- **0** — all critical checks passed (warnings may still appear)
- **1** — one or more critical checks failed

### How to run

**Recommended (Docker)** — uses the shared `worldcup_data` volume after the stack has run:

```bash
docker compose run --rm preprocessor python /scripts/integration_check.py
```

**Local** — only works if the database file exists on your machine (e.g. copied from the Docker volume):

```bash
python scripts/integration_check.py
```

**With custom paths** (optional):

```bash
DATABASE_PATH=/app/data/worldcup.db MODEL_DIR=/app/data/models python scripts/integration_check.py
```

Path resolution (if env vars are not set):

| Resource | Tried in order |
|---|---|
| Database | `DATABASE_PATH` → `/app/data/worldcup.db` → `data/worldcup.db` → `./worldcup.db` |
| Models | `MODEL_DIR` → `/app/data/models` → `data/models` → `./models` |

### Example output

```
INTEGRATION CHECK REPORT
========================

[PASS] Database found: /app/data/worldcup.db
[PASS] raw_reddit_posts rows: 300
[PASS] vader_compound range: -0.9976 to 0.9993
[PASS] Reddit publish dates: 13 distinct days, 2026-02-19 to 2026-06-02
[PASS] match_predictions 2026 rows: 72
[PASS] model.pkl found: /app/data/models/model.pkl
[PASS] metrics.json found: accuracy=0.538, f1_macro=0.581

Summary:
  Critical failures: 0
  Warnings: 0

Result: PASS
```

If you see `Result: FAIL`, inspect the `[FAIL]` lines in the report and check the collector, preprocessor, or trainer logs.

---

## Datasets

Place these files in the `datasets/` folder before running:

- **`matches.csv`** — FIFA World Cup match results (1930–2022)
  → https://www.kaggle.com/datasets/abecklas/fifa-world-cup

- **`rankings.csv`** — FIFA World Rankings 1992–2024
  → https://www.kaggle.com/datasets/cashncarry/fifaworldranking

---

## Dashboard Sections

- **Match Predictions** — 2026 group stage predictions with confidence scores
- **Sentiment Tracker** — Reddit sentiment per team over time
- **Trending Words** — Top words per team from Reddit posts
- **Historical Stats** — Win rates and head-to-head records (1930–2022)
- **Model Performance** — Accuracy, F1, confusion matrix from the trained model
