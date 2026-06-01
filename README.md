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

# 2. Copy and fill in your Reddit API credentials
cp .env.example .env
# Edit .env and add your REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET

# 3. Place CSV datasets in datasets/
#    matches.csv   — https://www.kaggle.com/datasets/abecklas/fifa-world-cup
#    rankings.csv  — https://www.kaggle.com/datasets/cashncarry/fifaworldranking

# 4. Build and run everything
docker-compose up --build

# 5. Open the dashboard
# http://localhost:8501
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
| `collector` | Loads CSV data + scrapes Reddit posts | On startup, then every 7 days |
| `preprocessor` | Cleans text, runs VADER/TextBlob sentiment | On startup, then every 7 days |
| `trainer` | Trains XGBoost/RandomForest model, generates 2026 predictions | On startup, then every 7 days |
| `dashboard` | Streamlit app — predictions, sentiment, historical stats | Continuous on port 8501 |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `REDDIT_CLIENT_ID` | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Reddit app client secret |
| `REDDIT_USER_AGENT` | User agent string (e.g. `worldcup-project/1.0`) |
| `COLLECTION_INTERVAL_HOURS` | How often to collect Reddit data (default: 168 = 7 days) |

Get Reddit credentials at: https://www.reddit.com/prefs/apps — choose "script" type.

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
