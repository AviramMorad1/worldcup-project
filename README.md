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

# 3. Place CSV datasets in datasets/  (see Datasets section below)
#    worldcup_historical.csv  — 1930–2014 World Cup matches
#    worldcup_2018.csv        — 2018 World Cup matches
#    matches.csv              — 2022 World Cup matches
#    rankings.csv             — FIFA World Rankings (historical)
#    current_rankings.csv     — (optional) current FIFA rankings for 2026 predictions

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

| Variable | Description | Default |
|---|---|---|
| `COLLECTION_INTERVAL_HOURS` | Hours between Reddit RSS collection cycles | 168 (7 days) |
| `PREPROCESS_INTERVAL_MINUTES` | Minutes between preprocessing cycles | 60 |
| `RECENCY_WEIGHTING_ENABLED` | Apply exponential recency weights during training | true |
| `RECENCY_DECAY_RATE` | Exponential decay rate for training sample weights | 0.08 |
| `MIN_RECENCY_WEIGHT` | Minimum weight floor for oldest tournament data | 0.35 |
| `CURRENT_RANKINGS_CSV` | Path to current FIFA rankings CSV | datasets/current_rankings.csv |

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

**Historical match data (recommended: all three for 1930–2022 coverage):**

- **`worldcup_historical.csv`** — FIFA World Cup matches 1930–2014
  → https://www.kaggle.com/datasets/abecklas/fifa-world-cup (WorldCupMatches.csv)

- **`worldcup_2018.csv`** — 2018 World Cup matches

- **`matches.csv`** — 2022 World Cup matches (same Kaggle source above)

Together these give the trainer 983 matches across 22 tournaments.
The loader handles both CSV formats automatically (flexible column mapping + team name normalization).

**Rankings:**

- **`rankings.csv`** — FIFA World Rankings 1992–2024
  → https://www.kaggle.com/datasets/cashncarry/fifaworldranking

- **`current_rankings.csv`** (optional, strongly recommended) — current FIFA rankings
  Columns: `team, rank, points`  |  Year will be stored as 2026 automatically.
  Providing this file significantly improves 2026 prediction quality.
  Without it, the model falls back to the most recent historical rankings.

**Team name normalization is automatic.** Common variants like `USA → United States`,
`South Korea → Korea Republic`, `Turkey → Türkiye` are handled in the loader.

---

## Dashboard Sections

- **Match Predictions** — 2026 group stage predictions with model confidence, confidence band (High/Medium/Low), and per-match explanations
- **Sentiment Tracker** — Reddit/Telegram sentiment per team over time
- **Trending Words** — Top words per team from Reddit/Telegram posts
- **Historical Stats** — Win rates and head-to-head records (1930–2022)
- **Model Performance** — Accuracy, F1, confusion matrix from the trained model

---

## Prediction Model — Honest Limitations

The match prediction model trains on all available World Cup data (1930–2022 = ~983 matches)
and predicts 2026 group stage outcomes using these features:

| Feature | Description |
|---|---|
| `rank_diff` | FIFA rank difference at tournament time (year-matched) |
| `elo_diff` | ELO rating difference computed from match history |
| `h2h_win_rate` | Head-to-head win rate in prior World Cup meetings |
| `h2h_goal_diff` | Average goal differential in prior meetings |
| `stage_encoded` | Group stage / Round of 16 / Quarter / Semi / Final |
| `is_host_a/b` | Whether a team is a host nation |
| `current_rank_diff` | Most-recent FIFA rank difference (modern strength proxy) |
| `current_points_diff` | Most-recent FIFA points difference |

**Training approach:**
- Chronological split: train 1930–2018, test 2022
- Recent tournaments receive higher sample weights (exponential decay, weight 1.00 → 0.35)
- Probability calibration via `CalibratedClassifierCV(method='sigmoid')` prevents overconfident scores

**What confidence scores mean:**
"Model Confidence" = the ML classifier's probability estimate, not a guaranteed win probability.
Real football has enormous variance. Even a 70% confidence prediction will be wrong 30% of the time.
The model also provides a simple ranking-baseline comparison — if the ML model does not beat
the baseline, the dashboard warns you.

**Current limitations:**
- No Telegram/Reddit sentiment in the prediction features (dashboard-only for now)
- Teams with no World Cup history get default ELO (1500) and fallback rankings
- Adding a `current_rankings.csv` significantly improves 2026 prediction quality

---

## Optional Squad / Player Strength Features

The project can optionally use current 2025/2026 club-season player stats to compute a
**squad strength score** (0–100) per national team, used as a conservative additional signal.

**How it works:**
1. `players_data_2025_2026.csv` (top 5 European leagues, ~2,839 players) is loaded into `raw_player_stats`
2. FBref nation codes (`us USA`, `fr FRA`, etc.) are mapped to national team names
3. A team score is computed from positional stats (attack/mid/defense/GK), playing time,
   league quality, and production (goals + assists)
4. Score is used to apply a **conservative probability adjustment** to 2026 predictions
   (max ±2% by default, configurable via `MAX_SQUAD_PROBA_ADJUSTMENT`)

**Important limitations:**
- Only covers players in the top 5 European leagues (some WC 2026 teams have 0 players there)
- Club season stats are a proxy — not actual national team performance
- Squad rosters may not reflect final WC selections
- Injury data is not available in this dataset
- Teams with low coverage (Curaçao, Cabo Verde, etc.) are not adjusted

**Squad data files:**
- `datasets/players_data_2025_2026.csv` — required for squad features
- `datasets/national_team_squads.csv` — optional manual squad roster (columns: team, player_name, position, age, club, league, is_key_player, is_injured)
- `FOOTBALL_DATA_API_TOKEN` — optional API token for football-data.org (squad endpoints require paid tier)

**To disable squad features:**
```
SQUAD_FEATURES_ENABLED=false
```
