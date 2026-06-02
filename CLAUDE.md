# CLAUDE.md — Instructions for Claude Code

This file tells Claude Code how to work on this project.
Read ARCHITECTURE.md first for the full technical picture.

---

## What This Project Is

A 4-service Docker microservices pipeline for World Cup analytics:
collector → preprocessor → trainer → dashboard
All services share one SQLite database on a Docker named volume.
See ARCHITECTURE.md for full details.

---

## How to Run the Project

```bash
# First time setup
cp .env.example .env
# Fill in Reddit API credentials in .env

# Place CSV files in datasets/ (see ARCHITECTURE.md > Datasets to Download)

# Build and run everything
docker-compose up --build

# Dashboard is at http://localhost:8501
```

To run a single service for development:
```bash
docker-compose up --build collector
```

To rebuild after code changes:
```bash
docker-compose up --build --force-recreate <service_name>
```

---

## Rules Claude Code Must Follow

### General
- Never modify files outside the service you're working on without being asked
- Always use `CREATE TABLE IF NOT EXISTS` — never drop existing tables
- All database paths hardcoded as `/app/data/worldcup.db` — never relative paths
- Log with `logging` module only — no `print()` statements
- When a task says "implement X", write the full working code, not stubs or TODOs

### Python
- Python 3.11
- All dependencies must be in `requirements.txt` with pinned versions (e.g. `pandas==2.2.1`)
- Use `venv` locally or Docker — never install globally

### Docker
- Base image for all services: `python:3.11-slim`
- Working directory inside containers: `/app`
- SQLite data volume mounted at: `/app/data`
- Keep Dockerfiles minimal — no unnecessary tools

### SQLite
- Connection pattern: always `with sqlite3.connect(DB_PATH) as conn:`
- All SQL in dedicated functions with descriptive names
- Use parameterized queries — never f-strings in SQL
- Each service creates its own tables on startup

### Streamlit (dashboard only)
- Single file: `services/dashboard/app.py`
- Use `st.cache_data(ttl=300)` on all data-loading functions (5-min cache)
- Use Plotly for all charts — not matplotlib
- Use `st.sidebar` for filters

---

## Service-Specific Notes

### collector
- Check for existing rows before inserting (use `INSERT OR IGNORE`)
- Reddit post `id` is the unique key — use it to avoid duplicates
- Reddit data comes from public RSS feeds (no PRAW/API credentials)
- Comment RSS collection is disabled; only posts are stored
- Football CSV loader must check if `raw_matches` already has rows before loading
- The scheduler should log "Starting collection run" at the beginning of each cycle
- Write `/app/data/collector_ready.flag` after each collection cycle completes

### preprocessor
- Wait for collector readiness (`collector_ready.flag` or existing `raw_reddit_posts`) before the first cycle
- Run preprocessing immediately after readiness, then every `PREPROCESS_INTERVAL_MINUTES` (default 60)
- Only process rows from `raw_reddit_posts` that don't have a corresponding row in `processed_posts`
- Log and continue on cycle failure — do not crash the container
- Always download NLTK data at startup: punkt, stopwords, vader_lexicon
- VADER compound score range: -1.0 to 1.0. Store as REAL in SQLite.
- Hype index formula: `(post_count / max_post_count_that_day) * max(0, avg_vader_compound)`

### trainer
- Split data by year: train on 1990–2018, test on 2022
- Save model to `/app/data/models/model.pkl` (create dir if not exists)
- Save metrics to `/app/data/models/metrics.json`
- Print confusion matrix to logs after training
- For 2026 predictions: hardcode the 48 group stage matchups (see FIFA official site)
  Store in `match_predictions` table with `tournament_year=2026`

### dashboard
- Read model metrics from `/app/data/models/metrics.json` if it exists
- If model not yet trained, show a placeholder message — do not crash
- If no Reddit data yet, show placeholder message — do not crash
- Refresh data every 5 minutes using `st.cache_data(ttl=300)`

---

## What NOT to Do

- Do not add Redis, PostgreSQL, MongoDB, or any external database
- Do not add authentication or login screens
- Do not add Docker networking beyond the single bridge network in docker-compose.yml
- Do not use async/await — keep everything synchronous for simplicity
- Do not add unit tests unless explicitly asked
- Do not use environment variables that aren't in `.env.example`
- Do not create Jupyter notebooks

---

## When Asked to "Implement a Service"

Do this in order:
1. Write `Dockerfile`
2. Write `requirements.txt`
3. Write the module files (e.g. `reddit_collector.py`, `text_cleaner.py`)
4. Write `main.py` last (it ties everything together)
5. Check that table creation SQL matches the schema in ARCHITECTURE.md exactly

---

## File Naming

Exact filenames matter — other services may reference them.
Do not rename files from what's defined in ARCHITECTURE.md.
