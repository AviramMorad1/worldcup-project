# TASKS.md — Team Task Breakdown

## Prerequisites (כולם ביחד, יום ראשון)

לפני שמתחילים לחלק, כולם ביחד:

- [ ] אחד מכם יוצר repo ב-GitHub ומוסיף את השאר כ-collaborators
- [ ] מורידים את שני ה-CSV מ-Kaggle ומניחים ב-`datasets/` (לינקים ב-ARCHITECTURE.md)
- [ ] פותחים Docker Desktop — Reddit API credentials are **not** required (collection uses public RSS feeds)
- [ ] כל אחד מוריד Docker Desktop למחשב שלו
- [ ] מריצים `docker-compose up` ריק (לפני שיש קוד) כדי לוודא ש-Docker עובד

---

## אדם 1 — Data Pipeline (collector + preprocessor)

### שבוע 1

**משימה 1.1 — collector: מבנה + CSV loader**
פנה ל-Claude Code עם:
> "Implement the collector service. Start with football_loader.py and main.py.
> Load matches.csv and rankings.csv into SQLite on first run.
> Follow ARCHITECTURE.md for table schema and CLAUDE.md for coding rules."

בדוק שהקונטיינר עולה ומייצר `worldcup.db` עם הטבלאות.

**משימה 1.2 — collector: Reddit RSS collector**
פנה ל-Claude Code עם:
> "Add reddit_collector.py to the collector service.
> Collect posts from football subreddits via public RSS feeds (no PRAW/API auth).
> Store in raw_reddit_posts. Use INSERT OR IGNORE to avoid duplicates."

בדוק בידנית שפוסטים נכנסים ל-DB.

### שבוע 2

**משימה 1.3 — preprocessor: text cleaning**
פנה ל-Claude Code עם:
> "Implement the preprocessor service. Start with text_cleaner.py.
> Clean raw Reddit posts (lowercase, remove URLs, tokenize with NLTK).
> Only process posts not yet in processed_posts table."

**משימה 1.4 — preprocessor: sentiment**
פנה ל-Claude Code עם:
> "Add sentiment.py to the preprocessor service.
> Run VADER and TextBlob on cleaned posts.
> Compute team_sentiment_daily aggregations and trending_words table.
> Follow the hype index formula in CLAUDE.md."

**בדיקות שצריך לעשות לפני שמעביר לאדם 3:**
- [ ] `raw_matches` מכיל נתונים מ-1990 עד 2022
- [ ] `raw_reddit_posts` מכיל לפחות כמה עשרות פוסטים
- [ ] `processed_posts` מכיל vader_compound בין -1 ל-1
- [ ] `team_sentiment_daily` מכיל שורות עם hype_index

---

## אדם 2 — ML Model (trainer)

### שבוע 1

**משימה 2.1 — feature engineering**
פנה ל-Claude Code עם:
> "Implement features.py in the trainer service.
> Compute ELO ratings from raw_matches history and store in team_elo table.
> Build a feature matrix with: rank_diff, elo_diff, h2h_win_rate, h2h_goal_diff,
> stage_encoded, is_host. Follow ARCHITECTURE.md."

קרא על ELO rating חישוב — Claude Code יממש, אבל כדאי שתבין את הלוגיקה.

**משימה 2.2 — model training**
פנה ל-Claude Code עם:
> "Implement model.py in the trainer service.
> Train XGBClassifier and RandomForestClassifier on the feature matrix.
> Split by year: train 1990-2018, test 2022.
> Save model.pkl and metrics.json to /app/data/models/.
> Log confusion matrix and F1 scores."

### שבוע 2

**משימה 2.3 — 2026 predictions**
פנה ל-Claude Code עם:
> "Add 2026 group stage predictions to trainer/main.py.
> Hardcode the 48 group stage matchups from FIFA World Cup 2026.
> Run the trained model on each matchup and store results in match_predictions table
> with confidence scores."

חפש את ה-48 משחקי הבתים של מונדיאל 2026 ב-Google ותעביר ל-Claude Code ברשימה.

**משימה 2.4 — main.py + Dockerfile**
פנה ל-Claude Code עם:
> "Write main.py for the trainer service that runs training on startup
> and then once per week via schedule. Also write the Dockerfile."

**בדיקות שצריך לעשות לפני שמעביר לאדם 3:**
- [ ] `models/model.pkl` נוצר בהצלחה
- [ ] `models/metrics.json` מכיל accuracy ו-F1
- [ ] `match_predictions` מכיל 48 שורות עם tournament_year=2026
- [ ] accuracy על test set לפחות 50% (מעל random)

---

## אדם 3 — Dashboard + DevOps

### שבוע 1

**משימה 3.1 — docker-compose.yml**
פנה ל-Claude Code עם:
> "Write docker-compose.yml for the worldcup project.
> Four services: collector, preprocessor, trainer, dashboard.
> Single named volume worldcup_data mounted at /app/data in all services.
> Single bridge network worldcup_net.
> Dashboard exposes port 8501. Load .env file in all services.
> depends_on: collector→preprocessor→trainer→dashboard."

וודא שכולם יכולים לעשות `docker-compose up` ושה-DB נוצר.

**משימה 3.2 — .env.example + README**
כתוב בעצמך (לא צריך Claude Code):
```
# .env.example
COLLECTION_INTERVAL_HOURS=168
```
וכתוב `README.md` קצר עם הוראות הרצה.

### שבוע 2

**משימה 3.3 — dashboard: עמוד predictions + historical**
פנה ל-Claude Code עם:
> "Implement services/dashboard/app.py with Streamlit.
> Add two sections: Match Predictions (from match_predictions table) and
> Historical Stats (from raw_matches). Use Plotly for charts.
> Use st.cache_data(ttl=300). If tables are empty, show placeholder messages."

**משימה 3.4 — dashboard: עמוד sentiment**
פנה ל-Claude Code עם:
> "Add Sentiment Tracker and Trending Words sections to dashboard/app.py.
> Sentiment: line chart of avg_vader over time per team (filterable by team).
> Trending: bar chart of top 10 words per team from trending_words table.
> Add team filter in st.sidebar."

**משימה 3.5 — dashboard: model performance**
פנה ל-Claude Code עם:
> "Add Model Performance section to dashboard/app.py.
> Load metrics.json from /app/data/models/metrics.json.
> Show accuracy, F1 macro, confusion matrix heatmap.
> Show last training time."

**בדיקות שצריך לעשות:**
- [ ] `docker-compose up --build` עולה בלי errors
- [ ] localhost:8501 נפתח
- [ ] כל 5 sections מוצגים (גם אם חלקם placeholders)
- [ ] אין crash כשה-DB ריק

---

## שבוע 3 — אינטגרציה

כולם ביחד:

- [ ] מריצים את כל ה-pipeline מהתחלה על מכונה אחת
- [ ] collector → preprocessor → trainer → dashboard, לפי הסדר
- [ ] בודקים שהדאשבורד מציג נתונים אמיתיים
- [ ] מתקנים bugs שעולים

**תחומי אחריות לבאגים:**
- בעיות collector/preprocessor → אדם 1
- בעיות trainer/model → אדם 2
- בעיות dashboard/docker → אדם 3

---

## שבוע 4 — Wrap Up

- [ ] אדם 2: כותב סקשן ב-README על המודל — איך הוא עובד, מה ה-features, מה ה-accuracy
- [ ] אדם 1: כותב סקשן ב-README על ה-pipeline — מאיפה הנתונים, כמה פוסטים נאספו
- [ ] אדם 3: screenshots של הדאשבורד ל-README
- [ ] כולם: עוברים על הקוד ומוחקים קוד מת / TODO comments לפני submission

---

## טיפים לעבודה עם Claude Code

1. **תמיד ספק הקשר** — כש-Claude Code פותח, תן לו לקרוא ARCHITECTURE.md ו-CLAUDE.md לפני שאתה מבקש משהו
2. **משימה אחת בכל פעם** — אל תבקש "בנה את כל ה-collector", תבקש קובץ ספציפי
3. **בדוק לפני שממשיך** — הרץ את הקוד לאחר כל משימה לפני שעוברים להבא
4. **כש-Claude Code "שוכח"** — אם הוא סוטה מה-architecture, ציין: "See CLAUDE.md rule: [הכלל הרלוונטי]"
5. **error messages** — כשיש error, תעתיק את ה-full stack trace ל-Claude Code, לא רק את השורה האחרונה
