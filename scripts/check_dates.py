import sqlite3
from datetime import datetime, timezone

conn = sqlite3.connect("/app/data/worldcup.db")
print("posts", conn.execute("SELECT COUNT(*) FROM raw_reddit_posts").fetchone()[0])
rows = conn.execute(
    "SELECT created_utc, substr(title, 1, 50) FROM raw_reddit_posts LIMIT 8"
).fetchall()
for ts, title in rows:
    print(datetime.fromtimestamp(ts, tz=timezone.utc).date(), "|", title)

print(
    "distinct post publish dates:",
    conn.execute(
        "SELECT COUNT(DISTINCT date(created_utc, 'unixepoch')) FROM raw_reddit_posts"
    ).fetchone()[0],
)
print(
    "sentiment table dates:",
    conn.execute(
        "SELECT DISTINCT date FROM team_sentiment_daily ORDER BY date"
    ).fetchall(),
)
conn.close()
