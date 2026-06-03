import sqlite3

DB = "/app/data/worldcup.db"

with sqlite3.connect(DB) as conn:
    conn.row_factory = sqlite3.Row

    # ── Query 1 ────────────────────────────────────────────────────
    print("=== Query 1: source x detected_team distribution ===")
    rows = conn.execute("""
        SELECT source, detected_team, COUNT(*) AS n
        FROM processed_posts
        GROUP BY source, detected_team
        ORDER BY source, n DESC
    """).fetchall()

    print(f"{'source':<22}  {'detected_team':<22}  {'n':>5}")
    print("-" * 54)
    for r in rows:
        src  = r["source"]        or "NULL"
        team = r["detected_team"] or "NULL"
        print(f"{src:<22}  {team:<22}  {r['n']:>5}")

    # ── Query 2 ────────────────────────────────────────────────────
    print()
    print("=== Query 2: Telegram comments + parent post preview ===")
    rows2 = conn.execute("""
        SELECT
          pp.post_id,
          pp.detected_team,
          substr(pp.cleaned_text, 1, 100) AS comment_preview,
          c.parent_post_id,
          substr(p.body, 1, 140)          AS parent_preview
        FROM processed_posts AS pp
        LEFT JOIN raw_telegram_comments AS c ON c.id = pp.post_id
        LEFT JOIN raw_telegram_posts    AS p ON p.id = c.parent_post_id
        WHERE pp.source = 'telegram_comment'
        ORDER BY pp.processed_at DESC
        LIMIT 20
    """).fetchall()

    if not rows2:
        print("  (no telegram_comment rows found)")
    else:
        for i, r in enumerate(rows2, 1):
            print(f"\n  [{i}]")
            print(f"  post_id        : {r['post_id']}")
            print(f"  detected_team  : {r['detected_team'] or 'NULL'}")
            print(f"  comment        : {r['comment_preview']}")
            print(f"  parent_post_id : {r['parent_post_id']}")
            print(f"  parent_body    : {r['parent_preview']}")
            print("  " + "-" * 60)
