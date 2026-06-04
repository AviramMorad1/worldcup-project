#!/usr/bin/env python3
"""
add_portugal_stats.py
---------------------
Find WC squad players who appear in the 2025-2026 Primeira Liga stats CSV
and add them to wc_players_with_stats.csv.

Inputs:
  datasets/portugal league stats.csv   — FBref copy-pasted table (full names)
  datasets/squads/all_squads.csv       — 1246 WC squad players
  datasets/squads/wc_players_with_stats.csv — existing matched file

Output:
  datasets/squads/wc_players_with_stats.csv — updated in-place (new rows appended)

Matching strategy:
  FBref uses full player names, same as the squad file.
  1. Only consider WC players whose club maps to a Primeira Liga club.
  2. Match by normalized full name, with optional fuzzy fallback.
  3. Validate with birth year (±1 year) to avoid false positives.
  4. Only add players NOT already in wc_players_with_stats.csv.
"""

import csv
import difflib
import io
import re
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PT_CSV     = Path("datasets/portugal league stats.csv")
SQUADS_CSV = Path("datasets/squads/all_squads.csv")
OUTPUT_CSV = Path("datasets/squads/wc_players_with_stats.csv")

# Fuzzy match threshold (0-1). 0.82 = fairly strict.
SIMILARITY_THRESHOLD = 0.78

# ---------------------------------------------------------------------------
# Club name mapping: WC squad club name  →  Primeira Liga FBref club name(s)
# (key = normalized squad club name fragment)
# ---------------------------------------------------------------------------
PT_CLUB_MAP: list[tuple[str, str]] = [
    # WC squad club fragment (lower)    →  FBref squad name
    ("benfica",              "Benfica"),
    ("porto",                "Porto"),
    ("sporting cp",          "Sporting CP"),
    ("sporting de braga",    "Braga"),
    ("sc braga",             "Braga"),
    ("braga",                "Braga"),
    ("moreirense",           "Moreirense"),
    ("famalicao",            "Famalicão"),
    ("famalicão",            "Famalicão"),
    ("vitoria guimaraes",    "Vitória Guimarães"),
    ("vitória guimarães",    "Vitória Guimarães"),
    ("vitoria de guimaraes", "Vitória Guimarães"),
    ("vitória de guimarães", "Vitória Guimarães"),
    ("santa clara",          "Santa Clara"),
    ("gil vicente",          "Gil Vicente FC"),
    ("rio ave",              "Rio Ave"),
    ("arouca",               "Arouca"),
    ("casa pia",             "Casa Pia"),
    ("estoril",              "Estoril"),
    ("estrela amadora",      "Estrela"),
    ("estrela",              "Estrela"),
    # "nacional" must NOT match "Internacional" or "Atletico Nacional"
    # Use exact-word check handled in _fbref_club below
    ("cd nacional",          "Nacional"),
    ("club desportivo nacional", "Nacional"),
    ("avs futebol",          "AVS Futebol"),
    ("avs",                  "AVS Futebol"),
    ("alverca",              "Alverca"),
    ("tondela",              "Tondela"),
    ("boavista",             "Boavista"),
]

# Exact-word matches to avoid false positives ("Internacional" ≠ "Nacional")
_PT_EXACT_WORD: dict[str, str] = {
    "nacional":   "Nacional",   # only when the whole club name IS "Nacional"
    "braga":      "Braga",      # only when NOT "bragantino"
}

# All valid FBref Primeira Liga club names
_PT_FBREF_CLUBS = {v for _, v in PT_CLUB_MAP}


def _fbref_club(squad_club: str) -> str:
    """Return FBref Primeira Liga club name for a WC squad club, or '' if not PL."""
    n = _norm(squad_club)

    # Exact-word guards: "Nacional" only when the whole word stands alone,
    # "Braga" only when not part of "Bragantino"
    words = set(re.split(r"[\s\-]+", n))
    if "nacional" in words and "internacional" not in n and "atletico" not in n:
        return "Nacional"
    if "braga" in words and "bragantino" not in n:
        return "Braga"

    # Fragment-based lookup for all other clubs
    for fragment, fbref_name in PT_CLUB_MAP:
        if fragment in n:
            return fbref_name
    return ""


# ---------------------------------------------------------------------------
# Name utilities
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).lower().strip()


def _tokens_sorted(name: str) -> str:
    """Return space-joined sorted tokens for order-insensitive comparison."""
    return " ".join(sorted(_norm(name).split()))


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _birth_year(dob: str) -> int | None:
    """Extract year from date like '2000-01-29' or just '2000'."""
    if not dob:
        return None
    m = re.search(r"\b(\d{4})\b", dob)
    return int(m.group(1)) if m else None


def _year_ok(squad_dob: str, fbref_born: str) -> bool:
    """Return True if birth years match within ±1 year (or if either is unknown)."""
    sy = _birth_year(squad_dob)
    fy = _birth_year(fbref_born)
    if sy is None or fy is None:
        return True
    return abs(sy - fy) <= 1


# ---------------------------------------------------------------------------
# Parse Primeira Liga CSV
# ---------------------------------------------------------------------------
# FBref CSV column indices (0-based) when exported via copy-paste:
#  0  Rk
#  1  Player
#  2  Nation
#  3  Pos
#  4  Squad
#  5  Age
#  6  Born
#  7  MP        (matches played)
#  8  Starts
#  9  Min       (minutes, may have comma: "1,234")
# 10  90s
# 11  Gls       (goals — absolute)
# 12  Ast       (assists — absolute)
# 13  G+A       (absolute)
# 14  G-PK      (non-PK goals)
# 15  PK
# 16  PKatt
# 17  CrdY      (yellow cards)
# 18  CrdR      (red cards)
# 19–24: per-90 stats (ignored here)

def _parse_pt_csv(path: Path) -> list[dict]:
    """
    Read the Primeira Liga CSV (FBref copy-paste).
    Row 0 = group headers → skip.
    Row 1 = column names  → use as reference but parse by index.
    Repeated header rows (first cell == 'Rk') → skip.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    for i, cells in enumerate(all_rows):
        # Skip first row (group header) and rows that repeat the column header
        if i == 0:
            continue
        if not cells or cells[0].strip() in ("", "Rk"):
            continue

        # Need at least 19 columns
        if len(cells) < 19:
            continue

        # Clean minutes: "1,234" → "1234"
        minutes_raw = cells[9].strip().replace(",", "").replace('"', '')

        rows.append({
            "player":         cells[1].strip(),
            "nation":         cells[2].strip(),
            "position":       cells[3].strip(),
            "squad":          cells[4].strip(),
            "age":            cells[5].strip(),
            "born":           cells[6].strip(),
            "matches_played": cells[7].strip(),
            "starts":         cells[8].strip(),
            "minutes":        minutes_raw,
            "90s":            cells[10].strip(),
            "goals":          cells[11].strip(),
            "assists":        cells[12].strip(),
            "G+A":            cells[13].strip(),
            "goals_non_pk":   cells[14].strip(),
            "yellow_cards":   cells[17].strip(),
            "red_cards":      cells[18].strip(),
        })

    return rows


# ---------------------------------------------------------------------------
# Find matching FBref row
# ---------------------------------------------------------------------------

def _find_match(squad_name: str, squad_dob: str, fbref_club: str,
                pt_rows: list[dict]) -> tuple[dict | None, str]:
    """
    Search pt_rows for the best match for a WC squad player.
    Returns (matched_row, method) or (None, '').
    """
    # Restrict to same club first
    club_rows = [r for r in pt_rows if r["squad"] == fbref_club]
    search_sets = [("club", club_rows), ("all", pt_rows)]

    for scope, candidates in search_sets:
        # Pass 1: exact normalized name match
        for r in candidates:
            if _norm(r["player"]) == _norm(squad_name) and _year_ok(squad_dob, r["born"]):
                return r, f"pt_exact_{scope}"

        # Pass 2: token-sorted match
        target_sorted = _tokens_sorted(squad_name)
        for r in candidates:
            if _tokens_sorted(r["player"]) == target_sorted and _year_ok(squad_dob, r["born"]):
                return r, f"pt_token_{scope}"

        # Pass 3: fuzzy similarity
        best_row, best_score = None, 0.0
        for r in candidates:
            if not _year_ok(squad_dob, r["born"]):
                continue
            score = max(
                _similarity(squad_name, r["player"]),
                _similarity(_tokens_sorted(squad_name), _tokens_sorted(r["player"])),
            )
            if score > best_score:
                best_row, best_score = r, score
        if best_score >= SIMILARITY_THRESHOLD and best_row is not None:
            return best_row, f"pt_fuzzy_{scope}({best_score:.2f})"

    return None, ""


# ---------------------------------------------------------------------------
# Output schema (same as wc_players_with_stats.csv)
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "team", "group", "shirt_no", "position_wc", "player_name",
    "date_of_birth", "age_wc", "caps", "goals_intl", "club_wc", "is_captain",
    "stats_position", "stats_squad", "stats_league",
    "matches_played", "starts", "minutes", "90s",
    "goals_stats", "assists", "G+A",
    "shots", "shots_on_target",
    "yellow_cards", "red_cards",
    "saves", "clean_sheets",
    "tackles_won", "interceptions", "crosses", "plus_minus",
    "match_method",
]


def _build_row(squad: dict, pt: dict, method: str) -> dict:
    """Merge WC squad info + Primeira Liga stats into one output row."""
    try:
        ga = int(pt.get("goals", 0) or 0) + int(pt.get("assists", 0) or 0)
    except (ValueError, TypeError):
        ga = pt.get("G+A", "")

    return {
        "team":           squad["team"],
        "group":          squad["group"],
        "shirt_no":       squad["shirt_no"],
        "position_wc":    squad["position"],
        "player_name":    squad["player_name"],
        "date_of_birth":  squad["date_of_birth"],
        "age_wc":         squad["age"],
        "caps":           squad["caps"],
        "goals_intl":     squad["goals"],
        "club_wc":        squad["club"],
        "is_captain":     squad["is_captain"],
        # Stats from Primeira Liga
        "stats_position": pt.get("position", squad["position"]),
        "stats_squad":    pt.get("squad", ""),
        "stats_league":   "pt Primeira Liga",
        "matches_played": pt.get("matches_played", ""),
        "starts":         pt.get("starts", ""),
        "minutes":        pt.get("minutes", ""),
        "90s":            pt.get("90s", ""),
        "goals_stats":    pt.get("goals", ""),
        "assists":        pt.get("assists", ""),
        "G+A":            ga,
        "shots":          "",
        "shots_on_target":"",
        "yellow_cards":   pt.get("yellow_cards", ""),
        "red_cards":      pt.get("red_cards", ""),
        "saves":          "",
        "clean_sheets":   "",
        "tackles_won":    "",
        "interceptions":  "",
        "crosses":        "",
        "plus_minus":     "",
        "match_method":   method,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load existing matched file
    print(f"[INFO] Loading {OUTPUT_CSV}")
    with open(OUTPUT_CSV, encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))
    already_matched = {_norm(r["player_name"]) for r in existing_rows}
    print(f"       {len(existing_rows)} players already in stats file.")

    # Load WC squad
    print(f"[INFO] Loading {SQUADS_CSV}")
    with open(SQUADS_CSV, encoding="utf-8") as f:
        squad_rows = list(csv.DictReader(f))
    print(f"       {len(squad_rows)} WC squad players.")

    # Load Primeira Liga FBref stats
    print(f"[INFO] Loading {PT_CSV}")
    pt_rows = _parse_pt_csv(PT_CSV)
    print(f"       {len(pt_rows)} Primeira Liga player rows parsed.")
    pt_clubs = {r["squad"] for r in pt_rows}
    print(f"       Clubs found: {sorted(pt_clubs)}")

    # Index WC players who are:
    #   a) NOT already matched
    #   b) Playing at a Primeira Liga club
    unmatched_pt_squad: list[dict] = []
    for sq in squad_rows:
        if _norm(sq["player_name"]) in already_matched:
            continue
        fbref_club = _fbref_club(sq["club"])
        if fbref_club:
            unmatched_pt_squad.append({**sq, "_fbref_club": fbref_club})

    print(f"\n[INFO] WC players at Primeira Liga clubs not yet in stats file: {len(unmatched_pt_squad)}")
    for sq in unmatched_pt_squad:
        print(f"       {sq['player_name']:<30} {sq['club']:<30} ({sq['team']})  -> FBref club: {sq['_fbref_club']}")

    # Match each unmatched PL-club WC player
    new_rows: list[dict] = []
    still_unmatched: list[str] = []

    for sq in unmatched_pt_squad:
        row, method = _find_match(
            squad_name = sq["player_name"],
            squad_dob  = sq["date_of_birth"],
            fbref_club = sq["_fbref_club"],
            pt_rows    = pt_rows,
        )
        if row:
            new_rows.append(_build_row(sq, row, method))
            print(f"[MATCH] {sq['player_name']:<30} <- FBref '{row['player']}' ({row['squad']})  [{method}]")
        else:
            still_unmatched.append(f"{sq['player_name']} ({sq['team']}, {sq['club']})")

    print(f"\n[RESULT] New Primeira Liga players matched: {len(new_rows)}")
    print(f"[RESULT] Still unmatched: {len(still_unmatched)}")
    for s in still_unmatched:
        print(f"  - {s}")

    if not new_rows:
        print("[INFO] Nothing to add.")
        return

    # Append new rows and rewrite sorted
    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda r: (r.get("group", ""), r.get("team", ""), int(r.get("shirt_no") or 0)))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[DONE] {OUTPUT_CSV} updated: {len(existing_rows)} + {len(new_rows)} = {len(all_rows)} players total.")


if __name__ == "__main__":
    main()
