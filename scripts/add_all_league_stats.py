#!/usr/bin/env python3
"""
add_all_league_stats.py
-----------------------
Generic script to integrate player stats from any FBref-format CSV into
wc_players_with_stats.csv.

Handles all 6 new league files at once:
  - Saudi Pro League
  - Argentine Primera División
  - Eredivisie (Netherlands)
  - South African Premiership
  - Brazilian Série A
  - Türkiye Süper Lig

Matching strategy (same as Portugal):
  1. Find WC squad players whose club fuzzy-matches a club in this league file.
  2. Match player names (exact → token-sorted → fuzzy) with birth year validation.
  3. Skip players already in wc_players_with_stats.csv.

Usage:
  python scripts/add_all_league_stats.py
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
# League files to process
# (path, league_label)
# ---------------------------------------------------------------------------
LEAGUE_FILES: list[tuple[Path, str]] = [
    (Path("datasets/saudi pro league stats .csv"),    "sa Saudi Pro League"),
    (Path("datasets/Argentine Primera División .csv"), "ar Argentine Primera División"),
    (Path("datasets/Eredivisie.csv"),                  "nl Eredivisie"),
    (Path("datasets/South African Premiership Stats.csv"), "za South African Premiership"),
    (Path("datasets/Brazilian Série A.csv"),           "br Brazilian Série A"),
    (Path("datasets/Türkiye Süper Lig.csv"),           "tr Türkiye Süper Lig"),
    (Path("datasets/English Championship.csv"),        "eng Championship"),
    (Path("datasets/Belgian Pro League Stats.csv"),   "be Belgian Pro League"),
    (Path("datasets/Persian Gulf Pro League Stats.csv"), "ir Persian Gulf Pro League"),
]

SQUADS_CSV = Path("datasets/squads/all_squads.csv")
OUTPUT_CSV = Path("datasets/squads/wc_players_with_stats.csv")

# Fuzzy thresholds
CLUB_SIM_THRESHOLD   = 0.72   # club name similarity for squad↔fbref
PLAYER_SIM_THRESHOLD = 0.78   # player name similarity

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_ = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_).lower().strip()


def _tokens_sorted(name: str) -> str:
    return " ".join(sorted(_norm(name).split()))


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _birth_year(s: str) -> int | None:
    m = re.search(r"\b(\d{4})\b", s or "")
    return int(m.group(1)) if m else None


def _year_ok(squad_dob: str, fbref_born: str) -> bool:
    sy = _birth_year(squad_dob)
    fy = _birth_year(fbref_born)
    if sy is None or fy is None:
        return True
    return abs(sy - fy) <= 1


# ---------------------------------------------------------------------------
# Parse any FBref-format CSV (identical layout across all leagues)
# Row 0 = group header  → skip
# Row 1 = column names  → parse by index
# Rows where col[0] == 'Rk' = repeated header → skip
# ---------------------------------------------------------------------------

def _parse_fbref_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    for i, cells in enumerate(all_rows):
        if i == 0:                          # group header row
            continue
        if not cells or cells[0].strip() in ("", "Rk"):
            continue
        if len(cells) < 19:
            continue

        # Clean minutes (may be "1,234" with comma)
        minutes_raw = cells[9].strip().replace(",", "").replace('"', "")

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
            "yellow_cards":   cells[17].strip(),
            "red_cards":      cells[18].strip(),
        })

    return rows


# ---------------------------------------------------------------------------
# Club matching: find the FBref club name that best matches a WC squad club
# ---------------------------------------------------------------------------

def _best_fbref_club(squad_club: str, fbref_clubs: set[str]) -> str | None:
    """
    Return the best-matching FBref club name for a WC squad club, or None.
    Uses fuzzy similarity + keyword shortcuts for common variants.
    """
    n_squad = _norm(squad_club)

    # Quick keyword shortcut map for known awkward variants
    shortcuts: list[tuple[str, str]] = [
        # keyword in squad club name → FBref club name (if unambiguous)
        ("al-hilal",        "Al-Hilal"),
        ("al hilal",        "Al-Hilal"),
        ("al-nassr",        "Al-Nassr"),
        ("al nassr",        "Al-Nassr"),
        ("al-ahli",         "Al-Ahli Saudi"),
        ("al-ittihad",      "Al-Ittihad Club"),
        ("al ittihad",      "Al-Ittihad Club"),
        ("al-qadsiah",      "Al-Qadsiah"),
        ("al-duhail",       "Al-Duhail"),
        ("al-shorta",       "Al-Shorta"),
        ("al-hussein",      "Al-Hussein"),
        ("al-sadd",         "Al-Sadd"),
        ("al-rayyan",       "Al-Rayyan"),
        ("psv eindhoven",   "PSV Eindhoven"),
        ("psv",             "PSV Eindhoven"),
        ("ajax",            "Ajax"),
        ("feyenoord",       "Feyenoord"),
        ("az alkmaar",      "AZ"),
        ("az ",             "AZ"),
        ("mamelodi",        "Mamelodi Sundowns"),
        ("orlando pirates", "Orlando Pirates"),
        ("kaizer chiefs",   "Kaizer Chiefs"),
        ("cape town city",  "Cape Town City"),
        ("flamengo",        "Flamengo"),
        ("palmeiras",       "Palmeiras"),
        ("atletico mineiro","Atlético Mineiro"),
        ("atlético mineiro","Atlético Mineiro"),
        ("fluminense",      "Fluminense"),
        ("corinthians",     "Corinthians"),
        ("boca juniors",    "Boca Juniors"),
        ("river plate",     "River Plate"),
        ("racing club",     "Racing Club"),
        ("independiente",   "Independiente"),
        ("galatasaray",     "Galatasaray"),
        ("fenerbahce",      "Fenerbahçe"),
        ("fenerbahçe",      "Fenerbahçe"),
        ("besiktas",        "Beşiktaş"),
        ("beşiktaş",        "Beşiktaş"),
        ("trabzonspor",     "Trabzonspor"),
        ("basaksehir",      "İstanbul Başakşehir"),
        ("başakşehir",      "İstanbul Başakşehir"),
        # English Championship
        ("norwich",         "Norwich City"),
        ("middlesbrough",   "Middlesbrough"),
        ("coventry",        "Coventry City"),
        ("sheffield united","Sheffield United"),
        ("sheffield utd",   "Sheffield United"),
        ("southampton",     "Southampton"),
        ("leicester",       "Leicester City"),
        ("watford",         "Watford"),
        ("hull city",       "Hull City"),
        ("swansea",         "Swansea City"),
        ("stoke city",      "Stoke City"),
        ("portsmouth",      "Portsmouth"),
        ("derby county",    "Derby County"),
        ("wrexham",         "Wrexham"),
        # Belgian Pro League
        ("club brugge",     "Club Brugge"),
        ("anderlecht",      "Anderlecht"),
        ("rsc anderlecht",  "Anderlecht"),
        ("genk",            "Genk"),
        ("union saint-gilloise", "Union SG"),
        ("union sg",        "Union SG"),
        ("standard liege",  "Standard Liège"),
        ("standard liège",  "Standard Liège"),
        ("royal antwerp",   "Antwerp"),
        ("antwerp",         "Antwerp"),
        ("charleroi",       "Charleroi"),
        ("dender",          "Dender"),
        ("cercle brugge",   "Cercle Brugge"),
        # Iranian (Persian Gulf Pro League)
        ("esteghlal",       "Esteghlal"),
        ("persepolis",      "Persepolis"),
        ("tractor",         "Tractor"),
        ("sepahan",         "Sepahan"),
        ("foolad",          "Foolad"),
        ("paykan",          "Paykan"),
        ("malavan",         "Malavan"),
    ]
    for keyword, target in shortcuts:
        if keyword in n_squad:
            if target in fbref_clubs:
                return target
            # Try case-insensitive lookup
            for fc in fbref_clubs:
                if _norm(fc) == _norm(target):
                    return fc

    # Fuzzy fallback
    best_club, best_score = None, 0.0
    for fc in fbref_clubs:
        score = _sim(squad_club, fc)
        if score > best_score:
            best_club, best_score = fc, score
    if best_score >= CLUB_SIM_THRESHOLD:
        return best_club
    return None


# ---------------------------------------------------------------------------
# Player matching within a club
# ---------------------------------------------------------------------------

def _find_player(squad_name: str, squad_dob: str, club_rows: list[dict],
                 all_rows: list[dict]) -> tuple[dict | None, str]:
    """
    Try to find an FBref row matching the WC squad player.
    Returns (row, method_label) or (None, '').
    """
    for scope, candidates in [("club", club_rows), ("all", all_rows)]:
        # Pass 1: exact normalized name
        for r in candidates:
            if _norm(r["player"]) == _norm(squad_name) and _year_ok(squad_dob, r["born"]):
                return r, f"exact_{scope}"

        # Pass 2: token-sorted
        tgt = _tokens_sorted(squad_name)
        for r in candidates:
            if _tokens_sorted(r["player"]) == tgt and _year_ok(squad_dob, r["born"]):
                return r, f"token_{scope}"

        # Pass 3: fuzzy
        best_row, best_score = None, 0.0
        for r in candidates:
            if not _year_ok(squad_dob, r["born"]):
                continue
            score = max(
                _sim(squad_name, r["player"]),
                _sim(_tokens_sorted(squad_name), _tokens_sorted(r["player"])),
            )
            if score > best_score:
                best_row, best_score = r, score
        if best_score >= PLAYER_SIM_THRESHOLD and best_row is not None:
            return best_row, f"fuzzy_{scope}({best_score:.2f})"

    return None, ""


# ---------------------------------------------------------------------------
# Build output row
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


def _build_row(squad: dict, pt: dict, league: str, method: str) -> dict:
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
        "stats_position": pt.get("position", squad["position"]),
        "stats_squad":    pt.get("squad", ""),
        "stats_league":   league,
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
    # Load current state once
    print(f"[INFO] Loading {OUTPUT_CSV}")
    with open(OUTPUT_CSV, encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))
    already_matched = {_norm(r["player_name"]) for r in existing_rows}
    print(f"       {len(existing_rows)} players already in stats file.")

    print(f"[INFO] Loading {SQUADS_CSV}")
    with open(SQUADS_CSV, encoding="utf-8") as f:
        squad_rows = list(csv.DictReader(f))
    print(f"       {len(squad_rows)} WC squad players.\n")

    all_new_rows: list[dict] = []
    league_summary: list[tuple[str, int, int]] = []  # (league, matched, unmatched)

    for csv_path, league_label in LEAGUE_FILES:
        if not csv_path.exists():
            print(f"[WARN] File not found: {csv_path} — skipping.")
            continue

        print(f"{'='*60}")
        print(f"Processing: {league_label}")
        print(f"File: {csv_path.name}")

        fbref_rows = _parse_fbref_csv(csv_path)
        fbref_clubs = {r["squad"] for r in fbref_rows}
        print(f"  Rows parsed: {len(fbref_rows)} — Clubs: {len(fbref_clubs)}")

        # Find WC players not yet matched whose club is in this league
        candidates: list[dict] = []
        for sq in squad_rows:
            if _norm(sq["player_name"]) in already_matched:
                continue
            matched_club = _best_fbref_club(sq["club"], fbref_clubs)
            if matched_club:
                candidates.append({**sq, "_fbref_club": matched_club})

        print(f"  WC candidates from this league: {len(candidates)}")
        if not candidates:
            print(f"  (no new players to add from this league)")
            league_summary.append((league_label, 0, 0))
            print()
            continue

        # Build club → rows index for faster lookup
        club_index: dict[str, list[dict]] = {}
        for r in fbref_rows:
            club_index.setdefault(r["squad"], []).append(r)

        new_rows: list[dict] = []
        unmatched: list[str] = []

        for sq in candidates:
            fbref_club = sq["_fbref_club"]
            club_rows  = club_index.get(fbref_club, [])
            row, method = _find_player(sq["player_name"], sq["date_of_birth"],
                                       club_rows, fbref_rows)
            if row:
                new_rows.append(_build_row(sq, row, league_label, method))
                # Mark as matched so later leagues don't double-add
                already_matched.add(_norm(sq["player_name"]))
                print(f"  [MATCH] {sq['player_name']:<28} <- '{row['player']}' ({row['squad']})  [{method}]")
            else:
                unmatched.append(f"{sq['player_name']} ({sq['team']}, {sq['club']})")

        print(f"\n  Matched: {len(new_rows)} | Unmatched: {len(unmatched)}")
        if unmatched:
            for u in unmatched:
                print(f"  [MISS]  {u}")

        all_new_rows.extend(new_rows)
        league_summary.append((league_label, len(new_rows), len(unmatched)))
        print()

    # -----------------------------------------------------------------------
    # Write everything
    # -----------------------------------------------------------------------
    if not all_new_rows:
        print("[INFO] No new players to add.")
        return

    all_rows = existing_rows + all_new_rows
    all_rows.sort(key=lambda r: (r.get("group", ""), r.get("team", ""), int(r.get("shirt_no") or 0)))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)
    total_new = sum(m for _, m, _ in league_summary)
    for lg, matched, unmatched in league_summary:
        print(f"  {lg:<40} +{matched:>3} matched  ({unmatched} unmatched)")
    print(f"\n  Previous total : {len(existing_rows)}")
    print(f"  New players    : +{total_new}")
    print(f"  New total      : {len(all_rows)}")
    print(f"\n  Written to     : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
