from datetime import date
from pathlib import Path
import re
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


URL = "https://inside.fifa.com/fifa-world-ranking/men"
OUTPUT_PATH = Path("datasets/rankings.csv")


def clean_points(value: str) -> float | None:
    if not value:
        return None

    value = value.strip().replace(",", "")
    match = re.search(r"\d+\.\d+", value)

    if not match:
        return None

    return float(match.group(0))


def clean_rank(value: str) -> int | None:
    if not value:
        return None

    match = re.search(r"^\s*(\d+)", value)

    if not match:
        return None

    return int(match.group(1))


def extract_team_name(row) -> str:
    """
    Extract team name from the team link inside the second table cell.
    This matches the XPath pattern:
    tr[x]/td[2]/div/div/a
    """
    try:
        team_link = row.find_element(By.CSS_SELECTOR, "td:nth-child(2) a")
        return " ".join(team_link.text.split())
    except Exception:
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(cells) >= 2:
            return " ".join(cells[1].text.split())
        return ""


def extract_points(row) -> float | None:
    """
    Extract points from the row.

    FIFA's row text can be inconsistent because it contains flags,
    hidden fields, and the More column. We therefore search all decimal
    numbers in the row and take the last decimal number, which is the
    points value in the visible ranking row.
    """
    row_text = row.text
    numbers = re.findall(r"\d+\.\d+", row_text.replace(",", ""))

    if not numbers:
        return None

    return float(numbers[-1])


def click_show_full_rankings(driver) -> None:
    """
    Click the Show full rankings button if it exists.
    """
    try:
        button = driver.find_element(
            By.XPATH,
            "//button[contains(., 'Show full rankings')]"
        )
    except Exception:
        try:
            button = driver.find_element(
                By.XPATH,
                "/html/body/div[1]/div[1]/div[2]/main/div[2]/div[2]/div[5]/button"
            )
        except Exception:
            print("Show full rankings button not found. Continuing with visible rows.")
            return

    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center'});",
        button,
    )
    time.sleep(1)
    driver.execute_script("arguments[0].click();", button)
    time.sleep(4)

    print("Clicked Show full rankings button")


def scrape_rankings() -> list[dict]:
    options = Options()
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(URL)
        time.sleep(6)

        click_show_full_rankings(driver)

        previous_count = 0

        for _ in range(80):
            rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            current_count = len(rows)

            driver.execute_script("window.scrollBy(0, 2500);")
            time.sleep(0.5)

            new_count = len(driver.find_elements(By.CSS_SELECTOR, "table tbody tr"))

            if new_count == previous_count and new_count > 10:
                break

            previous_count = new_count

        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        print(f"Found {len(rows)} ranking rows")

        today = date.today().isoformat()
        rankings = []

        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")

            if len(cells) < 4:
                continue

            rank = clean_rank(cells[0].text)
            country_full = extract_team_name(row)
            points = extract_points(row)

            if not rank or not country_full or points is None:
                continue

            rankings.append(
                {
                    "rank_date": today,
                    "country_full": country_full,
                    "rank": rank,
                    "total_points": points,
                }
            )

        return rankings

    finally:
        driver.quit()


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rankings = scrape_rankings()

    if not rankings:
        raise RuntimeError("No rankings were scraped. FIFA page structure may have changed.")

    df = pd.DataFrame(rankings)

    df = df.drop_duplicates(subset=["country_full", "rank_date"])
    df = df.sort_values("rank")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()