import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
from time import sleep
import random
from io import StringIO

def get_match_report_links(schedule_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    session = requests.Session()
    for attempt in range(10):
        try:
            response = session.get(schedule_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            report_links = set()
            rows = soup.select("table.stats_table tbody tr")
            for row in rows:
                match_report = row.find("td", {"data-stat": "match_report"})
                if match_report:
                    a = match_report.find("a", href=True)
                    if a and "href" in a.attrs:
                        href = a["href"]
                        home_team_td = row.find("td", {"data-stat": "home_team"})
                        away_team_td = row.find("td", {"data-stat": "away_team"})
                        home_team = home_team_td.text.strip() if home_team_td else None
                        away_team = away_team_td.text.strip() if away_team_td else None
                        full_url = f"https://fbref.com{href}"
                        report_links.add((full_url, home_team, away_team))
            return list(report_links)
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = 120 + random.randint(30, 60)
                print(f"Rate limit hit. Waiting {wait_time} seconds...")
                sleep(wait_time)
            else:
                raise e
    raise Exception("Unable to fetch schedule after multiple retries.")

def get_match_stats(report_url, session, home_team, away_team):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    for attempt in range(5):
        try:
            response = session.get(report_url, headers=headers)
            response.raise_for_status()
            html = response.text

            soup = BeautifulSoup(html, "lxml")
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                if "<table" in comment:
                    html += comment

            tables = pd.read_html(StringIO(html))
            team_stats = next(
                (tbl for tbl in tables if any("Possession" in str(col) for col in tbl.columns)), None
            )
            if team_stats is None:
                return {"url": report_url, "error": "No team stats found",
                        "home_team": home_team, "away_team": away_team}
            team_stats.columns = [tuple(map(str.strip, col)) for col in team_stats.columns]
            def get_stat(stat_name):
                cols = [col for col in team_stats.columns if stat_name in col[1]]
                return team_stats[cols[0]].iloc[0], team_stats[cols[1]].iloc[0] if len(cols) >= 2 else (None, None)
            poss_home, poss_away = get_stat("Possession")
            xg_home = soup.select_one(".score_xg").text.strip() if soup.select_one(".score_xg") else None
            xg_away = soup.select(".score_xg")[1].text.strip() if len(soup.select(".score_xg")) > 1 else None

            return {
                "url": report_url,
                "home_team": home_team,
                "away_team": away_team,
                "possession_home": poss_home,
                "possession_away": poss_away,
                "xg_home": xg_home,
                "xg_away": xg_away,
            }
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = 120 + random.randint(30, 60)
                print(f"Rate limit hit for {report_url}. Waiting {wait_time} seconds...")
                sleep(wait_time)
            else:
                raise e
    return {"url": report_url, "error": "Unable to fetch stats",
            "home_team": home_team, "away_team": away_team}

def process_matches_multiple_seasons(schedule_urls):
    session = requests.Session()
    for schedule_url, output_csv in schedule_urls.items():
        print(f"Processing schedule: {schedule_url} -> {output_csv}")
        report_links = get_match_report_links(schedule_url)
        print(f"Found {len(report_links)} report links.")

        all_stats = []
        for link, home_team, away_team in report_links:
            try:
                stats = get_match_stats(link, session, home_team, away_team)
                all_stats.append(stats)
                print(f"Processed: {link}")
                sleep(random.randint(5, 11))
            except Exception as e:
                print(f"Error processing {link}: {e}")

        pd.DataFrame(all_stats).to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    schedule_urls = {
        "https://fbref.com/en/comps/13/2023-2024/schedule/2023-2024-Ligue-1-Scores-and-Fixtures": "Ll_2023.csv",
        "https://fbref.com/en/comps/13/2022-2023/schedule/2022-2023-Ligue-1-Scores-and-Fixtures": "Ll_2022.csv",
        "https://fbref.com/en/comps/13/2021-2022/schedule/2021-2022-Ligue-1-Scores-and-Fixtures": "Ll_2021.csv",
        "https://fbref.com/en/comps/13/2020-2021/schedule/2020-2021-Ligue-1-Scores-and-Fixtures": "Ll_2020.csv",
        "https://fbref.com/en/comps/13/2019-2020/schedule/2019-2020-Ligue-1-Scores-and-Fixtures": "Ll_2019.csv"
    }

    process_matches_multiple_seasons(schedule_urls)
