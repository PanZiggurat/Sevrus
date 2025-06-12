import pandas as pd

# Wczytaj dane
df = pd.read_csv("games_with_stats.csv", parse_dates=["data"])

# Sortuj dane chronologicznie
df = df.sort_values(by=["season", "data"]).reset_index(drop=True)

# Zainicjuj kolumny formy
df["home_form"] = ""
df["away_form"] = ""

# Funkcja do konwersji punktów na kategorię formy
def form_category(points):
    if points <= 3:
        return "bardzo zła"
    elif points <= 5:
        return "zła"
    elif points <= 7:
        return "średnia"
    elif points <= 11:
        return "dobra"
    else:
        return "bardzo dobra"

# Historia meczów każdej drużyny
team_history = {}

# Iteracja po wierszach
for idx, row in df.iterrows():
    season = row["season"]
    date = row["data"]
    
    home_id = row["home_club_id"]
    away_id = row["away_club_id"]
    home_result = row["home_result"]
    away_result = row["away_result"]
    
    home_pos = row.get("home_club_position", None)
    away_pos = row.get("away_club_position", None)
    home_val = row.get("home_squad_value", None)
    away_val = row.get("away_squad_value", None)
    
    for team, result, opp_id, pos, opp_pos, val, opp_val, side in [
        (home_id, home_result, away_id, home_pos, away_pos, home_val, away_val, "home"),
        (away_id, away_result, home_id, away_pos, home_pos, away_val, home_val, "away")
    ]:
        key = (season, team)
        
        # Inicjalizacja historii jeśli nie istnieje
        if key not in team_history:
            team_history[key] = []

        # Oblicz forma na podstawie ostatnich 5 meczów
        recent_matches = team_history[key][-5:]
        total_points = 7  # Domyślnie "średnia forma"
        
        if recent_matches:
            total_points = 0
            for match in recent_matches:
                pts = 0
                res, opp_pos_m, opp_val_m, self_pos_m, self_val_m = match
                if res == "win":
                    pts = 3
                elif res == "draw":
                    pts = 1

                if opp_pos_m is not None and self_pos_m is not None:
                    if opp_pos_m - self_pos_m >= 15:
                        pts += 0.5
                if opp_val_m is not None and self_val_m is not None:
                    if opp_val_m < self_val_m / 2:
                        pts += 0.5
                    elif self_val_m < opp_val_m / 2 and res == "lose":
                        pts -= 0.5
                        
                total_points += pts

        df.at[idx, f"{side}_form"] = form_category(total_points)

    # Dodaj aktualny mecz do historii dla przyszłych ocen
    team_history[(season, home_id)].append((home_result, away_pos, away_val, home_pos, home_val))
    team_history[(season, away_id)].append((away_result, home_pos, home_val, away_pos, away_val))

# Zapisz zaktualizowany plik
df.to_csv("games_with_stats.csv", index=False)
