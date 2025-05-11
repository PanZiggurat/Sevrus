import pandas as pd
import numpy as np

df = pd.read_csv("games_with_stats.csv")

# Zakładamy, że masz (oprócz historii meczów) następujące kolumny:
# game_id, season, data, home_club_id, away_club_id, home_result, away_result
# home_club_position, away_club_position, squad_value_home, squad_value_away

def calc_match_points(result, own_pos, opp_pos, own_value, opp_value):
    # Punkty bazowe
    if result == "win":
        points = 3
    elif result == "draw":
        points = 1
    elif result == "lose":
        points = 0
    else:
        return np.nan

    # Dodatki/odejmowanie za różnicę pozycji
    if not (np.isnan(own_pos) or np.isnan(opp_pos)):
        pos_diff = own_pos - opp_pos
        if abs(pos_diff) >= 12:
            # Gorsza drużyna: większy numer pozycji
            if result == "draw":
                if pos_diff > 0:
                    # remis słabszej z lepszą
                    points += 0.5
                elif pos_diff < 0:
                    # remis lepszej ze słabszą
                    points -= 0.5
            elif result == "win":
                if pos_diff > 0:
                    # wygrana słabszej z lepszą
                    points += 1
                # wygrana lepszej = brak bonusu
            elif result == "lose":
                if pos_diff < 0:
                    # przegrana lepszej ze słabszą
                    points -= 1
                # przegrana słabszej = brak bonusu

    # Dodatki/odejmowanie za różnicę składu
    if not (np.isnan(own_value) or np.isnan(opp_value)):
        if own_value < opp_value / 2:
            # own to 2x słabsza drużyna
            if result == "draw":
                points += 0.5
            elif result == "win":
                points += 1
            # przegrana = 0
        elif own_value > opp_value * 2:
            # own to 2x lepsza drużyna
            if result == "draw":
                points -= 0.5
            elif result == "lose":
                points -= 1
            # wygrana = 0
    return points

# -- Wylicz punkty za każdy mecz osobno dla home i away
df['home_points'] = df.apply(
    lambda row: calc_match_points(
        row['home_result'],
        row['home_club_position'], row['away_club_position'],
        row['squad_value_home'], row['squad_value_away']),
    axis=1
)
df['away_points'] = df.apply(
    lambda row: calc_match_points(
        row['away_result'],
        row['away_club_position'], row['home_club_position'],
        row['squad_value_away'], row['squad_value_home']),
    axis=1
)

# Przygotuj formę dla każdej drużyny na każdy mecz
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values(['season', 'home_club_id', 'data'])  # posortuj chronologicznie w sezonie

def get_form(df, club_id_col, points_col, season_col='season', date_col='data'):
    form_list = []
    for idx, row in df.iterrows():
        club_id = row[club_id_col]
        season = row[season_col]
        date = row[date_col]
        # Wyciągnij 5 poprzednich meczów tego klubu w tym sezonie, które były PRZED bieżącą datą
        past_games = df[
            (df[season_col] == season) &
            ((df['home_club_id'] == club_id) | (df['away_club_id'] == club_id)) &
            (df[date_col] < date)
        ].sort_values(date_col, ascending=False).head(5)
        if past_games.shape[0] < 5:
            form_list.append('brak formy')
        else:
            # uwzględnij czy klub był gospodarzem/gościem wybierając właściwą kolumnę z punktami
            points = []
            for _, past in past_games.iterrows():
                if past['home_club_id'] == club_id:
                    points.append(past['home_points'])
                else:
                    points.append(past['away_points'])
            suma = np.nansum(points)
            # Możesz podzielić na kategorie, np.:
            if suma <= 3:
                form_cat = "bardzo zła"
            elif suma <= 5:
                form_cat = "zła"
            elif suma <= 8:
                form_cat = "średnia"
            elif suma <= 11:
                form_cat = "dobra"
            else:
                form_cat = "bardzo dobra"
            form_list.append(form_cat)
    return form_list

# Dla gospodarza
df['home_form'] = get_form(df, 'home_club_id', 'home_points')
# Dla gościa
df['away_form'] = get_form(df, 'away_club_id', 'away_points')

df.to_csv("games_with_stats.csv", index=False)
print("Dodano kolumny z formą drużyn. Gotowe!")