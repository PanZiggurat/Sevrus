import pandas as pd
import numpy as np

df = pd.read_csv('games_with_stats.csv')
df['data'] = pd.to_datetime(df['data'])

# Zamień zera na NaN, jeśli traktujemy zero jako błąd/brak
df['squad_value_home'] = df['squad_value_home'].replace(0, np.nan)
df['squad_value_away'] = df['squad_value_away'].replace(0, np.nan)

def fill_squad_value(row, club_col, value_col):
    if not np.isnan(row[value_col]):
        return row[value_col]
    # Szukamy innych meczów tego klubu
    club_id = row[club_col]
    match_date = row['data']
    club_games = df[
        ((df['home_club_id'] == club_id) | (df['away_club_id'] == club_id)) &
        (df.index != row.name) # nie bierzemy bieżącego meczu
    ].copy()
    if club_games.empty:
        return np.nan
    club_games['date_diff'] = (club_games['data'] - match_date).abs()
    # Szukamy meczów, gdzie dla badanego klubu jest squad_value
    club_games['value'] = np.where(
        club_games['home_club_id'] == club_id,
        club_games['squad_value_home'],
        club_games['squad_value_away']
    )
    valid_games = club_games[~club_games['value'].isna()]
    if valid_games.empty:
        return np.nan
    # Najbliższy mecz
    res = valid_games.sort_values('date_diff').iloc[0]['value']
    return res

# Uzupełnij brakujące squad_value_home
df['squad_value_home'] = df.apply(
    lambda row: fill_squad_value(row, 'home_club_id', 'squad_value_home'),
    axis=1
)
# Uzupełnij brakujące squad_value_away
df['squad_value_away'] = df.apply(
    lambda row: fill_squad_value(row, 'away_club_id', 'squad_value_away'),
    axis=1
)

df.to_csv('games_with_stats.csv', index=False)
print("Braki w squad_value_home oraz squad_value_away zostały uzupełnione wartościami z najbliższego meczu tego klubu.")