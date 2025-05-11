import pandas as pd
import numpy as np

# Wczytaj pliki
games = pd.read_csv('games_with_stats.csv')
lineups = pd.read_csv('game_lineups.csv')
player_vals = pd.read_csv('player_valuations.csv')

# Dla szybkości zamieniamy daty na datetime
games['data'] = pd.to_datetime(games['data'])
player_vals['date'] = pd.to_datetime(player_vals['date'])

# Filtrujemy wyjściowe składy
starters = lineups[lineups['type'] == 'starting_lineup']

def get_squad_value(game_id, club_id, game_date):
    players = starters[
        (starters['game_id'] == game_id) &
        (starters['club_id'] == club_id)
    ]['player_id'].drop_duplicates()
    
    total_value = 0
    for pid in players:
        values = player_vals[player_vals['player_id'] == pid]
        if not values.empty:
            # Liczymy dystans dni między wyceną a meczem
            values = values.copy()
            values['days_diff'] = (values['date'] - game_date).abs()
            rec = values.sort_values('days_diff').iloc[0]
            total_value += rec['market_value_in_eur']
        # Jeśli nie ma żadnej wyceny, pomijamy (czyli 0)
    return total_value

# Dodajemy nowe kolumny na squad values
games['squad_value_home'] = games.apply(
    lambda row: get_squad_value(row['game_id'], row['home_club_id'], row['data']),
    axis=1
)

games['squad_value_away'] = games.apply(
    lambda row: get_squad_value(row['game_id'], row['away_club_id'], row['data']),
    axis=1
)

games.to_csv('games_with_stats.csv', index=False)
print('Dodano squad_value_home i squad_value_away!')