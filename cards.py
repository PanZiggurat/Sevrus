import pandas as pd

# Wczytaj dane
games = pd.read_csv('games_with_stats.csv')
app = pd.read_csv('appearances.csv')

# Upewnij się, że nazwy kartek są ok:
# (zamień nazwę, jeśli trzeba, na tą która masz w pliku)
yellow_col = 'yellow_cards'
red_col = 'red_cards'  # czasem red_cards

# Zlicz sumarycznie żółte i czerwone kartki na klub i mecz
agg = app.groupby(['game_id', 'player_club_id']).agg({
    yellow_col: 'sum',
    red_col: 'sum'
}).reset_index().rename(
    columns={
        yellow_col: 'club_yellow_cards',
        red_col: 'club_red_cards'
    }
)

# Połącz z gospodarzem ("home");
games = games.merge(
    agg, left_on=['game_id', 'home_club_id'],
    right_on=['game_id', 'player_club_id'],
    how='left'
)
games = games.rename(columns={
    'club_yellow_cards': 'yellow_cards_home',
    'club_red_cards': 'red_cards_home'
})
games = games.drop('player_club_id', axis=1)

# Połącz z gościem ("away");
games = games.merge(
    agg, left_on=['game_id', 'away_club_id'],
    right_on=['game_id', 'player_club_id'],
    how='left'
)
games = games.rename(columns={
    'club_yellow_cards': 'yellow_cards_away',
    'club_red_cards': 'red_cards_away'
})
games = games.drop('player_club_id', axis=1)

# Jeśli były mecze bez kartek, mogą pojawić się NaNy, zamień je na zera
for col in ['yellow_cards_home', 'red_cards_home', 'yellow_cards_away', 'red_cards_away']:
    games[col] = games[col].fillna(0).astype(int)

games.to_csv('games_with_stats.csv', index=False)
print("Dodano kolumny: yellow_cards_home, red_cards_home, yellow_cards_away, red_cards_away!")