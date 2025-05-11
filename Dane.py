import pandas as pd

# Wczytanie plików CSV
games = pd.read_csv('games.csv')
competitions = pd.read_csv('competitions.csv')

# Wyciągnięcie unikalnych par (season, competition_id) z pliku games.csv
unique_pairs = games.groupby(['season', 'competition_id']).size().reset_index(name='count')

# Połączenie tych par z plikiem competitions.csv po kolumnie competition_id
merged_data = pd.merge(unique_pairs, competitions, left_on='competition_id', right_on='competition_id', how='left')

# Wybranie interesujących kolumn i zmiana nazw dla przejrzystości
result = merged_data[['season', 'competition_id', 'name', 'count']].rename(columns={'name': 'competition_name'})

# Wyświetlenie wyniku
print(result)

# Ewentualnie zapisanie do nowego pliku CSV
result.to_csv('unique_pairs_with_counts.csv', index=False)