import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Wczytaj plik
df = pd.read_csv('games_with_stats.csv')

# 2. Przygotuj zmienne wynikowe
df['home_win'] = df['home_result'].apply(lambda x: 1 if x == 'win' else 0)
df['home_nie_przegrana'] = df['home_result'].isin(['win', 'draw']).astype(int)

# 3. Macierz wygranych gospodarz-formacja vs gość-formacja (ogólnie)
table_wins = df.groupby(['home_club_formation', 'away_club_formation'])['home_win'].mean().unstack()
table_not_lost = df.groupby(['home_club_formation', 'away_club_formation'])['home_nie_przegrana'].mean().unstack()

print("PROCENT WYGRANYCH GOSPODARZY (formacje, ogólnie):")
print(table_wins)

print("\nPROCENT NIEPRZEGRANYCH GOSPODARZY (formacje, ogólnie):")
print(table_not_lost)

# Opcjonalnie - ładna heatmapa:
plt.figure(figsize=(12,9))
sns.heatmap(table_wins, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title('Procent wygranych gospodarzy (wszystkie mecze)')
plt.show()

# 4. ANALIZA PRZY PODOBNYM POZIOMIE DRUŻYN

# pomocnicze kolumny
df['position_diff'] = np.abs(df['home_club_position'] - df['away_club_position'])
df['value_ratio'] = df[['squad_value_home', 'squad_value_away']].min(axis=1) / df[['squad_value_home', 'squad_value_away']].max(axis=1)

# Filtr tylko podobne drużyny: różnica pozycji max 2 i value_ratio powyżej 0.8 (mniej-więcej +-20%)
df_similar = df[(df['position_diff'] <= 2) & (df['value_ratio'] > 0.8)]

# Macierz dla podobnych
similar_table_wins = df_similar.groupby(['home_club_formation', 'away_club_formation'])['home_win'].mean().unstack()
similar_table_not_lost = df_similar.groupby(['home_club_formation', 'away_club_formation'])['home_nie_przegrana'].mean().unstack()

print("\nPROCENT WYGRANYCH GOSPODARZY PRZY PODOBNYM POZIOMIE:")
print(similar_table_wins)

print("\nPROCENT NIEPRZEGRANYCH GOSPODARZY PRZY PODOBNYM POZIOMIE:")
print(similar_table_not_lost)

# Druga heatmapa:
plt.figure(figsize=(12,9))
sns.heatmap(similar_table_wins, annot=True, fmt=".2f", cmap='YlOrRd')
plt.title('Procent wygranych gospodarzy (przy porównywalnych drużynach)')
plt.show()