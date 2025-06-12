import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. --- WCZYTANIE I PRZYGOTOWANIE DANYCH ---
df = pd.read_csv('games_with_stats.csv')

form_map = {'bardzo zła': -2, 'zła': -1, 'średnia': 0, 'dobra': 1, 'bardzo dobra': 2}

# HOME
home = df[[
    'squad_value_home', 'home_club_position', 'home_club_goals', 'home_result', 'home_form'
]].copy()
home.columns = [
    'squad_value', 'position', 'goals', 'result', 'form'
]
home['side'] = 'home'
home['form_num'] = home['form'].map(form_map)

# AWAY
away = df[[
    'squad_value_away', 'away_club_position', 'away_club_goals', 'away_result', 'away_form'
]].copy()
away.columns = [
    'squad_value', 'position', 'goals', 'result', 'form'
]
away['side'] = 'away'
away['form_num'] = away['form'].map(form_map)

# Łączymy
df_long = pd.concat([home, away], ignore_index=True)

# Usuwamy rekordy bez formy
df_long = df_long[df_long['form'].isin(form_map.keys())].copy()

# Tworzymy kolumny z wynikiem numerycznym:
df_long['result_num'] = df_long['result'].map({'win': 1, 'draw': 0, 'lose': -1})
df_long['win'] = (df_long['result_num'] == 1).astype(int)
df_long['no_lose'] = (df_long['result_num'] >= 0).astype(int)

# Wybieramy cechy do korelacji i modelowania
features = ['squad_value', 'position', 'goals', 'form_num']

# --- KORELACJE ---
corr_table = df_long[features + ['result_num']].corr().round(2)
print('\n=== Tabela korelacji ===')
print(corr_table)

plt.figure(figsize=(7, 6))
sns.heatmap(corr_table, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Macierz korelacji (squad_value, position, goals, form_num, result)")
plt.show()

# --- MODELE: win i no_lose ---
X = df_long[features]
y_win = df_long['win']
y_nolose = df_long['no_lose']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_win, test_size=0.3, random_state=42)
model_win = LogisticRegression().fit(X_train, y_train)

X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(X_scaled, y_nolose, test_size=0.3, random_state=42)
model_nl = LogisticRegression().fit(X_train_nl, y_train_nl)

# --- TABELA WPŁYWÓW CECH ---
coef_table = pd.DataFrame({
    'cecha': features,
    'Współczynnik: wygrana': model_win.coef_[0],
    'Współczynnik: brak porażki': model_nl.coef_[0]
})
print('\n=== Wpływ cech na szansę wygranej/braku porażki ===')
print(coef_table.round(3))