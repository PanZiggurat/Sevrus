import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import itertools

# 1. --- WCZYTANIE I PRZYGOTOWANIE DANYCH ---
df = pd.read_csv('games_with_stats.csv')

df["result_num"] = df["home_result"].map({'win': 1, 'draw': 0, 'lose': -1})
df["no_lose"] = df["result_num"].apply(lambda x: 1 if x >= 0 else 0)
df["pos_diff"] = df["away_club_position"] - df["home_club_position"]
df["value_diff"] = df["squad_value_home"] - df["squad_value_away"]

form_map = {'bardzo zła': -2, 'zła': -1, 'średnia': 0, 'dobra': 1, 'bardzo dobra': 2}
df['home_form_num'] = df['home_form'].map(form_map)
df_reg = df[~df['home_form'].isin(['brak formy'])].copy()

# Ważne: Tutaj używamy 4 cech BEZ 'value_ratio'
features = ['pos_diff', 'value_diff', 'home_form_num', 'home_club_goals']
df_reg = df_reg.dropna(subset=features)

X = df_reg[features]
y_win = (df_reg['home_result'] == 'win').astype(int)
y_nolose = df_reg["no_lose"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. --- REGRESJA LOGISTYCZNA: wygrana/brak porażki ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_win, test_size=0.3, random_state=42)
model_win = LogisticRegression().fit(X_train, y_train)

X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(X_scaled, y_nolose, test_size=0.3, random_state=42)
model_nl = LogisticRegression().fit(X_train_nl, y_train_nl)

# 3. --- KORELACJE: tabela i wykres ---
corr_table = df_reg[features].corr().round(2)
print('\n=== Tabela korelacji ===')
print(corr_table)

plt.figure(figsize=(6, 5))
sns.heatmap(corr_table, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Macierz korelacji (4 wybrane cechy)")
plt.show()

# 4. --- PODSUMOWANIE MODELU: tabela wpływów cech ---
coef_table = pd.DataFrame({
    'cecha': features,
    'Współczynnik: wygrana': model_win.coef_[0],
    'Współczynnik: brak porażki': model_nl.coef_[0]
})
print('\n=== Wpływ cech na szansę wygranej/braku porażki ===')
print(coef_table.round(3))










