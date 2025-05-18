import pandas as pd
import numpy as np

# 1. Wczytanie i wstępne przetwarzanie danych
df = pd.read_csv('games_with_stats.csv')

# Procenty na liczby
df['possession_home'] = df['possession_home'].str.rstrip('%').astype(float)
df['possession_away'] = df['possession_away'].str.rstrip('%').astype(float)

# xG na float
df['xg_home'] = df['xg_home'].astype(float)
df['xg_away'] = df['xg_away'].astype(float)

# Formę na liczby
form_dict = {'brak formy': 0, 'zła': 1, 'bardzo zła': 0, 'dobra': 3, 'bardzo dobra': 4}
df['home_form'] = df['home_form'].map(form_dict).fillna(2)
df['away_form'] = df['away_form'].map(form_dict).fillna(2)

# Nowe cechy różnicowe
df['diff_squad_value'] = df['squad_value_home'] - df['squad_value_away']
df['diff_position'] = df['home_club_position'] - df['away_club_position']
df['diff_xG'] = df['xg_home'] - df['xg_away']
df['diff_possession'] = df['possession_home'] - df['possession_away']
df['diff_form'] = df['home_form'] - df['away_form']
df['diff_yellow_cards'] = df['yellow_cards_home'] - df['yellow_cards_away']
df['diff_red_cards'] = df['red_cards_home'] - df['red_cards_away']

# Przykładowe cechy wejściowe
features = [
    'diff_squad_value', 'diff_position', 'diff_xG', 'diff_possession', 'diff_form',
    'diff_yellow_cards', 'diff_red_cards'
]

# Formacje jako one-hot (czyli kolumny 0/1 dla każdej unikalnej formacji)
features_extra = ['home_club_formation', 'away_club_formation']
df = pd.get_dummies(df, columns=features_extra)
features.extend([col for col in df.columns if col.startswith('home_club_formation_') or col.startswith('away_club_formation_')])

# Target: wynik meczu
def match_result(row):
    if row['home_club_goals'] > row['away_club_goals']:
        return 2  # Home Win
    elif row['home_club_goals'] == row['away_club_goals']:
        return 1  # Draw
    else:
        return 0  # Away Win

df['result'] = df.apply(match_result, axis=1)

# 2. Podział na zbiór treningowy i testowy
from sklearn.model_selection import train_test_split

X = df[features]
y = df['result']
y_home = df['home_club_goals']
y_away = df['away_club_goals']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
# Tak samo do regresji: train/test split na tych samych indeksach
y_home_train, y_home_test = y_home.loc[X_train.index], y_home.loc[X_test.index]
y_away_train, y_away_test = y_away.loc[X_train.index], y_away.loc[X_test.index]

# 3. Klasyfikacja wyniku meczu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)

print("=== Raport klasyfikacji ===")
print(classification_report(y_test, y_pred, target_names=["Away win", "Draw", "Home win"]))

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Away win", "Draw", "Home win"])
disp.plot(cmap="Blues")
plt.title("Macierz pomyłek dla wyniku meczu")
plt.show()

# Przykładowe prawdopodobieństwa i kursy dla pierwszego meczu testowego
print("Przykładowe prawdopodobieństwa z klasyfikatora:", probs[0])
print("Kursy na dany mecz:", 1/np.clip(probs[0], 1e-8, None))  # Uwaga: czysto matematyczne, bez marży

# 4. Regresja liczby goli
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Regresja goli gospodarzy
reg_home = RandomForestRegressor(n_estimators=100, random_state=42)
reg_home.fit(X_train, y_home_train)
pred_goals_home = reg_home.predict(X_test)

# Regresja goli gości
reg_away = RandomForestRegressor(n_estimators=100, random_state=42)
reg_away.fit(X_train, y_away_train)
pred_goals_away = reg_away.predict(X_test)

# Ocena regresji
print("MAE home goals:", mean_absolute_error(y_home_test, pred_goals_home))
print("MAE away goals:", mean_absolute_error(y_away_test, pred_goals_away))

# Wyświetl przykładowe predykcje
print("\nPierwszy mecz testowy:")
print("True wynik: home_goals = {}, away_goals = {}".format(y_home_test.iloc[0], y_away_test.iloc[0]))
print("Prognoza:   home_goals = {:.2f}, away_goals = {:.2f}".format(pred_goals_home[0], pred_goals_away[0]))