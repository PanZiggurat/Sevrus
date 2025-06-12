import pandas as pd
import numpy as np

# Wczytanie i wstępne przetwarzanie
df = pd.read_csv('games_with_stats.csv')

# Procenty na liczby
df['possession_home'] = df['possession_home'].str.rstrip('%').astype(float)
df['possession_away'] = df['possession_away'].str.rstrip('%').astype(float)
df['xg_home'] = df['xg_home'].astype(float)
df['xg_away'] = df['xg_away'].astype(float)

form_dict = {'brak formy': 0, 'zła': 1, 'bardzo zła': 0, 'dobra': 3, 'bardzo dobra': 4}
df['home_form'] = df['home_form'].map(form_dict).fillna(2)
df['away_form'] = df['away_form'].map(form_dict).fillna(2)

# Cecha różnicowa
df['diff_squad_value'] = df['squad_value_home'] - df['squad_value_away']
df['diff_position'] = df['home_club_position'] - df['away_club_position']
df['diff_xG'] = df['xg_home'] - df['xg_away']
df['diff_possession'] = df['possession_home'] - df['possession_away']
df['diff_form'] = df['home_form'] - df['away_form']
df['diff_yellow_cards'] = df['yellow_cards_home'] - df['yellow_cards_away']
df['diff_red_cards'] = df['red_cards_home'] - df['red_cards_away']



# Target: win/draw/lose
def match_result(row):
    if row['home_club_goals'] > row['away_club_goals']:
        return 2
    elif row['home_club_goals'] == row['away_club_goals']:
        return 1
    else:
        return 0
df['result'] = df.apply(match_result, axis=1)

# WYBIERZEMY WSZYSTKO osobno Home/Away + różnice + formacje + menagerowie (one-hot)
features = [
    # różnicowe
    'diff_squad_value', 'diff_position', 'diff_xG', 'diff_possession', 'diff_form', 
    # raw home
    'squad_value_home','home_club_position', 'xg_home', 'possession_home', 'home_form', 'yellow_cards_home','red_cards_home',
    # raw away
    'squad_value_away','away_club_position', 'xg_away', 'possession_away', 'away_form', 'yellow_cards_away','red_cards_away',
]

# Dodajemy one-hot dla formacji oraz menedżerów (osobno dla home i away)
cat_cols = ['home_club_formation', 'away_club_formation', 'home_club_manager_name', 'away_club_manager_name']
df = pd.get_dummies(df, columns=cat_cols, drop_first=False) # nie robimy drop_first – chcemy wszystko
features.extend([col for col in df.columns if
                col.startswith('home_club_formation_') or
                col.startswith('away_club_formation_') or
                col.startswith('home_club_manager_name_') or
                col.startswith('away_club_manager_name_')
                ])

def compute_rolling_means(df, window=5):
    df = df.copy()
    # Najpierw sortujemy po sezonie i dacie meczu
    df = df.sort_values(['season', 'data'])
    # Do przechowania nowych kolumn
    df['xg_home_hist_mean'] = np.nan
    df['xg_away_hist_mean'] = np.nan
    df['poss_home_hist_mean'] = np.nan
    df['poss_away_hist_mean'] = np.nan

    for season in df['season'].unique():
        # Home team means
        for team in pd.unique(df.loc[df['season'] == season, 'home_club_name']):
            mask_home = (df['season'] == season) & (df['home_club_name'] == team)
            matches_team = df[mask_home].sort_values('data')
            for i, idx in enumerate(matches_team.index):
                matches_before = matches_team.iloc[:i]
                if len(matches_before) > 0:
                    # Bierzemy mecze tylko z tego sezonu!!!
                    n = min(window, len(matches_before))
                    last_matches = matches_before.iloc[-n:]  # ostatnie
                    # średnie dla HOME (team gra u siebie)
                    df.at[idx, 'xg_home_hist_mean'] = last_matches['xg_home'].mean()
                    df.at[idx, 'poss_home_hist_mean'] = last_matches['possession_home'].mean()
                else:
                    # Pierwszy mecz - weź średnią z 5 pierwszych (NIE licząc bieżącego)
                    next_matches = matches_team.iloc[1:window+1]
                    if len(next_matches) > 0:
                        df.at[idx, 'xg_home_hist_mean'] = next_matches['xg_home'].mean()
                        df.at[idx, 'poss_home_hist_mean'] = next_matches['possession_home'].mean()
        
        # Away team means
        for team in pd.unique(df.loc[df['season'] == season, 'away_club_name']):
            mask_away = (df['season'] == season) & (df['away_club_name'] == team)
            matches_team = df[mask_away].sort_values('data')
            for i, idx in enumerate(matches_team.index):
                matches_before = matches_team.iloc[:i]
                if len(matches_before) > 0:
                    n = min(window, len(matches_before))
                    last_matches = matches_before.iloc[-n:]  # ostatnie
                    df.at[idx, 'xg_away_hist_mean'] = last_matches['xg_away'].mean()
                    df.at[idx, 'poss_away_hist_mean'] = last_matches['possession_away'].mean()
                else:
                    # Pierwszy mecz - weź średnią z 5 pierwszych (NIE licząc bieżącego)
                    next_matches = matches_team.iloc[1:window+1]
                    if len(next_matches) > 0:
                        df.at[idx, 'xg_away_hist_mean'] = next_matches['xg_away'].mean()
                        df.at[idx, 'poss_away_hist_mean'] = next_matches['possession_away'].mean()
    return df

df['date'] = pd.to_datetime(df['data'])
df = compute_rolling_means(df, window=5)
df['xg_home'] = df['xg_home_hist_mean']
df['xg_away'] = df['xg_away_hist_mean']
df['possession_home'] = df['poss_home_hist_mean']
df['possession_away'] = df['poss_away_hist_mean']
# Dane wejściowe/wyjściowe
import pickle
with open("features.pkl", "wb") as f:
    pickle.dump(features, f)

X = df[features]
y = df['result']
y_home = df['home_club_goals']
y_away = df['away_club_goals']

# Reszta kodu JAK POPRZEDNIO!
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
y_home_train, y_home_test = y_home.loc[X_train.index], y_home.loc[X_test.index]
y_away_train, y_away_test = y_away.loc[X_train.index], y_away.loc[X_test.index]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 3
y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes)

model_cls = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model_cls.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_cls.fit(X_train_scaled, y_train_categorical, validation_split=0.1, epochs=25, batch_size=64)

# Predykcja
y_pred_prob = model_cls.predict(X_test_scaled)
y_pred_cls = y_pred_prob.argmax(axis=1)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
print(classification_report(y_test, y_pred_cls, target_names=["Away win", "Draw", "Home win"]))
cm = confusion_matrix(y_test, y_pred_cls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Away win", "Draw", "Home win"])
disp.plot(cmap="Blues")
plt.title("Macierz pomyłek NN classifier")
plt.show()

# NN na regresję (gole)
model_reg_home = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model_reg_home.compile(optimizer='adam', loss='mae')
model_reg_home.fit(X_train_scaled, y_home_train, validation_split=0.1, epochs=25, batch_size=64, verbose=1)

model_reg_away = keras.models.clone_model(model_reg_home)
model_reg_away.compile(optimizer='adam', loss='mae')
model_reg_away.fit(X_train_scaled, y_away_train, validation_split=0.1, epochs=25, batch_size=64, verbose=1)

# Predykcja regresji
pred_goals_home_nn = model_reg_home.predict(X_test_scaled).flatten()
pred_goals_away_nn = model_reg_away.predict(X_test_scaled).flatten()

pred_goals_home_nn = np.round(pred_goals_home_nn).astype(int)
pred_goals_away_nn = np.round(pred_goals_away_nn).astype(int)

from sklearn.metrics import mean_absolute_error
print("MAE home goals (NN):", mean_absolute_error(y_home_test, pred_goals_home_nn))
print("MAE away goals (NN):", mean_absolute_error(y_away_test, pred_goals_away_nn))


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay

# 1. Zamieniamy przewidziane gole na W/D/L (0/1/2)
def goals_to_result(h, a):
    if h > a:
        return 2 # Home win
    elif h == a:
        return 1 # Draw
    else:
        return 0 # Away win

# Konwersja przewidywanych goli na przewidywane wyniki meczów (0, 1, 2)
pred_results_from_goals = [goals_to_result(h, a) for h, a in zip(pred_goals_home_nn, pred_goals_away_nn)]
true_results = y_test.values  # Prawdziwe wyniki meczów (0, 1, 2) z danych testowych

# 2. Accuracy & macierz pomyłek dla predykcji na podstawie goli
acc_results_from_goals = accuracy_score(true_results, pred_results_from_goals)
cm_goals = confusion_matrix(true_results, pred_results_from_goals)

print("\n*** Wynik na podstawie przewidzianych goli (NN) ***")
print("Accuracy (wynik z regresji goli):", round(acc_results_from_goals, 3))
print(classification_report(true_results, pred_results_from_goals, target_names=["Away win", "Draw", "Home win"]))
ConfusionMatrixDisplay(confusion_matrix=cm_goals, display_labels=["Away win", "Draw", "Home win"]).plot(cmap="Purples")
plt.title("Macierz pomyłek – wynik z regresji goli")
plt.show()


print("\nPierwszy mecz testowy (NN):")
print("True wynik: home_goals = {}, away_goals = {}".format(y_home_test.iloc[0], y_away_test.iloc[0]))
print("Prognoza:   home_goals = {:.2f}, away_goals = {:.2f}".format(pred_goals_home_nn[0], pred_goals_away_nn[0]))

probs = y_pred_prob[0]
print("Prognoza (procentowo):")
print("Away win: {:.2f}%".format(probs[0]*100))
print("Draw: {:.2f}%".format(probs[1]*100))
print("Home win: {:.2f}%".format(probs[2]*100))


df['result_binary'] = df['home_club_goals'] > df['away_club_goals']
df['result_binary'] = df['result_binary'].astype(int)

y_binary = (df['result'] == 2).astype(int)

# Podział
y_train_binary, y_test_binary = y_binary.loc[X_train.index], y_binary.loc[X_test.index]

# Model binarny
model_bin = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_bin.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Trening modelu binarnego
model_bin.fit(X_train_scaled, y_train_binary, validation_split=0.1, epochs=25, batch_size=64)
y_pred_prob_bin = model_bin.predict(X_test_scaled).flatten()
y_pred_bin = (y_pred_prob_bin > 0.5).astype(int)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
print(classification_report(y_test_binary, y_pred_bin, target_names=["No win", "Win"]))
cm_bin = confusion_matrix(y_test_binary, y_pred_bin)
disp_bin = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=["No win", "Win"])
disp_bin.plot(cmap="Greens")
plt.title("Macierz pomyłek - wygrana vs brak wygranej")
plt.show()
from sklearn.metrics import accuracy_score

# Model 1 (3 klasy)
acc_model_3class = accuracy_score(y_test, y_pred_cls)

# Model 2 (Binary: Win vs No Win)


print("Accuracy – 3 klasy (dokładny wynik meczu):", round(acc_model_3class, 3))

from sklearn.metrics import classification_report

# Model 1
print("=== Model 3-klasowy ===")
print(classification_report(y_test, y_pred_cls, target_names=["Away win", "Draw", "Home win"]))

# Model 2
print("=== Model 2-klasowy (Win vs No Win) ===")

import joblib

joblib.dump(scaler, "scaler.pkl")



model_cls.save("model_classifier.keras")          
model_reg_home.save("model_regression_home.keras")
model_reg_away.save("model_regression_away.keras")
model_bin.save("model_regression_bin.keras")