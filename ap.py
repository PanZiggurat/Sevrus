import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib

# Wczytaj modele
model_cls = keras.models.load_model("model_classifier.keras")

model_home = keras.models.load_model("model_regression_home.keras")
model_away = keras.models.load_model("model_regression_away.keras")


# Wczytaj skalera
scaler = joblib.load("scaler.save")

st.title("Predykcja wyniku meczu ⚽")

# Form inputów
squad_value_home = st.number_input("Wartość składu - Gospodarze (mln)", value=500)
squad_value_away = st.number_input("Wartość składu - Goście (mln)", value=450)
xg_home = st.number_input("xG gospodarzy", value=1.3)
xg_away = st.number_input("xG gości", value=1.1)
possession_home = st.slider("Posiadanie piłki gospodarzy (%)", 0, 100, 55)
possession_away = 100 - possession_home

# I tak dalej...

# Tutaj przygotowujesz DataFrame tak samo jak w treningu
# Przykład:
df = pd.DataFrame({
    'squad_value_home': [squad_value_home],
    'squad_value_away': [squad_value_away],
    'xg_home': [xg_home],
    'xg_away': [xg_away],
    'possession_home': [possession_home],
    'possession_away': [possession_away],
    # ... inne cechy
})

# Oblicz różnice
df['diff_squad_value'] = df['squad_value_home'] - df['squad_value_away']
df['diff_xG'] = df['xg_home'] - df['xg_away']
df['diff_possession'] = df['possession_home'] - df['possession_away']

# Dodaj one-hot jeśli trzeba

# Ustal kolejność kolumn zgodnie z `features`
# Zakładamy że masz zapisane `features` i je tutaj wczytasz
features = joblib.load("features.save")  # np. lista kolumn
X = df[features]

# Skaluj
X_scaled = scaler.transform(X)

# Predykcja
pred_result = model_cls.predict(X_scaled).argmax(axis=1)[0]
pred_home_goals = round(model_reg_home.predict(X_scaled).flatten()[0])
pred_away_goals = round(model_reg_away.predict(X_scaled).flatten()[0])

# Wynik tekstowy
result_map = {0: "Goście wygrają", 1: "Remis", 2: "Gospodarze wygrają"}
st.subheader("Predykcja:")
st.write(f"Wynik meczu: {pred_home_goals} : {pred_away_goals}")
st.write(f"Klasyfikacja wyniku: **{result_map[pred_result]}**")
