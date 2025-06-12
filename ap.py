import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ---- ŁADOWANIE MODELI I PREPROCESSÓW ----
model_cls = keras.models.load_model("model_classifier.keras")
model_reg_home = keras.models.load_model("model_regression_home.keras")
model_reg_away = keras.models.load_model("model_regression_away.keras")
model_bin = keras.models.load_model("model_regression_bin.keras")
scaler = joblib.load("scaler.pkl")

import pickle
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# --- Opcjonalnie: listy dostępnych formacji/menedżerów ---
formations = sorted(list(set([x.split('home_club_formation_')[1]
                             for x in features if x.startswith("home_club_formation_")])))
managers = sorted(list(set([x.split('home_club_manager_name_')[1]
                            for x in features if x.startswith("home_club_manager_name_")])))

form_dict = {
    'brak formy': 2,
    'zła': 1,
    'bardzo zła': 0,
    'dobra': 3,
    'bardzo dobra': 4,
    'średnia': 2  # <-- DODAJEMY
}

# --- DOMYŚLNE WARTOŚCI ---
DEFAULT_xG = 1.2        # <- uzupełnij swoją średnią z ligi!
DEFAULT_POSSESSION = 50
DEFAULT_FORM = 2        # Neutralna
DEFAULT_MANAGER = managers[0] if len(managers)>0 else 'Unknown'

def preprocess_input(user_input):
    # --- Przekształć input w dataframe z wszystkimi potrzebnymi kolumnami i one-hotami ---
    # Używamy structure jak w treningu!
    row = {}
    # Podstawowe cechy
    row['squad_value_home'] = user_input['val_home']
    row['squad_value_away'] = user_input['val_away']
    row['home_club_position'] = user_input['pos_home']
    row['away_club_position'] = user_input['pos_away']
    row['xg_home'] = user_input['xg_home']
    row['xg_away'] = user_input['xg_away']
    row['possession_home'] = user_input['poss_home']
    row['possession_away'] = user_input['poss_away']
    row['home_form'] = form_dict.get(user_input['form_home'], DEFAULT_FORM)
    row['away_form'] = form_dict.get(user_input['form_away'], DEFAULT_FORM)
    row['yellow_cards_home'] = user_input.get('yellow_home', 2) #2 jako przykładowa średnia
    row['yellow_cards_away'] = user_input.get('yellow_away', 2)
    row['red_cards_home'] = user_input.get('red_home', 0)
    row['red_cards_away'] = user_input.get('red_away', 0)
    # Różnicowe
    row['diff_squad_value'] = row['squad_value_home'] - row['squad_value_away']
    row['diff_position'] = row['home_club_position'] - row['away_club_position']
    row['diff_xG'] = row['xg_home'] - row['xg_away']
    row['diff_possession'] = row['possession_home'] - row['possession_away']
    row['diff_form'] = row['home_form'] - row['away_form']
    row['diff_yellow_cards'] = row['yellow_cards_home'] - row['yellow_cards_away']
    row['diff_red_cards'] = row['red_cards_home'] - row['red_cards_away']
    # One-hot formations
    for formation in formations:
        row[f'home_club_formation_{formation}'] = int(user_input['form_home_tactic'] == formation)
        row[f'away_club_formation_{formation}'] = int(user_input['form_away_tactic'] == formation)
    # One-hot managers
    for manager in managers:
        row[f'home_club_manager_name_{manager}'] = int(user_input.get('manager_home', DEFAULT_MANAGER) == manager)
        row[f'away_club_manager_name_{manager}'] = int(user_input.get('manager_away', DEFAULT_MANAGER) == manager)
    # Pozostałe cechy one-hot zostają 0 (jeśli np. manager nie istnieje, to 0)
    # --- Odtwarzamy dokładnie ten SAM FORMAT jak features ---
    row_df = pd.DataFrame([row])
    # Uzupełnij brakujące kolumny zerami
    for f in features:
        if f not in row_df.columns:
            row_df[f] = 0
    row_df = row_df[features] # odpowiednia kolejność
    # Skalowanie
    X = scaler.transform(row_df)
    return X

def make_prediction(X):
    # Klasyfikacja 3-klasowa
    prob_cls = model_cls.predict(X)[0]
    result_index = np.argmax(prob_cls)
    labels = ["Wygrana gości", "Remis", "Wygrana gospodarzy"]
    main_pred = labels[result_index]

    prob_report = "\n".join([f"{lab}: {100*prob_cls[i]:.1f}%" for i, lab in enumerate(labels)])

    # Regresja na gole
    goal_home = model_reg_home.predict(X)[0, 0]
    goal_away = model_reg_away.predict(X)[0, 0]
    goal_home, goal_away = max(0, goal_home), max(0, goal_away)
    goal_home_rounded = int(round(goal_home))
    goal_away_rounded = int(round(goal_away))

    # Model binarny (wygrana gospodarzy vs brak wygranej)
    prob_win = model_bin.predict(X)[0, 0]
    win_label = "Wygrana gospodarzy" if prob_win > 0.5 else "Brak wygranej gospodarzy"

    result = (
        "=== Prognoza na podstawie modeli ===\n\n"
        f"Najbardziej prawdopodobny wynik (model 3-klasowy):\n"
        f"  {main_pred}\n\n"
        f"Rozkład prawdopodobieństw:\n"
        f"{prob_report}\n"
        f"\n"
        f"Przewidywane gole (zaokrąglone):\n"
        f"  Gospodarze: {goal_home_rounded}, Goście: {goal_away_rounded}\n"
        f"(model NN sugeruje surowo: {goal_home:.2f} - {goal_away:.2f})\n\n"
        f"Model binarny (wygrana gospodarzy vs brak wygranej):\n"
        f"  {win_label} ({prob_win*100:.1f}%)\n"
    )
    return result

# ---- TKINTER ----
def predict():
    try:
        # Zbierz dane z GUI
        user_input = {
            'val_home': float(entry_val_home.get()),
            'val_away': float(entry_val_away.get()),
            'pos_home': int(entry_pos_home.get()),
            'pos_away': int(entry_pos_away.get()),
            'xg_home': float(entry_xg_home.get() or DEFAULT_xG),
            'xg_away': float(entry_xg_away.get() or DEFAULT_xG),
            'poss_home': float(entry_poss_home.get() or DEFAULT_POSSESSION),
            'poss_away': float(entry_poss_away.get() or DEFAULT_POSSESSION),
            'form_home': combo_form_home.get() or "brak formy",
            'form_away': combo_form_away.get() or "brak formy",
            'form_home_tactic': combo_form_home_tactic.get(),
            'form_away_tactic': combo_form_away_tactic.get(),
            'manager_home': combo_man_home.get(),
            'manager_away': combo_man_away.get()
        }
        X = preprocess_input(user_input)
        result = make_prediction(X)
        messagebox.showinfo("Wynik prognozy", result)
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd wejścia: {e}")

root = tk.Tk()
root.title("Symulacja meczu")

tk.Label(root, text="Pozycja Gospodarzy:").grid(row=0, column=0)
entry_pos_home = tk.Entry(root)
entry_pos_home.grid(row=0, column=1)

tk.Label(root, text="Pozycja Gości:").grid(row=1, column=0)
entry_pos_away = tk.Entry(root)
entry_pos_away.grid(row=1, column=1)

tk.Label(root, text="Wartość składu Gosp. (miliony):").grid(row=2, column=0)
entry_val_home = tk.Entry(root)
entry_val_home.grid(row=2, column=1)

tk.Label(root, text="Wartość składu Gości (miliony):").grid(row=3, column=0)
entry_val_away = tk.Entry(root)
entry_val_away.grid(row=3, column=1)

tk.Label(root, text="Formacja Gospodarzy:").grid(row=4, column=0)
combo_form_home_tactic = ttk.Combobox(root, values=formations)
combo_form_home_tactic.grid(row=4, column=1)

tk.Label(root, text="Formacja Gości:").grid(row=5, column=0)
combo_form_away_tactic = ttk.Combobox(root, values=formations)
combo_form_away_tactic.grid(row=5, column=1)

tk.Label(root, text="xG Gosp. (opcjonalnie):").grid(row=6, column=0)
entry_xg_home = tk.Entry(root)
entry_xg_home.grid(row=6, column=1)

tk.Label(root, text="xG Gości (opcjonalnie):").grid(row=7, column=0)
entry_xg_away = tk.Entry(root)
entry_xg_away.grid(row=7, column=1)

tk.Label(root, text="Posiadanie Gosp. (opcjonalnie):").grid(row=8, column=0)
entry_poss_home = tk.Entry(root)
entry_poss_home.grid(row=8, column=1)

tk.Label(root, text="Posiadanie Gości (opcjonalnie):").grid(row=9, column=0)
entry_poss_away = tk.Entry(root)
entry_poss_away.grid(row=9, column=1)

tk.Label(root, text="Forma Gosp. (opcjonalny):").grid(row=10, column=0)
combo_form_home = ttk.Combobox(root, values=list(form_dict.keys()))
combo_form_home.grid(row=10, column=1)

tk.Label(root, text="Forma Gości (opcjonalny):").grid(row=11, column=0)
combo_form_away = ttk.Combobox(root, values=list(form_dict.keys()))
combo_form_away.grid(row=11, column=1)

tk.Label(root, text="Manager Gosp. (opcjonalnie):").grid(row=12, column=0)
combo_man_home = ttk.Combobox(root, values=managers)
combo_man_home.grid(row=12, column=1)

tk.Label(root, text="Manager Gości (opcjonalnie):").grid(row=13, column=0)
combo_man_away = ttk.Combobox(root, values=managers)
combo_man_away.grid(row=13, column=1)

btn = tk.Button(root, text="Przewiduj wynik", command=predict)
btn.grid(row=14, column=0, columnspan=2, pady=12)

root.mainloop()