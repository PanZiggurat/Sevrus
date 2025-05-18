import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('games_with_stats.csv')



# Dodaj nowe kolumny:
# – wynik jako liczba: 1=wygrana gospodarzy, 0=remis, -1=przegrana gospodarzy
df["result_num"] = df["home_result"].map({'win': 1, 'draw': 0, 'lose': -1})
# – brak porażki gospodarzy (win lub draw)
df["no_lose"] = df["result_num"].apply(lambda x: 1 if x >= 0 else 0)
# Różnica pozycji (im wyższa na minusie, tym "lepsza" pozycja gospodarzy)
df["pos_diff"] = df["away_club_position"] - df["home_club_position"]
# Różnica wartości składu
df["value_diff"] = df["squad_value_home"] - df["squad_value_away"]

df['pos_bin'] = pd.cut(df['home_club_position'], [0,5,10,15,20], labels=['1-5','6-10','11-15','16-20'])

df['value_ratio'] = df['squad_value_home'] / df['squad_value_away']
df['value_ratio_bin'] = pd.cut(df['value_ratio'], [0,0.8,1.2,2,10], labels=['Zdecydowanie słabszy','Podobny','Mocniejszy do 2x','>2x mocniejszy'])
df['value_home_bin'] = pd.qcut(df['squad_value_home'], 4, labels=['najsłabszy', 'średni-', 'średni+', 'najbogatszy'])
df_forma = df[~df['home_form'].isin(['brak formy'])]


wyniki_gosp = df['home_result'].value_counts(normalize=True) * 100
wyniki_gosp.plot(kind='bar')
plt.ylabel('Udział [%]')
plt.title('Procent wyników gospodarza (ogółem)')
plt.savefig("procenty_wynik_gospodarz.png", dpi=150, bbox_inches='tight')
plt.show()

sns.boxplot(x="home_result", y="home_club_position", data=df)
plt.title("Pozycja w tabeli gospodarza a wynik meczu")
plt.show()



print(df.groupby("no_lose")[["home_club_position","squad_value_home"]].mean())


sns.scatterplot(x='home_club_position', y='home_club_goals', data=df, alpha=0.3)
plt.title("Pozycja gospodarza a liczba strzelonych goli")
plt.show()

procenty = pd.crosstab(df['value_ratio_bin'], df['home_result'], normalize='index') * 100


# Krzyżowa tabela procentów
procenty = pd.crosstab(df['value_ratio_bin'], df['home_result'], normalize='index') * 100

# Wykres słupkowy procentów
procenty.plot(kind='bar', stacked=True)
plt.ylabel('Udział wyniku [%]')
plt.title('Procent wyników gospodarza w zależności od relacji wartości składów')
plt.legend(title='home_result', loc='upper left')
plt.savefig('wartosc_vs_wynik_proc.png', dpi=150, bbox_inches='tight')
plt.show()

procenty2 = pd.crosstab(df['value_home_bin'], df['home_result'], normalize='index') * 100
procenty2.plot(kind='bar', stacked=True)
plt.ylabel('Udział wyniku [%]')
plt.title('Procent wyników gospodarza wg kwartylu wartości składu')
plt.legend(title='home_result', loc='upper left')
plt.savefig('wynik_vs_value_gospodarz_proc.png', dpi=150, bbox_inches='tight')
plt.show()

form_map = {
    'bardzo zła': -2,
    'zła': -1,
    'średnia': 0,
    'dobra': 1,
    'bardzo dobra': 2
}
forma_kolejnosc = ['bardzo zła','zła','dobra','średnia','bardzo dobra']
posval_kolejnosc = [
    '1-5-niższa', '1-5-wyższa',
    '6-10-niższa', '6-10-wyższa',
    '11-15-niższa', '11-15-wyższa',
    '16-20-niższa', '16-20-wyższa'
]


df_gosp = df[~df['home_form'].isin(['brak formy'])].copy()
df_gosp['home_form_num'] = df_gosp['home_form'].map(form_map)
df_gosp['home_pos_bin'] = pd.cut(df_gosp['home_club_position'], [0,5,10,15,20], labels=['1-5','6-10','11-15','16-20'])
df_gosp['home_value_bin'] = pd.qcut(df_gosp['squad_value_home'], 2, labels=['niższa','wyższa'])
df_gosp['posval_bin'] = df_gosp['home_pos_bin'].astype(str) + '-' + df_gosp['home_value_bin'].astype(str)

# 2. Tworzymy tabelę krzyżową: procent zwycięstw gospodarza w każdej kombinacji
tab_gosp = pd.crosstab(
    df_gosp['posval_bin'],
    df_gosp['home_form'],
    values=(df_gosp['home_result'] == 'win').astype(int),
    aggfunc='mean'
).loc[:, ['bardzo zła', 'zła', 'dobra', 'średnia', 'bardzo dobra']]
tab_gosp = tab_gosp.reindex(index=posval_kolejnosc, columns=forma_kolejnosc) 

sns.heatmap(tab_gosp, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Procent zwycięstw GOSPODARZA: pozycja, wartość, forma")
plt.xlabel("Forma gospodarza")
plt.ylabel("Pozycja-wartość gospodarza")
plt.savefig("heatmapa_win_gospodarz.png", dpi=150, bbox_inches='tight')
plt.show()


form_map = {
    'bardzo zła': -2,
    'zła': -1,
    'średnia': 0,
    'dobra': 1,
    'bardzo dobra': 2
}
df_gosc = df[~df['away_form'].isin(['brak formy'])].copy()
df_gosc['away_form_num'] = df_gosc['away_form'].map(form_map)
df_gosc['away_pos_bin'] = pd.cut(df_gosc['away_club_position'], [0,5,10,15,20], labels=['1-5','6-10','11-15','16-20'])
df_gosc['away_value_bin'] = pd.qcut(df_gosc['squad_value_away'], 2, labels=['niższa','wyższa'])
df_gosc['posval_bin'] = df_gosc['away_pos_bin'].astype(str) + '-' + df_gosc['away_value_bin'].astype(str)

# Teraz twórz crosstab już na tej kolumnie!
tab_gosc = pd.crosstab(
    df_gosc['posval_bin'],
    df_gosc['away_form'],
    values=(df_gosc['away_result']=='win').astype(int),
    aggfunc='mean'
).loc[:, ['bardzo zła','zła','dobra','średnia','bardzo dobra']]
tab_gosc = tab_gosc.reindex(index=posval_kolejnosc, columns=forma_kolejnosc)

sns.heatmap(tab_gosc, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Procent zwycięstw GOŚCIA: pozycja, wartość, forma")
plt.xlabel("Forma gościa")
plt.ylabel("Pozycja-wartość gościa")
plt.savefig("heatmapa_win_gosc.png", dpi=150, bbox_inches='tight')
plt.show()

home = df[['home_club_position', 'squad_value_home', 'home_result']].copy()
home['side'] = 'home'
home.columns = ['position', 'squad_value', 'result', 'side']

away = df[['away_club_position', 'squad_value_away', 'away_result']].copy()
away['side'] = 'away'
away.columns = ['position', 'squad_value', 'result', 'side']

long_df = pd.concat([home, away], ignore_index=True)

# Klasyfikacja pozycji na kategorie (piątki/kroki po 5)
long_df['pos_bin'] = pd.cut(long_df['position'], [0,5,10,15,20], labels=['1-5','6-10','11-15','16-20'])
long_df['value_bin'] = pd.qcut(long_df['squad_value'], 4, labels=['najsłabszy', 'średni-', 'średni+', 'najbogatszy'])

# Oznaczamy kolumnę wygranej (0/1)
long_df['is_win'] = (long_df['result'] == 'win').astype(int)


# --- WYKRES SŁUPKOWY po pozycji ---
pos_win = long_df.groupby('pos_bin')['is_win'].mean() * 100
pos_win.plot(kind='bar')
plt.ylabel('Procent wygranych [%]')
plt.title('Procent wygranych wg pozycji w tabeli')
plt.show()

# --- WYKRES SŁUPKOWY po wartości składu ---
val_win = long_df.groupby('value_bin')['is_win'].mean() * 100
val_win.plot(kind='bar')
plt.ylabel('Procent wygranych [%]')
plt.title('Procent wygranych wg kwartylu wartości składu')
plt.show()

# --- Wykres boxplot pozycji a wygrana ---
sns.boxplot(x='result', y='position', data=long_df)
plt.title('Pozycja w tabeli a wynik meczu (ogółem)')
plt.show()

# --- Wykres boxplot wartości składu a wygrana ---
sns.boxplot(x='result', y='squad_value', data=long_df)
plt.title('Wartość składu a wynik meczu (ogółem)')
plt.show()

# --- Wykres udziału wygranych dla “najsłabszy"/"najbogatszy” po stronie gry (opcjonalnie) ---
sns.barplot(x='value_bin', y='is_win', data=long_df, hue='side')
plt.ylabel('Procent wygranych')
plt.title('Wygrane % wg wartości składu i miejsca gry')
plt.show()







df_multi = df[~df['home_form'].isin(['brak formy'])]
# kategorie
df_multi['pos_bin'] = pd.cut(df_multi['home_club_position'], [0,5,10,15,20], labels=['1-5','6-10','11-15','16-20'])
df_multi['value_bin'] = pd.qcut(df_multi['squad_value_home'], 2, labels=['niższa','wyższa'])

# Przykład: Tabela krzyżowa - 3 kategorie
pivot = pd.crosstab([df_multi['pos_bin'], df_multi['value_bin'], df_multi['home_form']], df_multi['home_result'], normalize='index')
print(pivot)


win_pivot = pivot['win'].unstack().fillna(0)
sns.heatmap(win_pivot, annot=True, cmap="YlGnBu")
plt.title("Procent zwycięstw: pozycja, wartość, forma")
plt.savefig("heatmapa_win_vs_pozycja_value_forma.png", dpi=150, bbox_inches='tight')
plt.show()

home = df[['home_club_position','squad_value_home','home_form','home_club_goals']].copy()
home.columns = ['position', 'value', 'form', 'goals']
home['side'] = 'home'

# AWAY
away = df[['away_club_position','squad_value_away','away_form','away_club_goals']].copy()
away.columns = ['position', 'value', 'form', 'goals']
away['side'] = 'away'

# Łączymy
long_df = pd.concat([home, away], ignore_index=True)

long_df['pos_bin'] = pd.cut(long_df['position'], [0,5,10,15,20], labels=['1-5','6-10','11-15','16-20'])
long_df['value_bin'] = pd.qcut(long_df['value'], 4, labels=['najsłabszy', 'średni-', 'średni+', 'najbogatszy'])

long_df = long_df[~long_df['form'].isin(['brak formy'])]

gole_heat_both = pd.pivot_table(
    long_df, 
    values='goals', 
    index='pos_bin', 
    columns='value_bin', 
    aggfunc='mean'
)
sns.heatmap(gole_heat_both, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Średnia liczba goli drużyn (gospodarz + gość): pozycja vs wartość składu")
plt.xlabel("Wartość składu (kwartyle)")
plt.ylabel("Pozycja w tabeli")
plt.savefig("heat_gole_both_pozycja_wartosc.png", dpi=150, bbox_inches='tight')
plt.show()



# Jeśli chcesz forma vs wartość:
gole_heat_forma = pd.pivot_table(
    long_df, 
    values='goals', 
    index='value_bin', 
    columns='form', 
    aggfunc='mean'
)
gole_heat_forma = gole_heat_forma[['bardzo zła','zła','średnia','dobra','bardzo dobra']]
sns.heatmap(gole_heat_forma, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Średnia liczba goli drużyn: wartość składu vs forma")
plt.xlabel("Forma")
plt.ylabel("Wartość składu (kwartyle)")
plt.savefig("heat_gole_both_value_forma.png", dpi=150, bbox_inches='tight')
plt.show()




# HOME
home = df[['home_club_position', 'squad_value_home', 'home_form', 'home_club_goals']].copy()
home.columns = ['position', 'value', 'form', 'goals']
home['side'] = 'home'

# AWAY
away = df[['away_club_position', 'squad_value_away', 'away_form', 'away_club_goals']].copy()
away.columns = ['position', 'value', 'form', 'goals']
away['side'] = 'away'

# Łącz w jedną tabelę
both = pd.concat([home, away], ignore_index=True)


# HOME
home = df[['home_club_position', 'squad_value_home', 'home_form', 'home_club_goals']].copy()
home.columns = ['position', 'value', 'form', 'goals']
home['side'] = 'home'

# AWAY
away = df[['away_club_position', 'squad_value_away', 'away_form', 'away_club_goals']].copy()
away.columns = ['position', 'value', 'form', 'goals']
away['side'] = 'away'

home['pos_bin'] = pd.cut(home['position'], [0,5,10,15,20], labels=['1-5','6-10','11-15','16-20'])
home['value_bin'] = pd.qcut(home['value'], 4, labels=['najsłabszy', 'średni-', 'średni+', 'najbogatszy'])

away['pos_bin'] = pd.cut(away['position'], [0,5,10,15,20], labels=['1-5','6-10','11-15','16-20'])
away['value_bin'] = pd.qcut(away['value'], 4, labels=['najsłabszy', 'średni-', 'średni+', 'najbogatszy'])


# Łącz w jedną tabelę
both = pd.concat([home, away], ignore_index=True)


both['triple_bin'] = both['pos_bin'].astype(str) + '_' + both['value_bin'].astype(str) + '_' + both['form'].astype(str)

plt.figure(figsize=(20,5))
sns.boxplot(x='triple_bin', y='goals', data=both,
    order=sorted(both['triple_bin'].unique(), 
        key=lambda x: (
            x.split('_')[0], x.split('_')[1], x.split('_')[2]
        )
    )
)
plt.title("Liczba goli drużyn (gospodarz + gość): pozycja/wartość/formą")
plt.ylabel("Gole")
plt.xlabel("Pozycja_Wartość_Forma")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("boxplot_gole_both_allbins.png", dpi=150)
plt.show()




