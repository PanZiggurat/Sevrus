import pandas as pd
import numpy as np

df = pd.read_csv('games_with_stats.csv')

# Przygotowanie procentów
df['possession_home'] = df['possession_home'].str.replace('%','').astype(float)
df['possession_away'] = df['possession_away'].str.replace('%','').astype(float)

# Najprostsze: value_ratio tylko dla home, bo od razu w pliku

# Zakładam, że masz w danych kolumny: 
# home_club_position, away_club_position, squad_value_home, squad_value_away, 
# xg_home, xg_away, home_club_goals, away_club_goals, home_result, away_result,
# home_form, away_form, yellow_cards_home/away, red_cards_home/away

# HOME
home = df[[
    'possession_home', 'xg_home', 'home_club_goals', 'home_result',
    'home_club_position', 'squad_value_home', 'home_form', 
    'yellow_cards_home', 'red_cards_home'
]].copy()
home.columns = [
    'possession', 'xG', 'goals', 'result',
    'position', 'squad_value', 'form',
    'yellow_cards', 'red_cards'
]
home['side'] = 'home'

# AWAY
away = df[[
    'possession_away', 'xg_away', 'away_club_goals', 'away_result',
    'away_club_position', 'squad_value_away', 'away_form',
    'yellow_cards_away', 'red_cards_away'
]].copy()
away.columns = [
    'possession', 'xG', 'goals', 'result',
    'position', 'squad_value', 'form',
    'yellow_cards', 'red_cards'
]
away['side'] = 'away'

long_df = pd.concat([home, away], ignore_index=True)

# Pozycja w piątkach
long_df['pos_bin'] = pd.cut(long_df['position'], [0,5,10,15,20], labels=['1-5','6-10','11-15','16-20'])
# Value w kwartylach
long_df['value_bin'] = pd.qcut(long_df['squad_value'], 4, labels=['najsłabszy', 'średni-', 'średni+', 'najbogatszy'])
# xG w kwantylach
long_df['xG_bin'] = pd.qcut(long_df['xG'], 4, labels=[f'{i+1}.Q' for i in range(4)])
# Posiadanie w kwantylach
long_df['poss_bin'] = pd.qcut(long_df['possession'], 4, labels=[f'{i+1}.Q' for i in range(4)])

# Mapa formy numerycznej
form_map = {'bardzo zła': -2, 'zła': -1, 'średnia': 0, 'dobra': 1, 'bardzo dobra': 2}
long_df['form_num'] = long_df['form'].map(form_map)

long_df['is_win'] = (long_df['result'] == 'win').astype(int)

df['value_ratio'] = df['squad_value_home'] / df['squad_value_away']
df['value_ratio_bin'] = pd.cut(df['value_ratio'], [0,0.8,1.2,2,10], labels=['Zdecydowanie słabszy','Podobny','Mocniejszy do 2x','>2x mocniejszy'])

# Przenieś do long_df:
long_df['value_ratio'] = np.nan
long_df.loc[long_df['side']=='home', 'value_ratio'] = df['value_ratio'].values
long_df['value_ratio_bin'] = np.nan
long_df.loc[long_df['side']=='home', 'value_ratio_bin'] = df['value_ratio_bin'].values

import seaborn as sns
import matplotlib.pyplot as plt

# Tabela krzyżowa: xG_bin vs value_bin
t = long_df.pivot_table(
    values='is_win',
    index='xG_bin', columns='value_bin',
    aggfunc='mean'
)



plt.figure(figsize=(7,5))
sns.heatmap(t, annot=True, fmt='.2%', cmap='YlGnBu')
plt.title('Prawdopodobieństwo wygranej: xG i wartość składu')
plt.show()

t1 = long_df.pivot_table(
    values='is_win',
    index='xG_bin', columns='pos_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(t1, annot=True, fmt='.2%', cmap='Blues')
plt.title('Prawdopodobieństwo wygranej: xG & pozycja')
plt.show()


t2 = long_df.pivot_table(
    values='is_win',
    index='xG_bin', columns='value_ratio_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(t2, annot=True, fmt='.2%', cmap='PuBu')
plt.title('Prawdopodobieństwo wygranej : xG & value_ratio')
plt.show()


t3 = long_df.pivot_table(
    values='is_win',
    index='value_bin', columns='poss_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(t3, annot=True, fmt='.2%', cmap='YlGn')
plt.title('Prawdopodobieństwo wygranej: wartość składu & posiadanie')
plt.show()

tt = long_df.pivot_table(
    values='is_win',
    index='pos_bin', columns='poss_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(tt, annot=True, fmt='.2%', cmap='YlOrRd')
plt.title('Prawdopodobieństwo wygranej: pozycja vs posiadanie')
plt.show()

t4 = long_df.pivot_table(
    values='is_win',
    index='value_bin', columns='pos_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(t4, annot=True, fmt='.2%', cmap='YlOrBr')
plt.title('Prawdopodobieństwo wygranej: wartość składu & pozycja')
plt.show()

# Procent zwycięstw po wartości, xG i formie (trzy kategorie naraz!)
ttt = long_df.pivot_table(
    values='is_win', 
    index=['value_bin', 'pos_bin'],
    columns='xG_bin', 
    aggfunc='mean'
)
plt.figure(figsize=(12,8))
sns.heatmap(ttt, annot=True, fmt='.2%', cmap='crest')
plt.title('Prawdopodobieństwo wygranej: value_bin/pozycja/xG')
plt.show()

pivot_goals = long_df.pivot_table(
    values='goals',
    index='xG_bin',
    columns='value_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(pivot_goals, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Średnia liczba goli: xG & wartość składu')
plt.show()

import statsmodels.api as sm
features = ['possession', 'xG', 'position', 'squad_value', 'form_num', 'yellow_cards', 'red_cards']
X = long_df[features].fillna(0)
X = sm.add_constant(X)
y = long_df['is_win']
model = sm.Logit(y, X).fit()
print(model.summary())


df['xg_diff'] = df['xg_home'] - df['xg_away']
df['value_diff'] = df['squad_value_home'] - df['squad_value_away']
df['pos_diff'] = df['away_club_position'] - df['home_club_position']


tg = long_df[long_df['side']=='home'].pivot_table(
    values='is_win',
    index='xG_bin', columns='value_bin',
    aggfunc='mean'
)



plt.figure(figsize=(7,5))
sns.heatmap(tg, annot=True, fmt='.2%', cmap='YlGnBu')
plt.title('Prawdopodobieństwo wygranej gosp: xG i wartość składu')
plt.show()

tg1 = long_df[long_df['side']=='home'].pivot_table(
    values='is_win',
    index='xG_bin', columns='pos_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(tg1, annot=True, fmt='.2%', cmap='Blues')
plt.title('Prawdopodobieństwo wygranej gosp: xG & pozycja')
plt.show()


tg2 = long_df[long_df['side']=='home'].pivot_table(
    values='is_win',
    index='xG_bin', columns='value_ratio_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(tg2, annot=True, fmt='.2%', cmap='PuBu')
plt.title('Prawdopodobieństwo wygranej gosp: xG & value_ratio')
plt.show()


tg3 = long_df[long_df['side']=='home'].pivot_table(
    values='is_win',
    index='value_bin', columns='poss_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(tg3, annot=True, fmt='.2%', cmap='YlGn')
plt.title('Prawdopodobieństwo wygranej gosp: wartość składu & posiadanie')
plt.show()

ttg = long_df[long_df['side']=='home'].pivot_table(
    values='is_win',
    index='pos_bin', columns='poss_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(ttg, annot=True, fmt='.2%', cmap='YlOrRd')
plt.title('Prawdopodobieństwo wygranej gosp: pozycja vs posiadanie')
plt.show()

tg4 = long_df[long_df['side']=='home'].pivot_table(
    values='is_win',
    index='value_bin', columns='pos_bin',
    aggfunc='mean'
)
plt.figure(figsize=(7,5))
sns.heatmap(tg4, annot=True, fmt='.2%', cmap='YlOrBr')
plt.title('Prawdopodobieństwo wygranej gosp: wartość składu & pozycja')
plt.show()



wyniki_value = (
    long_df
    .groupby(['xG_bin', 'value_bin'])['result']
    .value_counts(normalize=True)
    .rename('fraction')
    .reset_index()
    .pivot_table(index=['xG_bin', 'value_bin'], columns='result', values='fraction', fill_value=0)
    .reset_index()
)

# Dodaj jedną kolumnę na etykietę (np. '3.Q-najbogatszy')
wyniki_value['label'] = wyniki_value['xG_bin'].astype(str) + "-" + wyniki_value['value_bin'].astype(str)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16,6))
wyniki_value.set_index('label')[['win', 'draw', 'lose']].plot(
    kind='bar', 
    stacked=True, 
    color=['#417504', '#FDDD03', '#E02125'],  # kolory: zielony, żółty, czerwony (lub dowolne)
    ax=ax
)
plt.ylabel('Udział [%]')
plt.title('Rozkład wyników w zależności od xG & wartość składu')
plt.legend(title='Wynik', loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

wyniki_pos = (
    long_df
    .groupby(['xG_bin', 'pos_bin'])['result']
    .value_counts(normalize=True)
    .rename('fraction')
    .reset_index()
    .pivot_table(index=['xG_bin', 'pos_bin'], columns='result', values='fraction', fill_value=0)
    .reset_index()
)
wyniki_pos['label'] = wyniki_pos['xG_bin'].astype(str) + "-" + wyniki_pos['pos_bin'].astype(str)

fig, ax = plt.subplots(figsize=(16,6))
wyniki_pos.set_index('label')[['win', 'draw', 'lose']].plot(
    kind='bar',
    stacked=True,
    color=['#417504', '#FDDD03', '#E02125'],
    ax=ax
)
plt.ylabel('Udział [%]')
plt.title('Rozkład wyników w zależności od xG & pozycja')
plt.legend(title='Wynik', loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



long_home = long_df[long_df['side']=='home']

wyniki_value_home = (
    long_home
    .groupby(['xG_bin', 'value_bin'])['result']
    .value_counts(normalize=True)
    .rename('fraction')
    .reset_index()
    .pivot_table(index=['xG_bin', 'value_bin'], columns='result', values='fraction', fill_value=0)
    .reset_index()
)
wyniki_value_home['label'] = wyniki_value_home['xG_bin'].astype(str) + "-" + wyniki_value_home['value_bin'].astype(str)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16,6))
wyniki_value_home.set_index('label')[['win', 'draw', 'lose']].plot(
    kind='bar',
    stacked=True,
    color=['#417504', '#FDDD03', '#E02125'],  # zielony, żółty, czerwony
    ax=ax
)
plt.ylabel('Udział [%]')
plt.title('Rozkład wyników GOSPODARZA w zależności od xG & wartość składu')
plt.legend(title='Wynik', loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

wyniki_pos_home = (
    long_home
    .groupby(['xG_bin', 'pos_bin'])['result']
    .value_counts(normalize=True)
    .rename('fraction')
    .reset_index()
    .pivot_table(index=['xG_bin', 'pos_bin'], columns='result', values='fraction', fill_value=0)
    .reset_index()
)
wyniki_pos_home['label'] = wyniki_pos_home['xG_bin'].astype(str) + "-" + wyniki_pos_home['pos_bin'].astype(str)

fig, ax = plt.subplots(figsize=(16,6))
wyniki_pos_home.set_index('label')[['win', 'draw', 'lose']].plot(
    kind='bar',
    stacked=True,
    color=['#417504', '#FDDD03', '#E02125'],
    ax=ax
)
plt.ylabel('Udział [%]')
plt.title('Rozkład wyników GOSPODARZA w zależności od xG & pozycja')
plt.legend(title='Wynik', loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()