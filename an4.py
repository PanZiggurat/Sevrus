import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('games_with_stats.csv')

# Usuń % z posiadania
df['possession_home'] = df['possession_home'].str.replace('%','').astype(float)
df['possession_away'] = df['possession_away'].str.replace('%','').astype(float)

# Budujemy ramkę "long": po jednej drużynie na wiersz!
home = df[['possession_home', 'yellow_cards_home', 'red_cards_home', 'home_result']].copy()
home.columns = ['possession', 'yellow_cards', 'red_cards', 'result']
home['side'] = 'home'

away = df[['possession_away', 'yellow_cards_away', 'red_cards_away', 'away_result']].copy()
away.columns = ['possession', 'yellow_cards', 'red_cards', 'result']
away['side'] = 'away'

long_df = pd.concat([home, away], ignore_index=True)

import seaborn as sns

# Rysujemy boxplot: liczba żółtych kartek a wynik meczu
sns.boxplot(x='result', y='yellow_cards', data=long_df)
plt.title("Żółte kartki a wynik meczu")
plt.show()

sns.boxplot(x='result', y='red_cards', data=long_df)
plt.title("Czerwone kartki a wynik meczu")
plt.show()

sns.scatterplot(x='possession', y='yellow_cards', data=long_df, alpha=0.2)
plt.title("Posiadanie a żółte kartki")
plt.show()

sns.scatterplot(x='possession', y='red_cards', data=long_df, alpha=0.2)
plt.title("Posiadanie a czerwone kartki")
plt.show()

print("Korelacja żółte vs czerwone:", long_df[['yellow_cards','red_cards']].corr())

import statsmodels.api as sm
long_df['is_win'] = (long_df['result'] == 'win').astype(int)
X = long_df[['yellow_cards', 'red_cards', 'possession']]
X = sm.add_constant(X)
y = long_df['is_win']
model = sm.Logit(y, X).fit()
print(model.summary())