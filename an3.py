import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("games_with_stats.csv")

# Usuń %
df['possession_home'] = df['possession_home'].str.replace('%','').astype(float)
df['possession_away'] = df['possession_away'].str.replace('%','').astype(float)

# Rób "long format":
home = df[['possession_home', 'xg_home', 'home_club_goals', 'home_result']].copy()
home['side'] = 'home'
home.columns = ['possession', 'xG', 'goals', 'result', 'side']

away = df[['possession_away', 'xg_away', 'away_club_goals', 'away_result']].copy()
away['side'] = 'away'
away.columns = ['possession', 'xG', 'goals', 'result', 'side']

long_df = pd.concat([home, away], ignore_index=True)

print(long_df[['possession', 'xG', 'goals']].corr())

# --- WYKRESY podstawowe (z przezroczystymi kropkami i grubą linią) ---

sns.lmplot(
    x='possession', y='xG', data=long_df,
    scatter_kws={'alpha':0.08},
    line_kws={'color':'red','lw':3}
)
plt.title("Posiadanie piłki a xG (pojedyncza drużyna)")
plt.show()

sns.lmplot(
    x='xG', y='goals', data=long_df,
    scatter_kws={'alpha':0.08},
    line_kws={'color':'red','lw':3}
)
plt.title("xG a liczba goli")
plt.show()

sns.lmplot(
    x='possession', y='goals', data=long_df,
    scatter_kws={'alpha':0.08},
    line_kws={'color':'red','lw':3}
)
plt.title("Posiadanie piłki a gole")
plt.show()

# --- Boxploty ---
sns.boxplot(x='result', y='xG', data=long_df)
plt.title("xG a wynik meczu (pojedyncza drużyna)")
plt.show()

sns.boxplot(x='result', y='possession', data=long_df)
plt.title("Posiadanie a wynik meczu (pojedyncza drużyna)")
plt.show()

# --- REGRESJE ---
X = long_df[['possession', 'xG']]
X = sm.add_constant(X)
y = long_df['goals']
model = sm.OLS(y, X).fit()
print(model.summary())

long_df['is_win'] = (long_df['result']=='win').astype(int)
X2 = sm.add_constant(long_df[['possession', 'xG']])
y2 = long_df['is_win']
logit = sm.Logit(y2, X2).fit()
print(logit.summary())

# --- HEATMAPA xG & possession vs liczba goli ---
long_df['xG_bin'] = pd.qcut(long_df['xG'], 5, labels=[f'{i+1}.Q' for i in range(5)])
long_df['poss_bin'] = pd.qcut(long_df['possession'], 5, labels=[f'{i+1}.Q' for i in range(5)])

pivot = long_df.pivot_table(
    values='goals',
    index='xG_bin',
    columns='poss_bin',
    aggfunc='mean'
)

plt.figure(figsize=(7,6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title("Średnia liczba goli:\npoziomy xG & posiadania piłki")
plt.ylabel('xG (kwantyle)')
plt.xlabel('Posiadanie (kwantyle)')
plt.tight_layout()
plt.show()

# --- HEATMAPA xG & possession vs P(wygranej) ---
pivot2 = long_df.pivot_table(
    values='is_win',
    index='xG_bin',
    columns='poss_bin',
    aggfunc='mean'
)

plt.figure(figsize=(7,6))
sns.heatmap(pivot2, annot=True, fmt='.2%', cmap='YlGnBu')
plt.title("Prawdopodobieństwo wygranej:\npoziomy xG & posiadania piłki")
plt.ylabel('xG (kwantyle)')
plt.xlabel('Posiadanie (kwantyle)')
plt.tight_layout()
plt.show()

# --- Siatka predykcji OLS ---
xg_space = np.linspace(long_df['xG'].min(), long_df['xG'].max(), 20)
poss_space = np.linspace(long_df['possession'].min(), long_df['possession'].max(), 20)

xx, yy = np.meshgrid(xg_space, poss_space)
X_pred = pd.DataFrame({'const':1, 'possession':yy.ravel(), 'xG':xx.ravel()})
zz = model.predict(X_pred).values.reshape(xx.shape)

plt.figure(figsize=(7,5))
cp = plt.contourf(xx, yy, zz, 20, cmap='YlOrRd')
plt.xlabel('xG')
plt.ylabel('possession')
plt.title('Prognozowana liczba goli')
plt.colorbar(cp)
plt.show()