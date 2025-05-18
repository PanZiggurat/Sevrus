import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('games_with_stats.csv')

# Ramka long
home = df[['xg_home', 'xg_away', 'home_result']].copy()
home['my_xg'] = home['xg_home']
home['opp_xg'] = home['xg_away']
home['result'] = home['home_result']
home['side'] = 'home'

away = df[['xg_away', 'xg_home', 'away_result']].copy()
away['my_xg'] = away['xg_away']
away['opp_xg'] = away['xg_home']
away['result'] = away['away_result']
away['side'] = 'away'

long_xg = pd.concat([home, away], ignore_index=True)
long_xg['xg_diff'] = long_xg['my_xg'] - long_xg['opp_xg']
long_xg['is_win'] = (long_xg['result'] == 'win').astype(int)

# Binning przewagi xG (możesz zmienić szerokość do swojego rozkładu)
long_xg['xg_diff_bin'] = pd.cut(long_xg['xg_diff'], bins=np.arange(-5, 5.5, 0.5))
winrate = long_xg.groupby('xg_diff_bin')['is_win'].mean()

plt.figure(figsize=(10,5))
winrate.plot(kind='bar')
plt.ylabel('Prawdopodobieństwo wygranej drużyny')
plt.xlabel('Przewaga xG nad przeciwnikiem')
plt.title('Szansa wygranej a przewaga xG (obie drużyny razem)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()