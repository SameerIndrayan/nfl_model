import numpy as np
import pandas as pd

# used python 3.10 bc 3.13 doesnt work with matplotlib and sklearn
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

nfl_df = pd.read_csv('nfl_gamelog_vegas_2015-2024.csv')

# copy
nfl_df_mod = nfl_df.copy()

# abb_to_name = {'CRD': 'Cardinals', 'ATL': 'Falcons', 'RAV': 'Ravens', 'BUF':'Bills', 'CAR': "Panthers", 'CHI': 'Bears', 'CIN': 'Bengals', 'CLE': 'Browns', 'DAL': 'Cowboys', 'DEN': 'Broncos', 'DET': 'Lions', 'GNB': 'Packers', 'HTX': 'Texans', 'CLT': 'Colts', 'JAX': 'Jaguars', 'KAN': 'Chiefs', 'SDG': 'Chargers', 'RAM': 'Rams', 'RAI': 'Raiders', 'MIA': 'Dolphins', 'MIN': 'Vikings', 'NWE': 'Patriots', 'NOR':'Saints', 'NYG': 'Giants', 'NYJ': 'Jets', 'PHI':'Eagles', 'PIT':'Steelers', 'SEA':'Seahawks', 'SFO': '49ers', 'TAM':'Buccaneers', 'OTI':'Titans', 'WAS':'Commanders'}

nfl_df_mod['True_Total'] = nfl_df_mod['Tm_Pts'] + nfl_df_mod['Opp_Pts']
nfl_df_mod['Over'] = np.where(nfl_df_mod['True_Total'] > nfl_df_mod['Total'], 1, 0)
nfl_df_mod['Under'] = np.where(nfl_df_mod['True_Total'] < nfl_df_mod['Total'], 1, 0)
nfl_df_mod['Push'] = np.where(nfl_df_mod['True_Total'] == nfl_df_mod['Total'], 1, 0)
# nfl_df_mod["Name"] = nfl_df_mod["Team"].map(abb_to_name)
# nfl_df_mod["Opp_x"] = nfl_df_mod["Opponent"].map(abb_to_name)


# nfl_df_mod.to_csv("nfl_gamelogs_model_version.csv", index = False)
# print(nfl_df_mod)

nfl_df_mod = nfl_df_mod.sort_values(by=['Season', 'Week']).reset_index(drop=True)

# set df (all homegames, bc rn each game has 2 rows and we will cut it into 1)
nfl_df_mod = nfl_df.query('Home ==1').reset_index(drop=True)
nfl_df_mod.to_csv("nfl_gamelogs_model_version.csv", index = False)

