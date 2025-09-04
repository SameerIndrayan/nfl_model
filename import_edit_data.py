import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Read the original data
nfl_df = pd.read_csv('nfl_gamelog_vegas_2015-2024.csv')

# Use a single dataframe for all modifications
# Filter for home games first, as that's what you want for the model
nfl_df_mod = nfl_df.query('Home==1').reset_index(drop=True)

# Now, perform all the calculations on this filtered dataframe
nfl_df_mod['True_Total'] = nfl_df_mod['Tm_Pts'] + nfl_df_mod['Opp_Pts']
nfl_df_mod['Over'] = np.where(nfl_df_mod['True_Total'] > nfl_df_mod['Total'], 1, 0)
nfl_df_mod['Under'] = np.where(nfl_df_mod['True_Total'] < nfl_df_mod['Total'], 1, 0)
nfl_df_mod['Push'] = np.where(nfl_df_mod['True_Total'] == nfl_df_mod['Total'], 1, 0)

# Sort the dataframe as intended
nfl_df_mod = nfl_df_mod.sort_values(by=['Season', 'Week']).reset_index(drop=True)

# You can save this to a new CSV if you want, but it's not necessary for the code to run
# nfl_df_mod.to_csv("nfl_gamelogs_model_version.csv", index=False)

features = ['Spread', 'Total']
target = 'Under'

# 17 game seasons
for season in [2021, 2022, 2023, 2024]:
    print(f'\nResults for {season}:')
    
    # Season aggregates
    y_preds = []
    y_trues = []

    for week in range(1, 19):
        print(f' Week{week:>2}', end=' ')

        # training set: all data from before the current week
        train_df = nfl_df_mod.query('Season < @season or (Season == @season and Week < @week)')

        # testing set: the current week's data, excluding pushes
        test_df = nfl_df_mod.query('Season == @season and Week == @week and Push == 0')
        
        # Check if test_df is not empty before proceeding
        if test_df.empty:
            continue

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        model = KNeighborsClassifier(n_neighbors=7)
        
        # Train model
        clf = model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = clf.predict(X_test)

        # Append to aggregates
        y_true = y_test
        print(f'accuracy score={accuracy_score(y_true, y_pred):.2%}')
        y_preds +=list (y_pred)
        y_trues +=list (y_true)

    # After the inner loop completes for all weeks of a season, print the final results
    if len(y_trues) > 0:
        print(f'\nAccuracy score={accuracy_score(y_trues, y_preds):.2%}')

        print(f'\nClassification Report for {season}:')
        print(classification_report(y_trues, y_preds, target_names=['Over', 'Under']))

        # cm for season
        cm = confusion_matrix(y_trues, y_preds)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Over', 'Under'])
        display.plot()
        plt.grid(False)
        plt.title(f'Confusion Matrix for {season}')
        plt.show()

    # Iterate through the test instances and find their nearest neighbors
index = 0
for game_index, spread, total in X_test.itertuples():
# Get the neighbor distances and indices in the dataframe
    nbr_distance = clf.kneighbors(X_test)[0][index]
    nbr_index = clf.kneighbors(X_test)[1][index]
    index += 1
    # Display the games, the distances, and the neighbors
    print(f'GAME')
    print(nfl_df_mod.iloc[[game_index],:][['Season','Week','Team','Opp_x','Spread','Total','True_Total','Under']])
    print(f'\nNEAREST NEIGHBORS (distances={[round(value, 2) for value in nbr_distance]})')
    print(nfl_df_mod.iloc[nbr_index,:][['Season','Week','Team','Opp_x','Spread','Total','True_Total','Under']])
    print('\n')

# Make predictions for NFL Totals (Season = 2025, Week = 1)

# Features and target
features = ['Spread', 'Total']
target = 'Under'

# Set the season and the week
season = 2025
week = 1

# Training data (all past games before Week 1 of 2025)
train_df = nfl_df_mod.query('Season < @season or (Season == @season and Week < @week)')
X_train = train_df[features]
y_train = train_df[target]

# Upcoming Week 1 games (home team's perspective)
week1 = [
    ['Cowboys @ Eagles', -8.5, 48.5],
    ['Chiefs @ Chargers', 3.5, 46.5],
    ['49ers @ Seahawks', 2.5, 43.5],
    ['Lions @ Packers', -1.5, 47.5],
    ['Ravens @ Bills',  1.5, 50.5],
]

# Create DataFrame for new games
X_new = pd.DataFrame(week1, columns=['Game', 'Spread', 'Total'])

# Train model
model = KNeighborsClassifier(n_neighbors=7)
clf = model.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_new[features])

# Add predictions to the DataFrame
X_new['KNC(7)'] = y_pred
X_new['KNC(7)'] = X_new['KNC(7)'].apply(lambda x: 'Under' if x == 1 else 'Over')

# Display results
print(f'\nMODEL PREDICTIONS FOR WEEK {week} OF THE {season} NFL SEASON\n')
print(X_new[['Game', 'Spread', 'Total', 'KNC(7)']])
