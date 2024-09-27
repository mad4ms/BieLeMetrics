# load all *.csv files in the data/processed

import pandas as pd
from glob import glob
import os
from imblearn.under_sampling import RandomUnderSampler

# train test split
from sklearn.model_selection import train_test_split

from supervised.automl import AutoML
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# Load all CSV files in the data/processed directory
list_files = glob("data/processed/**/*.csv", recursive=True)

# Load all CSV files into a dictionary of DataFrames
dfs = {file: pd.read_csv(file) for file in list_files}

# Print the first 5 rows of each DataFrame
# for file, df in dfs.items():
#     print(f"\n{file}")
#     print(df.head())

# return {
#     "distance_player_to_goal": self.distance_player_to_goal,
#     "distance_player_to_goalkeeper": self.distance_player_to_goalkeeper,
#     "distance_player_to_blocker": self.distance_player_to_blocker,
#     "distance_player_to_nearst_opponent": self.distance_player_to_nearst_opponent,
#     "id_nearst_opponent": self.id_nearst_opponent,
#     "distance_player_to_nearest_teammate": self.distance_player_to_nearest_teammate,
#     "id_nearest_teammate": self.id_nearest_teammate,
#     "distance_goalkeeper_to_goal": self.distance_goalkeeper_to_goal,
#     "angle_player_to_goal": self.angle_player_to_goal,
#     "angle_ball_to_goal": self.angle_ball_to_goal,
#     "num_opponents_between_player_and_goal": self.num_opponents_between_player_and_goal,
#     "num_opponents_close_to_player": self.num_opponents_close_to_player,
#     "home_advantage": self.home_advantage,
#     "speed_player": self.speed_player,
#     "speed_ball": self.speed_ball,
# }


features = [
    "distance_player_to_goal",  # distance between player and the goal
    "distance_player_to_goalkeeper",  # distance between player and goalkeeper
    "distance_goalkeeper_to_goal",  # distance between goalkeeper and goal
    "angle_player_to_goal",  # angle of the ball relative to the goal
    # "angle_ball_to_goal",
    # "speed_ball",  # speed of the ball when thrown
    "speed_player",  # speed of the player
    "distance_player_to_nearst_opponent",  # distance to the nearest defender
    "distance_player_to_nearest_teammate",  # number of defenders close to the player
    # "num_opponents_between_player_and_goal",  # number of defenders between player and goal
    "num_opponents_close_to_player",  # number of defenders close to the player
    "efficiency_shots_team",  # efficiency of the team in scoring goals
    "efficiency_shots_player",  # efficiency of the player in scoring goals
    "efficiency_goalkeeper",  # efficiency of the goalkeeper in saving goals
    # "home_advantage",  # home advantage
]

target_column = "event_type"  # target column to predict

# Combine all DataFrames into a single DataFrame
df = pd.concat(dfs.values(), ignore_index=True)

# Filter for relevant events
relevant_event_types = [
    "score_change",
    "shot_off_target",
    "shot_blocked",
    "shot_saved",
    "seven_m_missed",
]
print()
df = df[df["event_type"].isin(relevant_event_types)]

# drop nan values
df = df.dropna(subset=features + [target_column])

# print length
print(f"0. Length of DataFrame: {len(df)}")

# "distance_player_goal" must be < 20m
df = df[df["distance_player_to_goal"] < 20]

print(f"1. Length of DataFrame: {len(df)}")

# "distance_goalkeeper_goal" must be < 20m
df = df[df["distance_goalkeeper_to_goal"] < 20]

# print length
print(f"2. Length of DataFrame after filtering: {len(df)}")

# Select the features and target column
X = df[features]
y = df[target_column]


# make y = 1 when event_type is 'score_changed', 0 otherwise
y = y.apply(lambda x: 1 if x == "score_change" else 0)
# print distribution of target column
print(y.value_counts())

sampling_strategy = 1
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_res, y_res = rus.fit_resample(X, y)

# X_res = X
# y_res = y

# print distribution of target column
print(y_res.value_counts())

# Print the shape of the combined DataFrame
print(f"\nCombined DataFrame shape: {df.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# potential output: ./data/processed/automl_1
path_out = "./data/ml_stuff/automl_2"
# check if exists, else increase the number
# if os.path.exists(path_out):
#     number = path_out.split("_")[-1]
#     number = int(number) + 1
#     path_out = path_out.split("_")[0] + "_" + str(number)

automl = AutoML(
    results_path=path_out,
    algorithms=[
        "Xgboost",
        "CatBoost",
        "Random Forest",
    ],
    total_time_limit=5 * 60,
    # n_jobs=6,
    explain_level=2,
    mode="Explain",
    random_state=42,
)

automl.fit(X_train, y_train)


# Predict the target values
y_pred = automl.predict(X_test)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print the evaluation metrics
print(f"\nAccuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
