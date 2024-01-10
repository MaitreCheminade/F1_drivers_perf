import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import joblib
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score


"""This model is a ridge regression stacking one random forest regressor per track"""
# variables
df_path = "../../tmp/working_data/filtered.csv"
binary_path = "../../tmp/working_data/binaries.csv"

df = pd.read_csv(df_path, sep=";", header=0, encoding="UTF-8")
binary_df = pd.read_csv(binary_path, sep=";", header=0, encoding="UTF-8").drop(
    "Unnamed: 0", axis=1
)

track_list = list(df["EventName"].unique())
# track_list = ["EventName_Austrian Grand Prix", "EventName_Belgian Grand Prix", "EventName_Singapore Grand Prix",
#               "EventName_Azerbaijan Grand Prix", "EventName_Sakhir Grand Prix", "EventName_Brazilian Grand Prix",
#               "EventName_United States Grand Prix", "EventName_Russian Grand Prix", "EventName_Bahrain Grand Prix",
#               "EventName_French Grand Prix", "EventName_Chinese Grand Prix", "EventName_Japanese Grand Prix",
#               "EventName_British Grand Prix", "EventName_Portuguese Grand Prix", "EventName_Hungarian Grand Prix",
#               "EventName_Mexican Grand Prix", "EventName_Emilia Romagna Grand Prix", "EventName_German Grand Prix",
#               "EventName_Spanish Grand Prix", "EventName_Italian Grand Prix", "EventName_Turkish Grand Prix",
#               "EventName_Qatar Grand Prix", "EventName_Australian Grand Prix", "EventName_Eifel Grand Prix",
#               "EventName_Tuscan Grand Prix", "EventName_Dutch Grand Prix", "EventName_Canadian Grand Prix",
#               "EventName_Monaco Grand Prix", "EventName_Miami Grand Prix", "EventName_Saudi Arabian Grand Prix"]
track_models = {}
track_predictions = {}

# random forest params
seed = 54
trees = 10

"""Main functions"""


def train_random_forest_model(x_train, y_train, n_trees, seed):
    """
    Train a Random Forest model
    Params :
    - train sets (x and y)
    - number of trees
    - seed for retrieval

    Returns : model
    """
    model = RandomForestRegressor(n_estimators=n_trees, random_state=seed)
    model.fit(x_train, y_train)
    return model


def print_evaluation_metrics(y_true, y_pred):
    """Print RMSE and R2 score.
    Params : true and predicted data
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.3f}")
    print(f"R2 Score: {r2:.3f}")
    print("\n")


def evaluate_model_performance(model, x_train, y_train, x_test, y_test):
    """
    Evaluate the performance of a regression model.
    Prints : benchmarks as RMSE and R2 score
    """
    # Training set evaluation
    y_train_predict = model.predict(x_train)
    print("Training Set Performance:")
    print_evaluation_metrics(y_train, y_train_predict)

    # Testing set evaluation
    y_test_predict = model.predict(x_test)
    print("Testing Set Performance:")
    print_evaluation_metrics(y_test, y_test_predict)


def stack_models_into_ridge_reg(input_models, x, y):
    """
    Function to stack models into a ridge regression stacked model.
    Enter the first models as a dictionary (with model_name : model_type),
    then x and y.
    Prints performance (RMSE + R²)
    Returns : the stacked model
    """
    stacked_model = StackingRegressor(
        estimators=list(list(input_models.items())[:1]),
        n_jobs=-1,
        final_estimator=Ridge(alpha=0.1),
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.22, random_state=24
    )
    stacked_model.fit(x_train, y_train)
    evaluate_model_performance(stacked_model, x_train, y_train, x_test, y_test)

    return stacked_model


# Train one random forest model per track
track_models = {}
track_predictions = {}

for track_name in track_list:
    track_df = binary_df[binary_df[f"{track_name}"] == 1]
    track_binaries_path = (
        f"../../tmp/working_data/track_data/track_binary_{track_name}.csv"
    )
    track_df.to_csv(track_binaries_path, header=0, sep=";", encoding="UTF-8")
    # binary_track_df = transform_vars_to_binary(track_df, track_binaries_path)

    print(f"Number of samples for {track_name}: {len(track_df)}")
    if len(track_df) < 2:
        print(f"Skipping {track_name} due to insufficient samples.")
        continue

    x = track_df.drop("LapTime", axis=1)
    y = track_df["LapTime"]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.22, random_state=24
    )

    # Train the model
    model = train_random_forest_model(x_train, y_train, trees, seed)
    track_models[track_name] = model

    # Evaluate model performance
    evaluate_model_performance(model, x_train, y_train, x_test, y_test)

# stack the models
x2 = binary_df.iloc[:, :].drop("LapTime", axis=1)
y2 = binary_df["LapTime"]
ridge_stacked_model_1 = stack_models_into_ridge_reg(track_models, x2, y2)
# store the model
joblib.dump(ridge_stacked_model_1, "../../tmp/models/ridged_stacked_model_1.joblib")
