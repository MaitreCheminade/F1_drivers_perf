import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score

from data_processing.process_data import transform_vars_to_binary

# variables
# df_path = "C:/Users/pycharm_projects/f1_project/tmp/working_data/filtered.csv"
# binary_path = "C:/Users/pycharm_projects/f1_project/tmp/working_data/binaries.csv"
df_path = os.path.join(os.getcwd(), "tmp", "working_data", "filtered.csv")
binary_path = os.path.join(os.getcwd(), "tmp", "working_data", "binaries.csv")

df = pd.read_csv(df_path, sep=";", header=0, encoding="UTF-8")
binary_df = pd.read_csv(binary_path, sep=";", header=0, encoding="UTF-8").drop(
    "Unnamed: 0", axis=1
)

model_path = os.path.join(os.getcwd(), "tmp", "models", "ridged_stacked_model_1.joblib")
stacked_model = joblib.load(model_path)

X = binary_df.drop(["LapTime"], axis=1)
y = binary_df["LapTime"]


def get_hypothetical_predictions(
    model,
    ref_data,
    original_data,
    driver_to_replace,
    replacement_driver,
    event_name,
    year,
):
    """
    Get predicted lap times for a hypothetical scenario where a specific driver is replaced.

    Params:
    - model: StackingRegressor model
    - original_data: DataFrame containing the original features as binaries, which corresponds to the X in the model.
    - driver_to_replace: String, the driver to be replaced (e.g., 'Driver_HAM')
    - replacement_driver: String, the driver to replace the original driver (e.g., 'Driver_ALO')

    Returns: predicted lap times for the hypothetical scenario
    """
    # Copy the original features to create a hypothetical scenario
    new_features = original_data.copy()
    ref_data.loc[ref_data["EventName"] == event_name]
    ref_data.loc[ref_data["Year"] == year]
    writing_path = f"../../tmp/streamlit/{driver_to_replace}-{replacement_driver}-{event_name}-{year}.csv"
    new_features = transform_vars_to_binary(ref_data, writing_path)
    # Find the index of the columns representing the original and replacement drivers
    # original_driver_index = new_features.columns.get_loc(driver_to_replace)
    # replacement_driver_index = new_features.columns.get_loc(replacement_driver)

    # Set the presence of the original driver to 0 and replacement driver to 1
    # new_features.iloc[:, original_driver_index] = 0
    # new_features.iloc[:, replacement_driver_index] = 1
    new_features = new_features.loc[new_features[driver_to_replace] == 0]
    new_features = new_features.loc[new_features[replacement_driver] == 1]
    new_features = new_features.drop(["LapTime"], axis=1)
    print("There are your columns :", new_features.columns)
    # Make predictions for the hypothetical scenario
    predictions = model.predict(new_features)

    # Display the predicted lap times for the hypothetical scenario
    print("Predicted Lap Times for the Hypothetical Scenario:")
    print(predictions)

    return predictions


# get_hypothetical_predictions(
#     stacked_model,
#     df,
#     binary_df,
#     "Driver_HAM",
#     "Driver_ALO",
#     "British Grand Prix",
#     "2021",
# )
