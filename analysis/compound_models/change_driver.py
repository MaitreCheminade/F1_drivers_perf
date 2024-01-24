import pandas as pd
import numpy as np
import joblib
import sys
import os


# variables
# df_path = "../../tmp/working_data/filtered.csv"
# binary_path = "../../tmp/working_data/binaries.csv"
# binary_years_path = "../../tmp/working_data/binaries_years.csv"

df_path = os.path.join(os.getcwd(), "tmp", "working_data", "filtered.csv")
binary_path = os.path.join(os.getcwd(), "tmp", "working_data", "binaries.csv")
binary_years_path = os.path.join(
    os.getcwd(), "tmp", "working_data", "binaries_years.csv"
)

df = pd.read_csv(df_path, sep=";", header=0, encoding="UTF-8")
binary_df = pd.read_csv(binary_path, sep=";", header=0, encoding="UTF-8").drop(
    "Unnamed: 0", axis=1
)
binary_years_df = pd.read_csv(
    binary_years_path, sep=";", header=0, encoding="UTF-8"
).drop("Unnamed: 0", axis=1)

# model_path = "../../tmp/models/ridged_stacked_model_1.joblib"
model_path = os.path.join(os.getcwd(), "tmp", "models", "ridged_stacked_model_1.joblib")
stacked_model = joblib.load(model_path)

X = binary_df.drop(["LapTime"], axis=1)
y = binary_df["LapTime"]


def get_hypothetical_predictions(
    model, base_data, driver_to_replace, replacement_driver, event_name, year: int
):
    """
    Get predicted lap times for a hypothetical scenario where a specific driver is replaced.

    Params:
    - model: StackingRegressor model
    - base_data: basic dataframe with all data in binary features
    - driver_to_replace: String, the driver to be replaced (e.g., 'Driver_HAM')
    - replacement_driver: String, the driver to replace the original driver (e.g., 'Driver_ALO')
    - event_name : str, name of track
    - year : int
    Returns: predicted lap times for the hypothetical scenario
    """
    # Filter data according to chosen parameters
    year_num = "Year_" + str(year)
    track_name = "EventName_" + event_name
    driver_1_name = "Driver_" + driver_to_replace
    driver_2_name = "Driver_" + replacement_driver
    filtered_data = base_data[
        (base_data[f"{year_num}"] == 1)
        & (base_data[f"{track_name}"] == 1)
        & (base_data[f"{driver_1_name}"] == 1)
    ].copy()

    # Set the presence of the original driver to 0 and replacement driver to 1
    filtered_data[driver_1_name] = 0
    filtered_data[driver_2_name] = 1

    filtered_data = filtered_data.drop(
        columns=[col for col in filtered_data.columns if "Year_" in col]
    )
    new_features = filtered_data.drop(["LapTime"], axis=1)
    # # Make predictions for the hypothetical scenario
    predictions = np.round(model.predict(new_features), 3)
    # print("Number of original laptimes :", len(new_features))
    # print("Number of predicted laptimes :", len(predictions))
    # print("Actual laptimes for the replacement driver :", predictions)
    return predictions


get_hypothetical_predictions(
    stacked_model, binary_years_df, "HAM", "ALO", "French Grand Prix", 2021
)
