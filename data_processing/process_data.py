import pandas as pd
import os
import re
from datetime import datetime


"""Params"""
# files
dir_path = "../tmp/years/"
df_path = "../tmp/working_data/filtered.csv"
complete_df = pd.read_csv("../tmp/working_data/test.csv", header=0, sep=";", encoding="UTF-8")
# patterns
time_pattern = re.compile(r"(0 days )")
point_pattern = re.compile(r"\.")


# processing functions
def clean_lap_times(df):
    """convert the laptimes into usable float (seconds)"""
    laptimes_list = df["LapTime"].tolist()
    new_laptimes = []
    for lap in laptimes_list:
        lap = re.sub(time_pattern, "", str(lap))
        if '.' not in lap:
            lap += ".0"
        laptime = datetime.strptime(lap, "%H:%M:%S.%f").time()
        total_seconds = laptime.second + laptime.microsecond / 1e6 + laptime.minute * 60 + laptime.hour * 3600
        new_laptimes.append(total_seconds)

    df["LapTime"] = new_laptimes
    return df


def join_dataframes(directory):
    """Concatenate all csv in a directory as df
    Then write the new concatenated file"""
    final_df = pd.DataFrame()
    for file in os.listdir(directory):
        file_df = pd.read_csv(f"{directory}{file}", sep=";", header=0, encoding="UTF-8")
        final_df = pd.concat([final_df, file_df], ignore_index=True).dropna(subset=["LapTime"])

    final_df = clean_lap_times(final_df).dropna()
    final_df = final_df.replace("São Paulo Grand Prix", "Brazilian Grand Prix")
    final_df = final_df.replace("Mexico City Grand Prix", "Mexican Grand Prix")
    final_df = final_df.replace("Styrian Grand Prix", "Austrian Grand Prix")

    final_df = final_df.replace("Renault", "Alpine")
    final_df = final_df.replace("Alfa Romeo", "Sauber")
    final_df = final_df.replace("Alfa Romeo Racing", "Sauber")
    final_df = final_df.replace("Toro Rosso", "AlphaTauri")
    final_df = final_df.replace("Racing Point", "Aston Martin")
    final_df = final_df.replace("Force India", "Aston Martin")
    final_df = final_df.loc[final_df["TrackStatus"] == 1].drop("TrackStatus", axis=1)

    final_df.to_csv("../tmp/working_data/test.csv", header=True, sep=";", encoding="UTF-8")
    return final_df


def transform_vars_to_binary(path):
    """Transform discrete variables into one binary (T/F) variable for each unique value
    input origin csv, output prints new csv
    Commented lines may be of use if you only want to binarize some variables"""
    df = pd.read_csv(path, sep=";", header=0, encoding="UTF-8").dropna()

    columns_to_convert = ["Driver", "Compound", "Team", "FreshTyre", "Stint", "TyreLife", "Position", "EventName"]
    # base_df = df[["Stint", "TyreLife", "Position"]]
    combined_df = pd.DataFrame()
    for col in columns_to_convert:
        for value in df[col].unique():
            new_col_name = col + "_" + str(value)
            new_df = df[col].apply(lambda x: 1 if x == value else 0)
            new_df = new_df.rename(new_col_name)
            combined_df = pd.concat([combined_df, new_df], axis=1)

    # combined_df = pd.concat([combined_df, base_df], axis=1)
    combined_df = pd.concat([combined_df, df["LapTime"]], axis=1)
    combined_df.to_csv("../tmp/working_data/binaries.csv", encoding="UTF-8", sep=";", header=True)

    return combined_df


df = pd.read_csv(df_path, encoding='UTF-8', sep=";", header=0)
filtered_df = df.loc[~df.index.isin(df.groupby(['Year', 'EventName', 'Driver'])['LapTime'].nlargest(3).reset_index()['level_3'].values)]
filtered_df.to_csv("../tmp/working_data/filtered.csv", header=True, sep=";")

join_dataframes(dir_path)
transform_vars_to_binary(df_path)
