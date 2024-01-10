import fastf1
from fastf1.ergast import Ergast
import pandas as pd
import sys

# in params, need to choose range of years and number of races per year, accounting for shorter years
# also, meta function will imply to select drivers to not get overwhelmed
default_drivers_list = [
    "HAM",
    "VER",
    "BOT",
    "LEC",
    "ALO",
    "TSU",
    "RIC",
    "GAS",
    "OCO",
    "PER",
    "SAI",
    "NOR",
    "RUS",
    "HUL",
    "MAG",
    "MAZ",
    "MSC",
    "GRO",
    "VET",
    "RAI",
    "VAN",
    "STR",
    "ERI",
    "HAR",
    "SIR",
    "ALB",
    "GIO",
    "KVY",
]
start_year = 2019
finish_year = 2023
default_year_list = range(start_year, finish_year + 1)


def get_races_by_year(year):
    """Return number of races in year"""
    ergast = Ergast()
    races = ergast.get_race_schedule(year)
    print("Number of races :", len(races))
    return len(races)


def get_drivers_race_pace(race_number: int, year):
    """Get dataframe with race pace and pertinent data affecting it
    for each driver during a race"""
    year = year
    race_num = race_number
    session = fastf1.get_session(year, race_number, "R")
    session.load(laps=True, telemetry=False, messages=False)
    driver_laps = pd.DataFrame()
    try:
        driver_laps = session.laps
        print("success :", driver_laps)
    except:
        print("Unsuccessful")
        pass
    if not driver_laps.empty:
        driver_laps = driver_laps.reset_index()
        driver_laps.drop(
            [
                "Time",
                "index",
                "LapNumber",
                "PitOutTime",
                "PitInTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Sector1SessionTime",
                "Sector2SessionTime",
                "Sector3SessionTime",
                "SpeedI1",
                "SpeedI2",
                "SpeedFL",
                "SpeedST",
                "LapStartTime",
                "LapStartDate",
                "DeletedReason",
                "Deleted",
                "FastF1Generated",
                "IsAccurate",
            ],
            axis=1,
            inplace=True,
        )
        # driver_laps.to_csv("../tmp/raw_data/test_drivers.csv", encoding="UTF-8", header=True, sep=";")
        driver_laps["Year"] = f"{year}"
        return driver_laps

    return None


def get_drivers_race_information(drivers_list, year_list):
    ranged_results = {}
    for year in year_list:
        race_num = get_races_by_year(year)
        year_results = pd.DataFrame()
        for race_number in range(1, race_num + 1):
            session = fastf1.get_session(year, race_number, "R")
            session.load(laps=True)
            race_data = get_drivers_race_pace(race_number, year)
            filtered_data = race_data[race_data["Driver"].isin(drivers_list)]
            filtered_data["EventName"] = session.event["EventName"]
            year_results = pd.concat([year_results, filtered_data], ignore_index=True)

        year_results.to_csv(
            f"../tmp/years/{year}.csv", sep=";", header=True, encoding="UTF-8"
        )

        ranged_results[year] = year_results

    print(ranged_results)


get_drivers_race_information(default_drivers_list, default_year_list)
