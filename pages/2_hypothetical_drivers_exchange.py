import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import json
import plotly.express as px
import os
import sys
import joblib

from analysis.compound_models.change_driver import get_hypothetical_predictions

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="fan_f1ction", page_icon=":racing_car:")

# data


@st.cache_resource
def load_data():
    file_path = "./tmp/working_data/filtered.csv"
    data = pd.read_csv(file_path, sep=";", encoding="UTF-8", header=0)
    return data.drop(["Unnamed: 0.1", "Unnamed: 0"], axis=1)


@st.cache_resource
def load_full_binary():
    file_path = "./tmp/working_data/binaries_years.csv"
    data = pd.read_csv(file_path, sep=";", encoding="UTF-8", header=0)
    return data.drop(["Unnamed: 0"], axis=1)


df = load_data()
full_binary_df = load_full_binary()

# page
st.title("Fan f1ction with machine learning 🚦")
st.markdown("---")

# Body
st.markdown(
    "### ⚠ Warning : the model used to simulate lap times cannot pretend to replicate accurately true racing "
    "conditions nor driver talent. "
)
st.markdown(
    "Also, it is a work in progress, and the model size is constrained by the storage capacity of a free github "
    "public repository."
)
st.markdown("1. First, select a year and a track to get a race.")
st.markdown("2. Then, select a performance from a driver having run that race.")
st.markdown(
    "3. Finally, choose any other driver that wasn't in the same team during that race, to model its "
    "performance all other things being equal."
)
st.markdown("---")


# Sidebar
sidebar_menu = st.sidebar
with sidebar_menu:
    with open("./tmp/lottie/racecar.json", "r") as f:
        lottie_json = json.load(f)
    st_lottie(lottie_json)

    st.sidebar.header("Choose a race :")
    year = st.sidebar.selectbox("Pick the year :", df["Year"].unique())
    year_df = df.loc[df["Year"] == year]

    circuit = st.sidebar.selectbox("Pick the track :", year_df["EventName"].unique())
    circuit_df = year_df.loc[df["EventName"] == circuit]

    driver = st.sidebar.selectbox(
        "Choose the real performance from a driver to be compared :",
        options=circuit_df["Driver"].unique(),
    )
    driver_df = circuit_df.loc[circuit_df["Driver"] == driver]
    selected_driver_team = circuit_df.loc[circuit_df["Driver"] == driver, "Team"].iloc[
        0
    ]

    options = df.loc[
        (df["Driver"] != driver) & (df["Team"] != selected_driver_team), "Driver"
    ].unique()

    drivers = st.sidebar.multiselect(
        "Choose a driver to compare his hypothetical performance all other things being equal:",
        options=options,
        default=options[0],
    )
    driver_2_df = circuit_df.loc[circuit_df["Driver"].isin(drivers)]
    st.sidebar.subheader("Plot parameters")
    plot_height = st.sidebar.slider("Specify plot height :", 400, 700, 500)
    plot_width = st.sidebar.slider("Specify plot width :", 500, 1200, 900)

# generate data from model
model_path = os.path.join(os.getcwd(), "tmp", "models", "ridged_stacked_model_1.joblib")
stacked_model = joblib.load(model_path)
new_df = pd.DataFrame(columns=["Driver", "LapTime", "Team", "Year", "EventName"])

# make final dataframe
hypothetical_pace = []
for d in drivers:
    predictions = get_hypothetical_predictions(
        stacked_model, full_binary_df, driver, d, circuit, year
    )
    if len(predictions) > 0:
        hypothetical_pace.extend(predictions)
        print(f"Driver: {d}, Number of predictions: {len(predictions)}")
    else:
        print(f"Warning: No predictions found for Driver {d}")

# Repeat each driver for the corresponding lap time
new_df["Driver"] = (
    [driver for d in drivers for driver in [d] * len(predictions)]
    if len(predictions) > 0
    else []
)
new_df["LapTime"] = hypothetical_pace
new_df["Year"] = driver_df["Year"].iloc[0]
new_df["EventName"] = driver_df["EventName"].iloc[0]
new_df["Team"] = driver_df["Team"].iloc[0]

final_df = pd.concat([driver_df, new_df], ignore_index=True).drop(
    [
        "DriverNumber",
        "IsPersonalBest",
        "Position",
        "Stint",
        "Compound",
        "FreshTyre",
        "TyreLife",
        "Unnamed: 0.2",
    ],
    axis=1,
)
final_df["LapTime"] = final_df["LapTime"].astype(float)

# body 2
st.markdown(
    f"***If {drivers} would have been on the same car, tyres, starting position and strategy as {driver}"
    f" during the {year} {circuit}, this could have been a selection of their laptimes***"
)
# violin plot
fig = px.violin(
    final_df,
    x="Driver",
    y="LapTime",
    color="Team",
    box=True,
    points="outliers",
    width=plot_width,
    height=plot_height,
    violinmode="overlay",
    color_discrete_map={
        "Ferrari": "#FD2819",
        "Mercedes": "#010000",
        "AlphaTauri": "#6D9EF2",
        "Alpine": "#0225B5",
        "Red Bull Racing": "#061964",
        "Aston Martin": "#066432",
        "McLaren": "#DF6801",
        "Sauber": "#650505",
        "Haas": "#FEFEFE",
        "Williams": "#471DEC",
    },
)
st.plotly_chart(fig)
