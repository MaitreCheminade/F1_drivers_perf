import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import json
import plotly.express as px
import os
print("Current working directory:", os.getcwd())

from analysis.compound_models.change_driver import get_hypothetical_predictions

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="Drivers race pace comparison", page_icon=":racing_car:")

# Data loading
# lottie_image = "https://lottie.host/080804c8-cacf-461c-bed8-632ebb2ad5b6/2a83ly2Sw1.json"


@st.cache_resource
def load_data():
    file_path = "./tmp/working_data/filtered.csv"
    data = pd.read_csv(file_path, sep=";",
                       encoding="UTF-8",
                       header=0)
    return data.drop(["Unnamed: 0.1", "Unnamed: 0"], axis=1)


df = load_data()

# Title
st.title("F1 drivers comparative performance app :racing_car:")
st.markdown("---")
st.markdown("## Homepage")

# Sidebar
sidebar_menu = st.sidebar
with sidebar_menu:
    with open("./tmp/lottie/racecar.json", "r") as f:
        lottie_json = json.load(f)
    st_lottie(lottie_json)
    st.sidebar.header("Select a season to get some overview statistics")
    year = st.sidebar.selectbox("Pick the year :", df["Year"].unique())
    year_df = df.loc[df["Year"] == year]

# Body
st.markdown("Hello !")
st.markdown("Welcome to the F1 drivers comparative performance app !")
st.markdown("This is a personal project designed to compare F1 drivers performance. ")
st.markdown("This homepage allows you to select a year and get some overview statistics of that season.")
st.markdown("The race pace comparisons page will show you violin plot comparisons of selectable drivers race pace"
            "for a selected race.")
st.markdown("The hypothetical drivers exchange page runs a machine learning model simulating the race pace of a selected"
            "driver if he replaced another selected driver (same car).")
st.markdown("Data covers F1 seasons from 2018 to 2022 (included), and has been mostly retrieved through the "
            "fastF1 python package (check it here : https://github.com/theOehrly/Fast-F1 )")

st.markdown("#### ***It's lights out and away we go ! 🏁***")
st.markdown("---")

# Body 2
# TODO upload data for drivers and constructors championship evolution of standings
st.markdown("### Season statistics ")

left_column, right_column = st.columns((3, 3))
with left_column:
    st.markdown(f"**Those were the drivers standings in {year}**")
    drivers_standings_df = pd.read_csv(f"./tmp/working_data/drivers_standings/{year}.csv", sep=";", encoding="UTF-8",
                                   header=0)
    drivers_standings_df[["Position", "Driver", "Points"]]


