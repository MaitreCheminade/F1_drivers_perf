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
left_column, right_column = st.columns((3, 1))
st.title("F1 drivers comparative performance app :racing_car:")
st.markdown("---")
st.markdown("Hello !")
st.markdown("Welcome to the F1 drivers comparative performance app !")
st.markdown("This is a personal project designed to compare F1 drivers performance. ")
st.markdown("### ***It's lights out and away we go ! 🏁***")

# Sidebar
sidebar_menu = st.sidebar
with sidebar_menu:
    with open("./tmp/lottie/racecar.json", "r") as f:
        lottie_json = json.load(f)
    st_lottie(lottie_json)
    st.sidebar.header("Compare drivers performance for a single race")
    year = st.sidebar.selectbox("Pick the year :", df["Year"].unique())
    year_df = df.loc[df["Year"] == year]
    circuit = st.sidebar.selectbox("Pick your track :", year_df["EventName"].unique())
    circuit_df = year_df.loc[df["EventName"] == circuit]
    drivers = st.sidebar.multiselect("Choose the drivers to compare :",
                                     options=circuit_df["Driver"].unique(),
                                     default=["HAM", "BOT", "VER"])
    drivers_df = circuit_df.loc[circuit_df["Driver"].isin(drivers)]
    st.sidebar.subheader("Plot parameters")
    plot_height = st.sidebar.slider("Specify plot height :", 400, 700, 500)
    plot_width = st.sidebar.slider("Specify plot width :", 500, 1200, 900)


# Body
left_column, right_column = st.columns((3, 1))
# make violin plot to compare race pace of drivers
# TODO fix the alignment of x labels
fig = px.violin(drivers_df, x="Driver", y="LapTime", color="Team",
                box=True, hover_data=drivers_df.columns, points="outliers",
                width=plot_width, height=plot_height, violinmode="overlay",
                color_discrete_map={"Ferrari": "#FD2819", "Mercedes": "#010000", "AlphaTauri": "#6D9EF2",
                                    "Alpine": "#0225B5", "Red Bull Racing": "#061964", "Aston Martin": "#066432",
                                    "McLaren": "#DF6801", "Sauber": "#650505", "Haas": "#FEFEFE",
                                    "Williams": "#471DEC"})
st.plotly_chart(fig)

# TODO display drivers standings in chosen year (maybe to the right ?)
st.markdown(f"### Those were the drivers standings in {year}")

# TODO hypothetical scenario of replacing a driver by another (toggle if exists ?)
