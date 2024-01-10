import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import json
import plotly.express as px
import os
print("Current working directory:", os.getcwd())

from analysis.compound_models.change_driver import get_hypothetical_predictions

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="Homepage", page_icon=":racing_car:")

