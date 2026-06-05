import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pages.insights import show_insights
from pages.overview import show_overview
from pages.predictor import show_predictor

# Sidebar Navigator 
st.set_page_config(
    page_title="Event Revenue Forecasting",
    page_icon="🎬",
    layout="wide"
)

# Sidebar
st.sidebar.title("🎬 Event Revenue")
st.sidebar.caption("Forecasting dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Insights",
        "Revenue Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Predict event revenue using Instagram engagement and movie popularity data."
)

df = pd.read_csv("../Data/enriched_events.csv")

# Page routing
if page == "Overview":
    show_overview()

elif page == "Insights":
    show_insights()

elif page == "Revenue Predictor":
    show_predictor()