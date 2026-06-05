import streamlit as st

from views.overview import show_overview
from views.insights import show_insights
from views.predictor import show_predictor

st.set_page_config(
    page_title="Event Revenue Forecasting",
    page_icon="🎬",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] h2 {
        color: #0f172a;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## 🎬 Event Revenue")
    st.caption("Forecasting dashboard")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Insights",
            "Revenue Predictor"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model")
    st.caption("Best CV R²: **0.35**")
    st.caption("Events analyzed: **84**")

    st.divider()

    st.info(
        "Predict concession revenue using Instagram engagement and movie popularity data."
    )

if page == "Overview":
    show_overview()

elif page == "Insights":
    show_insights()

elif page == "Revenue Predictor":
    show_predictor()