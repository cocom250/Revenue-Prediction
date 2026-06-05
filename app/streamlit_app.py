import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Event Revenue Insights")

df = pd.read_csv("../Data/enriched_events.csv")

# Revenue Distribution plot
st.subheader("Revenue Distribution")

fig = px.histogram(
    df,
    x="revenue",
    nbins=25,
    title="Distribution of Event Revenue",
    labels={"revenue": "Revenue"},
    color_discrete_sequence=px.colors.qualitative.Pastel
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Most events generated relatively modest revenue, while a small number of high-performing events produced substantially higher sales."
)

# Revenue vs movie popularity plot
st.subheader("Revenue vs Movie Popularity")

fig = px.scatter(
    df,
    x="max_review_count",
    y="revenue",
    hover_data=["movie_1_title", "movie_2_title"],
    title="Revenue vs Movie Popularity",
    labels={
        "max_review_count": "Max Letterboxd Review Count",
        "revenue": "Revenue"
    },
    color="popularity_bucket",
    log_x=True
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Events featuring more widely reviewed films tended to generate higher revenue, suggesting movie popularity is an important predictor."
)

# Revenue by popularity tier
st.subheader("Revenue by Movie Popularity Tier")

fig = px.box(
    df,
    x="popularity_bucket",
    y="revenue",
    title="Revenue by Movie Popularity Tier",
    labels={
        "popularity_bucket": "Movie Popularity Tier",
        "revenue": "Revenue"
    },
    color="popularity_bucket"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Revenue generally increased across movie popularity tiers, supporting the use of movie metadata in the forecasting model."
)

# Model Comparison
st.subheader("Model Performance Comparison")

results = pd.DataFrame({
    "Feature Set": [
        "Baseline",
        "Movie Only",
        "Combined"
    ],
    "Best CV R²": [
        0.179,
        0.298,
        0.354
    ]
})

fig = px.bar(
    results,
    x="Feature Set",
    y="Best CV R²",
    title="Best Cross-Validated Model Performance by Feature Set",
    labels={
        "Best CV R²": "Mean CV R²"
    },
    color_discrete_sequence=px.colors.qualitative.Pastel
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Adding movie metadata improved predictive performance, with the combined Instagram and movie feature set performing best."
)

# Feature importance
st.subheader("Feature Importance")

importance_df = pd.read_csv("../Data/feature_importance.csv")

fig = px.bar(
    importance_df.sort_values("Importance").tail(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Feature Importances",
    color="Importance",
    color_continuous_scale="Viridis"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Movie popularity, measured by Letterboxd review count, was the strongest predictor of event revenue."
)