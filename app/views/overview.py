import streamlit as st
import pandas as pd
import plotly.express as px

def show_overview():
    st.title("Event Revenue Forecasting")
    st.write(
        "A dashboard for forecasting UBC Film Society concession revenue "
        "using Instagram engagement and movie popularity data."
    )

    st.divider()

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Events Analyzed", "84")
    col2.metric("Best CV R²", "0.35")
    col3.metric("Best Test R²", "0.61")
    col4.metric("Best Model", "RidgeCV")

    st.divider()

    # Two-column layout
    left_col, right_col = st.columns([1, 1.4])

    with left_col:
        st.subheader("Project Summary")
        st.write(
            """
            This project predicts concession revenue for film screenings using
            event data, Instagram engagement metrics, and movie metadata.
            """
        )

        st.markdown(
            """
            **Data used:**
            - Square concession sales
            - Instagram likes, comments, and shares
            - Letterboxd ratings and review counts
            - Event and movie metadata
            """
        )

        st.info(
            "Key finding: movie popularity was the strongest predictor of event revenue."
        )

    with right_col:
        st.subheader("Model Performance")

        results = pd.DataFrame({
            "Feature Set": [
                "Baseline",
                "Movie Only",
                "Combined"
            ],
            "Mean CV R²": [
                0.18,
                0.30,
                0.35
            ]
        })

        fig = px.bar(
            results,
            x="Feature Set",
            y="Mean CV R²",
            color="Feature Set",
            text="Mean CV R²",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Cross-Validated Performance by Feature Set"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig.update_layout(
            showlegend=False,
            yaxis_range=[0, 0.45]
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Main Takeaways")

    takeaway1, takeaway2, takeaway3 = st.columns(3)

    with takeaway1:
        st.success("Movie metadata improved prediction performance.")

    with takeaway2:
        st.success("Instagram engagement still added useful signal.")

    with takeaway3:
        st.success("The combined feature set performed best.")