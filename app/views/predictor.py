import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "revenue_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "model_features.pkl"
MODEL_RMSE = 91

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_features():
    return joblib.load(FEATURE_PATH)


def show_predictor():
    st.title("Revenue Predictor")

    st.write(
        "Enter pre-event information to estimate expected concession revenue."
    )

    model = load_model()
    model_features = load_features()

    left, right = st.columns([1.2, 1])

    with left:
        with st.container(border=True):
            st.subheader("Event Details")

            col1, col2, = st.columns(2)

            with col1:
                event_type = st.selectbox(
                    "Event Type",
                    ["double screening", "special event", "single screening"]
                )

                collab = st.selectbox(
                    "Collaboration",
                    ["No", "Yes"]
                )

            with col2:
                day_of_week = st.selectbox(
                    "Day of Week",
                    [
                        "Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday", "Sunday"
                    ]
                )

                month = st.slider(
                    "Month",
                    min_value=1,
                    max_value=12,
                    value=9
                )

        with st.container(border=True):
            st.subheader("Instagram Engagement")

            col1, col2, col3 = st.columns(3)

            with col1:
                num_posts = st.number_input(
                    "Number of Promotional Posts",
                    min_value=0,
                    value=1
                )
            with col2:
                days_from_event = st.number_input(
                    "Days Between Post and Event",
                    min_value=0,
                    value=3
                )

            with col3:
                likes = st.number_input(
                    "Likes",
                    min_value=0,
                    value=100
                )

            col4, col5 = st.columns(2)

            with col4:
                num_comments = st.number_input(
                    "Comments",
                    min_value=0,
                    value=5
                )

            with col5:
                num_shares = st.number_input(
                    "Shares",
                    min_value=0,
                    value=10
                )

        with st.container(border=True):
            st.subheader("General Promo")

            col1, col2 = st.columns(2)
            with col1:
                general_promo_likes = st.number_input(
                    "Grouped Promo Likes",
                    min_value=0,
                    value=0
                )

            with col2:
                general_promo_shares = st.number_input(
                    "Grouped Promo Shares",
                    min_value=0,
                    value=0
                )

        with st.container(border=True):
            st.subheader("Movie Metadata")

            screening_format = st.selectbox(
                "Screening Format",
                ["Single Screening", "Double Screening"]
            )

            is_double_feature = screening_format == "Double Screening"

            st.markdown("**Movie 1**")

            col1, col2, col3 = st.columns(3)

            with col1:
                movie_1_rating = st.number_input(
                    "Rating",
                    min_value=0.0,
                    max_value=5.0,
                    value=3.8,
                    step=0.1,
                    key="movie_1_rating"
                )

            with col2:
                movie_1_review_count = st.number_input(
                    "Review Count",
                    min_value=0,
                    value=100000,
                    step=1000,
                    key="movie_1_reviews"
                )

            with col3:
                movie_1_age = st.number_input(
                    "Movie Age",
                    min_value=0,
                    value=20,
                    key="movie_1_age"
                )

            primary_genre = st.selectbox(
                "Primary Genre",
                ["Action", "Animation", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Documentary", "Other"]
            )

            if is_double_feature:
                st.markdown("**Movie 2**")

                col1, col2, col3 = st.columns(3)

                with col1:
                    movie_2_rating = st.number_input(
                        "Rating",
                        min_value=0.0,
                        max_value=5.0,
                        value=3.8,
                        step=0.1,
                        key="movie_2_rating"
                    )

                with col2:
                    movie_2_review_count = st.number_input(
                        "Review Count",
                        min_value=0,
                        value=100000,
                        step=1000,
                        key="movie_2_reviews"
                    )

                with col3:
                    movie_2_age = st.number_input(
                        "Movie Age",
                        min_value=0,
                        value=20,
                        key="movie_2_age"
                    )

            else:
                movie_2_rating = movie_1_rating
                movie_2_review_count = movie_1_review_count
                movie_2_age = movie_1_age

    # Feature engineering
    total_engagement = (
        likes + num_comments + num_shares
    )

    if num_posts > 0:
        engagement_per_post = total_engagement / num_posts
    else:
        engagement_per_post = 0

    avg_movie_rating = (movie_1_rating + movie_2_rating) / 2

    max_review_count = max(
        movie_1_review_count,
        movie_2_review_count
    )

    if is_double_feature:
        total_review_count = movie_1_review_count + movie_2_review_count
    else:
        total_review_count = movie_1_review_count

    avg_movie_age = (movie_1_age + movie_2_age) / 2

    double_feature_num = 1 if is_double_feature else 0

    log_max_review_count = np.log1p(max_review_count)
    log_total_review_count = np.log1p(total_review_count)

    input_data = pd.DataFrame({
    "likes": [likes],
    "num_comments": [num_comments],
    "num_shares": [num_shares],
    "num_posts": [num_posts],
    "days_from_event": [days_from_event],
    "general_promo_likes": [general_promo_likes],
    "general_promo_shares": [general_promo_shares],
    "total_engagement": [total_engagement],
    "engagement_per_post": [engagement_per_post],
    "month": [month],

    "avg_movie_rating": [avg_movie_rating],
    "log_max_review_count": [log_max_review_count],
    "log_total_review_count": [log_total_review_count],
    "avg_movie_age": [avg_movie_age],
    "double_feature": [double_feature_num],

    "type": [event_type],
    "collab": [collab],
    "day_of_week": [day_of_week],
    "primary_genre": [primary_genre],
    })

    # Match exact feature order used during training
    input_data = input_data[model_features]

    with right:
        st.subheader("Prediction")

        st.info(
            "This prediction is based on the final model trained on historical Film Society events."
        )

        if st.button("Predict Revenue", use_container_width=True):
            pred_log = model.predict(input_data)
            prediction = np.expm1(pred_log)[0]

            lower_bound = max(0, prediction - MODEL_RMSE)
            upper_bound = prediction + MODEL_RMSE

            st.metric(
                "Predicted Revenue",
                f"${prediction:,.2f}"
            )

            st.metric(
                "Prediction Range",
                f"${lower_bound:,.1f} - {upper_bound:,.1f}"
            )

            st.caption(
                "Prediction is an estimate and should be interpreted alongside event context."
            )

            st.markdown("### Prediction Signals")

            signals = get_prediction_signals(
                max_review_count=max_review_count,
                total_engagement=total_engagement,
                num_shares=num_shares,
                avg_movie_rating=avg_movie_rating,
                is_double_feature=is_double_feature,
                collab=collab
            )

            for signal in signals:
                st.write(f"✔️ {signal}")

            if prediction < 75:
                st.warning(
                    "Expected low-revenue event. Consider preparing fewer concessions."
                )
            elif prediction < 200:
                st.success(
                    "Expected moderate-revenue event. Standard preparation is likely sufficient."
                )
            else:
                st.success(
                    "Expected high-revenue event. Consider preparing additional concessions."
                )

        with st.expander("View model input"):
            st.dataframe(input_data)

def get_prediction_signals(
    max_review_count,
    total_engagement,
    num_shares,
    avg_movie_rating,
    is_double_feature,
    collab
):
    signals = []

    if max_review_count >= 300000:
        signals.append("High movie popularity")
    elif max_review_count >= 100000:
        signals.append("Moderate movie popularity")
    else:
        signals.append("Lower movie popularity")

    if total_engagement >= 150:
        signals.append("Strong Instagram engagement")
    elif total_engagement >= 75:
        signals.append("Moderate Instagram engagement")
    else:
        signals.append("Lower Instagram engagement")

    if num_shares >= 15:
        signals.append("High share activity")

    if avg_movie_rating >= 4.0:
        signals.append("Highly rated movie")

    if is_double_feature:
        signals.append("Double feature event")

    if collab == "Yes":
        signals.append("Collaboration event")

    return signals
