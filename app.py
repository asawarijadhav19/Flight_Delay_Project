
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND FILES
# =========================================================

model = joblib.load("best_flight_delay_model.pkl")
model_features = joblib.load("model_features.pkl")

origin_frequency = joblib.load("origin_frequency.pkl")
destination_frequency = joblib.load("destination_frequency.pkl")


# =========================================================
# CONVERT AIRPORT KEYS TO STRING
# =========================================================

origin_frequency_str = {
    str(k): v for k, v in origin_frequency.items()
}

destination_frequency_str = {
    str(k): v for k, v in destination_frequency.items()
}


# =========================================================
# TITLE
# =========================================================

st.title("✈️ Flight Delay Prediction System")

st.write(
    "Machine Learning Based Flight Delay Prediction "
    "using Decision Tree / Random Forest"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📌 Project Information")

st.sidebar.write("### Features Used")

st.sidebar.write("""
- Year
- Month
- Day
- Day of Week
- Flight Number
- Scheduled Departure
- Scheduled Time
- Distance
- Origin Airport
- Destination Airport
- Airline
""")

st.sidebar.write("### Machine Learning Models")

st.sidebar.write("""
- Decision Tree
- Random Forest
- K-Means Clustering
""")


# =========================================================
# INPUT SECTION
# =========================================================

st.header("🛫 Enter Flight Details")

col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    year = st.number_input(
        "Year",
        min_value=2015,
        max_value=2030,
        value=2015,
        step=1
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1,
        step=1
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=1,
        step=1
    )

    day_of_week = st.number_input(
        "Day of Week",
        min_value=1,
        max_value=7,
        value=1,
        step=1,
        help="1 = Monday, 7 = Sunday"
    )

    flight_number = st.number_input(
        "Flight Number",
        min_value=1,
        max_value=9999,
        value=100,
        step=1
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    scheduled_departure = st.number_input(
        "Scheduled Departure",
        min_value=0,
        max_value=2359,
        value=800,
        step=1,
        help="Enter time in HHMM format. Example: 8:00 AM = 800"
    )

    scheduled_time = st.number_input(
        "Scheduled Flight Time (minutes)",
        min_value=1,
        max_value=1000,
        value=120,
        step=1
    )

    distance = st.number_input(
        "Distance (miles)",
        min_value=1,
        max_value=5000,
        value=500,
        step=1
    )

    origin_airport = st.selectbox(
        "Origin Airport",
        sorted([str(x) for x in origin_frequency.keys()])
    )

    destination_airport = st.selectbox(
        "Destination Airport",
        sorted([str(x) for x in destination_frequency.keys()])
    )


# =========================================================
# AIRLINE SELECTION
# =========================================================

st.subheader("✈️ Select Airline")

airline_columns = [
    feature.replace("AIRLINE_", "")
    for feature in model_features
    if feature.startswith("AIRLINE_")
]


airline = st.selectbox(
    "Airline",
    airline_columns
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

if st.button(
    "🔮 Predict Flight Delay",
    use_container_width=True
):

    # -----------------------------------------------------
    # CREATE EMPTY INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_features
    )


    # -----------------------------------------------------
    # ADD NUMERIC VALUES
    # -----------------------------------------------------

    input_data["YEAR"] = year

    input_data["MONTH"] = month

    input_data["DAY"] = day

    input_data["DAY_OF_WEEK"] = day_of_week

    input_data["FLIGHT_NUMBER"] = flight_number

    input_data["SCHEDULED_DEPARTURE"] = scheduled_departure

    input_data["SCHEDULED_TIME"] = scheduled_time

    input_data["DISTANCE"] = distance


    # -----------------------------------------------------
    # AIRPORT FREQUENCY ENCODING
    # -----------------------------------------------------

    input_data["ORIGIN_AIRPORT_FREQ"] = (
        origin_frequency_str.get(origin_airport, 0)
    )

    input_data["DESTINATION_AIRPORT_FREQ"] = (
        destination_frequency_str.get(destination_airport, 0)
    )


    # -----------------------------------------------------
    # AIRLINE ONE-HOT ENCODING
    # -----------------------------------------------------

    airline_column = "AIRLINE_" + airline

    if airline_column in input_data.columns:

        input_data[airline_column] = 1


    # -----------------------------------------------------
    # MAKE SURE FEATURE ORDER IS CORRECT
    # -----------------------------------------------------

    input_data = input_data[model_features]


    # -----------------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(input_data)[0]


    # -----------------------------------------------------
    # PREDICTION PROBABILITY
    # -----------------------------------------------------

    probability = model.predict_proba(input_data)[0][1]


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.divider()

    st.header("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Flight is likely to be DELAYED"
        )

        st.metric(
            "Delay Probability",
            f"{probability * 100:.2f}%"
        )

        st.write(
            "The model predicts that this flight has a "
            "higher chance of arriving more than 15 minutes late."
        )


    else:

        st.success(
            "✅ Flight is likely to be ON TIME"
        )

        st.metric(
            "Delay Probability",
            f"{probability * 100:.2f}%"
        )

        st.write(
            "The model predicts that this flight is likely "
            "to arrive on time."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Flight Delay Prediction System | "
    "Machine Learning Project | "
    "B.Tech Artificial Intelligence and Data Science"
)

