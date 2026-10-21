import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Bike Rental Prediction", page_icon="🚲")

@st.cache_resource
def load_model():
    return joblib.load(
        Path(__file__).parent / "best_bike_rental_model.joblib"
    )

model = load_model()

st.title("🚲 Bike-Sharing Rental Prediction")
st.write("Enter the details below to predict the expected rental count.")

with st.form("prediction_form"):
    season = st.selectbox(
        "Season", ["spring", "summer", "fall", "winter"]
    )
    yr = st.selectbox("Year", [0, 1], format_func=lambda x: 2011 + x)
    mnth = st.selectbox("Month", list(range(1, 13)))
    hr = st.slider("Hour", 0, 23, 8)
    holiday = st.selectbox("Holiday", [0, 1])
    weekday = st.selectbox("Weekday (0–6)", list(range(7)))
    workingday = st.selectbox("Working day", [0, 1])
    weathersit = st.selectbox("Weather situation", [1, 2, 3, 4])

    temp = st.slider("Temperature (dataset scale)", 0.0, 1.0, 0.5)
    atemp = st.slider("Apparent temperature (dataset scale)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (dataset scale)", 0.0, 1.0, 0.6)
    windspeed = st.slider("Windspeed (dataset scale)", 0.0, 1.0, 0.2)
    day = st.slider("Day of month", 1, 31, 15)

    submitted = st.form_submit_button("Predict Rentals")

if submitted:
    input_data = pd.DataFrame([{
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "day": day
    }])

    try:
        prediction = max(0, float(model.predict(input_data)[0]))
        st.success(f"Estimated bike rentals: {prediction:,.0f}")
        with st.expander("View input details"):
            st.dataframe(input_data, use_container_width=True)
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.write(
            "Check that the feature names, categories, and data scales "
            "match those used to train your model."
        )

st.caption("Machine Learning project • Bike Rental Demand Prediction")
