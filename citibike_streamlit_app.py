import streamlit as st
import pandas as pd
import hopsworks
import joblib
import matplotlib.pyplot as plt

# ----------------- Setup -------------------
st.set_page_config(page_title="CitiBike Demand App", layout="wide")

st.title("🚲 CitiBike Demand Prediction & Monitoring")

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load model from file
model = joblib.load("citibike_best_model.pkl")  # Ensure this file is in your app directory

# ----------------- UI Tabs -------------------
tab1, tab2 = st.tabs(["📈 Prediction", "📊 Monitoring"])

# ----------------- Tab 1: Prediction -------------------
with tab1:
    st.header("Predict CitiBike Rides")

    hour = st.slider("Hour of the Day", 0, 23, 14)
    station = st.selectbox("Station", ["Top_Station"])

    # Simulated lag values
    lags = {f"lag_{i}": st.number_input(f"Lag {i} value", min_value=0, value=10-i) for i in range(1, 29)}

    input_data = pd.DataFrame([lags])
    input_data["hour"] = hour
    input_data["start_station_name"] = station

    if st.button("Predict"):
        pred = model.predict(input_data)[0]
        st.success(f"🚴 Predicted rides: {pred:.2f}")

# ----------------- Tab 2: Monitoring -------------------
with tab2:
    st.header("Monitoring Predictions from Hopsworks")

    try:
        pred_fg = fs.get_feature_group("citibike_demand_predictions", version=1)
        df = pred_fg.read()

        st.subheader("📉 Latest Predictions")
        st.write(df.sort_values("inference_time", ascending=False).head(10))

        st.subheader("📊 Ride Predictions Over Time")
        chart_df = df.sort_values("inference_time").tail(100)
        plt.figure(figsize=(10, 4))
        plt.plot(chart_df["inference_time"], chart_df["predicted_rides"], label="Predicted Rides")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error("Could not load monitoring data from Hopsworks.")
        st.exception(e)
