import sys
from pathlib import Path

# Ensure parent directory is in Python path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions

# Streamlit title
st.title("Mean Absolute Error (MAE) by Start Hour - Citi Bike Demand")

# Sidebar controls
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Analyze",
    min_value=12,
    max_value=24 * 28,  # up to 28 days of hourly data
    value=24,
    step=1,
)

# Fetch hourly actual and predicted data
st.write(f"Loading Citi Bike data for the past {past_hours} hours...")
actual_df = fetch_hourly_rides(past_hours)
predicted_df = fetch_predictions(past_hours)

# Join on start_station_name and pickup_hour
merged_df = pd.merge(actual_df, predicted_df, on=["start_station_name", "pickup_hour"])

# Calculate MAE
merged_df["absolute_error"] = (merged_df["predicted_demand"] - merged_df["rides"]).abs()

# Group by pickup_hour to compute average MAE per hour
mae_by_hour = merged_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

# Plot using Plotly
fig = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"MAE by Hour for Citi Bike Demand (Last {past_hours} Hours)",
    labels={"pickup_hour": "Start Hour (UTC)", "MAE": "Mean Absolute Error"},
    markers=True,
)

# Display
st.plotly_chart(fig)
st.write(f"**Average MAE over this period:** `{mae_by_hour['MAE'].mean():.2f}`")
