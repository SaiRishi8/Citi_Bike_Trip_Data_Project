import os
import sys
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from config import DATA_DIR
from inference import (
    get_model_predictions,
    load_batch_of_features_from_store,
    load_model_from_registry,
)
from plot_utils import plot_aggregated_time_series

# UI setup
st.set_page_config(layout="wide")
current_date = pd.Timestamp.now(tz="UTC")
st.title("Citi Bike Demand Prediction")
st.header(f"Predicted demand for {current_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")

progress_bar = st.sidebar.progress(0)
N_STEPS = 5

# Step 1: Load NYC zones shapefile
with st.spinner("Loading Citi Bike zones shapefile"):
    shapefile_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    shapefile_dir = DATA_DIR / "citibike_zones"
    zip_path = shapefile_dir / "zones.zip"
    shapefile_path = shapefile_dir / "taxi_zones.shp"
    shapefile_dir.mkdir(parents=True, exist_ok=True)

    if not shapefile_path.exists():
        import requests, zipfile
        r = requests.get(shapefile_url)
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(shapefile_dir)

    geo_df = gpd.read_file(shapefile_path).to_crs("epsg:4326")
    st.sidebar.success("Shapefile loaded")
    progress_bar.progress(1 / N_STEPS)

# Step 2: Load feature batch
with st.spinner("Loading features from Hopsworks"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.success("Features loaded")
    progress_bar.progress(2 / N_STEPS)

# Step 3: Load model
with st.spinner("Loading model from Hopsworks"):
    model = load_model_from_registry()
    st.sidebar.success("Model loaded")
    progress_bar.progress(3 / N_STEPS)

# Step 4: Generate predictions
with st.spinner("Making predictions"):
    predictions = get_model_predictions(model, features)
    st.sidebar.success("Predictions computed")
    progress_bar.progress(4 / N_STEPS)

# Step 5: Visualize predictions on map
with st.spinner("Visualizing predictions"):
    # Merge shapefile with predictions
    geo_df["predicted_demand"] = geo_df["LocationID"].map(
        predictions.set_index("pickup_location_id")["predicted_demand"]
    ).fillna(0)

    m = folium.Map(location=[40.75, -73.97], zoom_start=12)
    folium.Choropleth(
        geo_data=geo_df,
        data=geo_df,
        columns=["LocationID", "predicted_demand"],
        key_on="feature.properties.LocationID",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Predicted Rides",
    ).add_to(m)

    st.subheader("Citi Bike Demand Heatmap")
    st_folium(m, width=800, height=600)
    progress_bar.progress(5 / N_STEPS)

# Top demand zones
st.subheader("Top 10 Locations by Predicted Demand")
top10 = predictions.sort_values("predicted_demand", ascending=False).head(10)
st.dataframe(top10)

# Optional: Line plot per zone
st.subheader("Predicted Ride Trends for Top Locations")
for row in top10.itertuples():
    fig = plot_aggregated_time_series(
        features=features,
        targets=predictions["predicted_demand"],
        row_id=row.pickup_location_id,
        predictions=predictions["predicted_demand"]
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
