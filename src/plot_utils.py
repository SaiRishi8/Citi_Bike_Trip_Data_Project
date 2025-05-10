from datetime import timedelta
from typing import Optional

import pandas as pd
import plotly.express as px


def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: str,
    predictions: Optional[pd.Series] = None,
):
    """
    Plots the time series data for a specific Citi Bike station.

    Args:
        features (pd.DataFrame): Feature data with historical ride counts and metadata.
        targets (pd.Series): Actual ride counts (target values).
        row_id (str): Name of the start station to visualize.
        predictions (Optional[pd.Series]): Optional predicted values.

    Returns:
        Plotly Figure object.
    """
    station_features = features[features["start_station_name"] == row_id]
    actual_target = targets[features["start_station_name"] == row_id].values[0]

    # Historical ride count features
    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]
    time_series_values = station_features[time_series_columns].values.flatten().tolist() + [actual_target]

    # Construct hourly datetime index
    pickup_hour = pd.Timestamp(station_features["pickup_hour"].values[0])
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h"
    )

    title = f"Station: {row_id}, Hour: {pickup_hour}"

    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        title=title,
        template="plotly_white",
        markers=True,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    fig.add_scatter(
        x=[pickup_hour],
        y=[actual_target],
        mode="markers",
        marker=dict(size=10, color="green"),
        name="Actual Value"
    )

    if predictions is not None:
        predicted_value = predictions[features["start_station_name"] == row_id].values[0]
        fig.add_scatter(
            x=[pickup_hour],
            y=[predicted_value],
            mode="markers",
            marker=dict(size=12, color="red", symbol="x"),
            name="Prediction"
        )

    return fig


def plot_prediction(features: pd.DataFrame, prediction: pd.DataFrame):
    """
    Plot historical ride data and predicted ride count for a Citi Bike station.

    Args:
        features (pd.DataFrame): Feature data for one station (1 row).
        prediction (pd.DataFrame): Predicted result with 'predicted_demand' column.

    Returns:
        Plotly figure.
    """
    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]
    ride_values = [features[col].iloc[0] for col in time_series_columns]
    prediction_value = prediction["predicted_demand"].iloc[0]
    all_values = ride_values + [prediction_value]

    pickup_hour = pd.Timestamp(features["pickup_hour"].iloc[0])
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h"
    )

    df_plot = pd.DataFrame({
        "datetime": time_series_dates,
        "rides": all_values
    })

    title = f"Prediction for: {features['start_station_name'].iloc[0]}, Hour: {pickup_hour}"

    fig = px.line(
        df_plot,
        x="datetime",
        y="rides",
        title=title,
        markers=True,
        template="plotly_white",
        labels={"datetime": "Time", "rides": "Ride Counts"},
    )

    fig.add_scatter(
        x=[pickup_hour],
        y=[prediction_value],
        mode="markers",
        marker=dict(size=10, color="red", symbol="x"),
        name="Prediction"
    )

    return fig
