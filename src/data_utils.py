import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import RAW_DATA_DIR

# Fetch raw Citi Bike data from cloud storage
def fetch_raw_citibike_data(year: int, month: int) -> Path:
    URL = f"https://s3.amazonaws.com/tripdata/{year}{month:02}-citibike-tripdata.csv.zip"
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f"citibike_{year}_{month:02}.zip"
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f"{URL} is not available")

# Process a single month's Citi Bike data
def load_and_process_citibike_data(year: int, month: int) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"cleaned_citibike_top3_{year}.csv"
    df = pd.read_csv(path, parse_dates=["started_at"])

    # Only keep the target month
    df = df[df['started_at'].dt.month == month]

    # Clean: remove missing station info
    df = df.dropna(subset=["started_at", "start_station_name"])

    df = df[["started_at", "start_station_name"]]
    df = df.rename(columns={"started_at": "pickup_datetime", "start_station_name": "pickup_location_id"})
    return df

# Aggregate to hourly ride counts
def transform_to_hourly_ts(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_hour"] = df["pickup_datetime"].dt.floor("H")
    ride_counts = (
        df.groupby(["pickup_hour", "pickup_location_id"])
        .size()
        .reset_index(name="rides")
    )
    return fill_missing_rides_full_range(ride_counts, "pickup_hour", "pickup_location_id", "rides")

# Fill missing (hour, location) slots
def fill_missing_rides_full_range(df, hour_col, location_col, rides_col):
    df[hour_col] = pd.to_datetime(df[hour_col])
    full_hours = pd.date_range(start=df[hour_col].min(), end=df[hour_col].max(), freq="H")
    all_locations = df[location_col].unique()
    full_combinations = pd.DataFrame(
        [(hour, loc) for hour in full_hours for loc in all_locations],
        columns=[hour_col, location_col],
    )
    merged = pd.merge(full_combinations, df, on=[hour_col, location_col], how="left")
    merged[rides_col] = merged[rides_col].fillna(0).astype(int)
    return merged

# Transform into sliding-window features
def transform_features_targets(df, feature_col="rides", window_size=12, step_size=1):
    location_ids = df["pickup_location_id"].unique()
    all_data = []

    for loc in location_ids:
        temp = df[df["pickup_location_id"] == loc].reset_index(drop=True)
        values = temp[feature_col].values
        times = temp["pickup_hour"].values

        if len(values) <= window_size:
            continue

        rows = []
        for i in range(0, len(values) - window_size, step_size):
            features = values[i : i + window_size]
            target = values[i + window_size]
            target_time = times[i + window_size]
            rows.append(np.append(features, [target, loc, target_time]))

        cols = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
        df_features = pd.DataFrame(rows, columns=cols + ["target", "pickup_location_id", "pickup_hour"])
        all_data.append(df_features)

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# Split into train-test
def split_ts_data(df, cutoff_date, target_col="target"):
    train = df[df["pickup_hour"] < cutoff_date].reset_index(drop=True)
    test = df[df["pickup_hour"] >= cutoff_date].reset_index(drop=True)
    return train.drop(columns=[target_col]), train[target_col], test.drop(columns=[target_col]), test[target_col]
