# src/feature_pipeline.py
import pandas as pd
import hopsworks
import os

# Step 1: Load raw Citi Bike trip data
csv_path = os.path.join(os.path.dirname(__file__), "..", "cleaned_citibike_top3_2023.csv")
csv_path = os.path.abspath(csv_path)

df = pd.read_csv(csv_path, low_memory=False)

# Step 2: Convert started_at to datetime and aggregate to hourly ride counts
df['started_at'] = pd.to_datetime(df['started_at'])
df['rides'] = 1  # each row is a ride
df = df.set_index('started_at').resample('H').sum().reset_index()

# Step 3: Create lag features
for lag in range(1, 29):
    df[f'lag_{lag}'] = df['rides'].shift(lag)

# Step 4: Drop missing rows due to lagging
df.dropna(inplace=True)

# Step 5: Extract additional columns for keys
df['date'] = df['started_at'].dt.floor('D')  # datetime-compatible type
df['hour'] = df['started_at'].dt.hour
df['start_station_name'] = "Top_Station"  # replace with actual if available

# Step 6: Keep only relevant columns
features = [f'lag_{i}' for i in range(1, 29)] + ['date', 'hour', 'start_station_name', 'rides']
df = df[features]

# Step 7: Connect to Hopsworks
project = hopsworks.login(
    project="Citi_Bike_TripData",
    api_key_value="eVrgcmUQIaYJz4kj.QNITpj9s3ieWAofZVNhhPtsjGng1ra5ZA9BsSGNRuI6i9WLGojdUuD0i0TBKfIx1"
)
fs = project.get_feature_store()

# Step 8: Create or get the feature group
fg = fs.get_or_create_feature_group(
    name="citibike_features_hourly",
    version=1,
    primary_key=["start_station_name", "date", "hour"],
    event_time="date",
    description="Hourly lag features for Citi Bike demand prediction"
)

# Step 9: Insert data in chunks to avoid disconnect errors
batch_size = 5000
for i in range(0, len(df), batch_size):
    chunk = df.iloc[i:i+batch_size]
    fg.insert(chunk, wait=True)
    print(f"Inserted rows {i} to {i+len(chunk)-1}")

print("Feature group successfully inserted into Hopsworks.")
