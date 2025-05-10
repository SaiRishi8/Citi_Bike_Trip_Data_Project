# pipelines/inference_pipeline.py
import pandas as pd
import hopsworks
import mlflow.pyfunc
from datetime import datetime

# Step 1: Login to Hopsworks
project = hopsworks.login(
    project="Citi_Bike_TripData",
    api_key_value="eVrgcmUQIaYJz4kj.QNITpj9s3ieWAofZVNhhPtsjGng1ra5ZA9BsSGNRuI6i9WLGojdUuD0i0TBKfIx1"
)
fs = project.get_feature_store()

# Step 2: Read feature group
fg = fs.get_feature_group("citibike_features_hourly", version=1)
df = fg.read()

# Step 3: Sort and take latest row
latest_df = df.sort_values("date", ascending=False).dropna().head(1)

# Step 4: Extract only the lag features used in training
feature_columns = [f"lag_{i}" for i in range(1, 29)]
X = latest_df[feature_columns]

# Step 5: Load model from MLflow
mlflow.set_tracking_uri("https://dagshub.com/SaiRishi9/Citi_Bike_tripdata.mlflow")
model = mlflow.pyfunc.load_model("models:/citibike_demand_model/1")

# Step 6: Predict
y_pred = model.predict(X)

# Step 7: Store result in new feature group
result = latest_df[["start_station_name", "date", "hour"]].copy()
result["predicted_rides"] = y_pred
result["inference_time"] = datetime.now()

pred_fg = fs.get_or_create_feature_group(
    name="citibike_demand_predictions",
    version=1,
    primary_key=["start_station_name", "date", "hour"],
    event_time="inference_time",
    description="Predicted hourly demand using production model"
)
pred_fg.insert(result, wait=True)

print("Inference completed and predictions logged to Hopsworks.")
