import pandas as pd
import hopsworks
import mlflow
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Step 1: Login and load feature group
project = hopsworks.login(
    project="Citi_Bike_TripData",
    api_key_value="eVrgcmUQIaYJz4kj.QNITpj9s3ieWAofZVNhhPtsjGng1ra5ZA9BsSGNRuI6i9WLGojdUuD0i0TBKfIx1"
)
fs = project.get_feature_store()
fg = fs.get_feature_group("citibike_features_hourly", version=1)
df = fg.read()

# Step 2: Preprocessing
df['date'] = pd.to_datetime(df['date'])  # ensure datetime
df['start_station_name'] = df['start_station_name'].astype('category')

# Drop non-numeric or unsupported columns
X = df.drop(columns=["rides", "date"])
y = df["rides"]

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

# Step 6: Log to MLflow
mlflow.set_tracking_uri("https://dagshub.com/SaiRishi9/Citi_Bike_tripdata.mlflow")
mlflow.set_experiment("CitiBike_Demand_Prediction")
mlflow.login(os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"])

with mlflow.start_run(run_name="LightGBM_All28Lags"):
    mlflow.log_metric("mae", mae)
    mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name="citibike_demand_model")
    mlflow.set_tag("model_type", "LightGBM")
    mlflow.set_tag("feature_set", "All 28 lag features")
