import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Define directories
PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PARENT_DIR / "models"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Hopsworks API and Project Details
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# Feature group and view names for Citi Bike
FEATURE_GROUP_NAME = "citibike_features_hourly"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "citibike_hourly_feature_view"
FEATURE_VIEW_VERSION = 1

# Model details
MODEL_NAME = "citibike_demand_model"
MODEL_VERSION = 1

# Feature group to store predictions
FEATURE_GROUP_MODEL_PREDICTION = "citibike_demand_predictions"
