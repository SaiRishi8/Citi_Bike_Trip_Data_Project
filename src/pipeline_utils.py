import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


# Function to calculate the average rides over the last 4 weeks
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    last_4_weeks_columns = [
        f"rides_t-{7*24}",   # 1 week ago
        f"rides_t-{14*24}",  # 2 weeks ago
        f"rides_t-{21*24}",  # 3 weeks ago
        f"rides_t-{28*24}",  # 4 weeks ago
    ]

    for col in last_4_weeks_columns:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)
    return X


# FunctionTransformer to inject average ride features
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)


# Custom transformer for extracting temporal features
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek

        # Drop non-numeric or irrelevant features
        return X_.drop(columns=["pickup_hour", "start_station_name"])


# Instantiate the temporal feature engineer
add_temporal_features = TemporalFeatureEngineer()


# Full pipeline construction
def get_pipeline(**hyper_params):
    """
    Returns a complete sklearn pipeline for Citi Bike demand prediction.

    Parameters:
        hyper_params: Optional LightGBM hyperparameters.

    Returns:
        sklearn.pipeline.Pipeline object.
    """
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyper_params),
    )
   
    return pipeline