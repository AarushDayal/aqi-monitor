import pandas as pd
import numpy as np
import joblib
import time
import os
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TRAIN_PATH = "../Data/processed/train_temporal.csv"
TEST_PATH = "../Data/processed/test_temporal.csv"
MODEL_PATH = "../models/saved/multi_horizon_model.pkl"

def engineer_features(df):
    df = df.copy()
    drop_cols = ["NO", "O3", "Benzene", "Xylene"]
    df = df.drop(columns=drop_cols, errors="ignore")
    df["PM_ratio"] = df["PM2.5"] / (df["PM10"] + 1e-5)
    df["pollution_load"] = df["PM2.5"] + df["PM10"] + df["NO2"] + df["CO"]
    if "NOx" in df.columns and "NO2" in df.columns:
        df["NO_proxy"] = df["NOx"] - df["NO2"]
    return df

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

targets = ["AQI_8h", "AQI_24h", "AQI_168h"]
features = [c for c in train_df.columns if c not in targets and c != "AQI"]

X_train = train_df[features]
y_train = train_df[targets]
X_test = test_df[features]
y_test = test_df[targets]

# Log-transform targets just like in model3.py
y_train_log = np.log1p(y_train)

# Base model
base_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    random_state=42,
    tree_method="hist",
    verbosity=0
)

# MultiOutputRegressor wraps the base model
model = MultiOutputRegressor(base_model, n_jobs=1)

print(f"Training Multi-Horizon Model on {X_train.shape[0]} rows and {len(features)} features...")
start = time.time()
model.fit(X_train, y_train_log)
elapsed = time.time() - start

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_pred = np.clip(y_pred, 0, 500)

print(f"\nTime taken: {elapsed:.2f}s")
for i, target in enumerate(targets):
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"\nPerformance on {target}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

os.makedirs("../models/saved", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")
