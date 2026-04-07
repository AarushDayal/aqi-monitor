import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
TRAIN_PATH = "../Data/processed/train_temporal.csv"
TEST_PATH = "../Data/processed/test_temporal.csv"
MODEL_PATH = "../models/saved/stacking_ensemble.pkl"
RESULT_PATH = "../Data/processed/model3_results.csv"


# ─────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    drop_cols = ["NO", "O3", "Benzene", "Xylene"]
    df = df.drop(columns=drop_cols, errors="ignore")

    df["PM_ratio"] = df["PM2.5"] / (df["PM10"] + 1e-5)
    df["pollution_load"] = df["PM2.5"] + df["PM10"] + df["NO2"] + df["CO"]

    if "NOx" in df.columns and "NO2" in df.columns:
        df["NO_proxy"] = df["NOx"] - df["NO2"]

    return df


# ─────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

X_train = train_df.drop(["AQI", "AQI_8h", "AQI_24h", "AQI_168h"], axis=1, errors="ignore")
y_train = train_df["AQI"]
X_test = test_df.drop(["AQI", "AQI_8h", "AQI_24h", "AQI_168h"], axis=1, errors="ignore")
y_test = test_df["AQI"]

print(f"Train shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")
print(f"Features:    {list(X_train.columns)}")


# Log transform target
y_train = np.log1p(y_train)
y_test_raw = y_test.copy()  # keep original for evaluation
y_test = np.log1p(y_test)

# ─────────────────────────────────────────
# Base Models
# ─────────────────────────────────────────
estimators = [
    (
        "xgb",
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=4,
            tree_method="hist",
            device="cuda",
            random_state=42,
            verbosity=0,
        ),
    ),
    (
        "lgbm",
        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=4,
            device="cpu",
            random_state=42,
            verbose=-1,
        ),
    ),
    (
        "rf",
        RandomForestRegressor(
            n_estimators=50,
            n_jobs=2,
            max_depth=15,
            max_samples=0.3,
            bootstrap=True,
            random_state=42,
        ),
    ),
    (
        "et",
        ExtraTreesRegressor(
            n_estimators=50,
            n_jobs=2,
            max_depth=15,
            bootstrap=True,
            max_samples=0.3,
            random_state=42,
        ),
    ),
]

# ─────────────────────────────────────────
# Stacking Ensemble
# ─────────────────────────────────────────
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=3,
    n_jobs=1,
    passthrough=False,
)

print("\nTraining Stacking Ensemble (this will take a while)...")
start = time.time()
stack.fit(X_train, y_train)
elapsed = time.time() - start

# ─────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────

y_pred_log = stack.predict(X_test)
y_pred = np.expm1(y_pred_log)  # back to original AQI scale
y_pred = np.clip(y_pred, 0, 500)  # clip negatives just in case

rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
mae = mean_absolute_error(y_test_raw, y_pred)
r2 = r2_score(y_test_raw, y_pred)

print(f"\n{'─' * 40}")
print(f"Stacking Ensemble Results")
print(f"{'─' * 40}")
print(f"RMSE:     {rmse:.4f}")
print(f"MAE:      {mae:.4f}")
print(f"R²:       {r2:.4f}")
print(f"Time:     {elapsed:.2f}s")
print(f"{'─' * 40}")

# ─────────────────────────────────────────
# Save model + results
# ─────────────────────────────────────────
joblib.dump(stack, MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

results = pd.DataFrame(
    [
        {
            "Model": "Stacking Ensemble",
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "Train Time (s)": round(elapsed, 2),
        }
    ]
)
results.to_csv(RESULT_PATH, index=False)
print(f"Results saved → {RESULT_PATH}")

