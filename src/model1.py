import time
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# -------------------------------
# Load training data
# -------------------------------
train_df = pd.read_csv("train_data.csv")

X_train = train_df.drop("AQI", axis=1)
y_train = train_df["AQI"]

# -------------------------------
# Load test data
# -------------------------------
test_df = pd.read_csv("test_data.ycsv")

X_test = test_df.drop("AQI", axis=1)
y_test = test_df["AQI"]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# -------------------------------
# Evaluation function
# -------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test, results):
    print(f"\nTraining {name}...")
    start = time.time()

    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([name, rmse, mae, r2, train_time])
    print(f"{name} done in {train_time:.2f} seconds")


# -------------------------------
# Model definitions
# -------------------------------
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=42
    ),
}

# -------------------------------
# Train and compare
# -------------------------------
results = []

for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test, results)

# -------------------------------
# Results table
# -------------------------------
results_df = pd.DataFrame(
    results, columns=["Model", "RMSE", "MAE", "R2 Score", "Train Time (s)"]
)

print("\nModel Comparison:")
print(results_df.sort_values("RMSE"))


results_df.to_csv("model_results.csv", index=False)
print("\nResults saved to model_results.csv")
