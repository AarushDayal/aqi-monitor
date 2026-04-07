import pandas as pd
import numpy as np

RAW_PATH   = "../Data/raw/original_data.csv"
TRAIN_PATH = "../Data/processed/train_temporal.csv"
TEST_PATH  = "../Data/processed/test_temporal.csv"

print("Loading raw data...")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Raw shape: {df.shape}")

# ─────────────────────────────────────────
# Drop rows with missing AQI
# ─────────────────────────────────────────
df = df.dropna(subset=["AQI"])
print(f"After dropping AQI nulls: {df.shape}")

# ─────────────────────────────────────────
# Drop extreme AQI outliers
# ─────────────────────────────────────────
df = df[df["AQI"] <= 500]
print(f"After capping AQI at 500: {df.shape}")

# ─────────────────────────────────────────
# Parse datetime + extract temporal features
# ─────────────────────────────────────────
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df = df.dropna(subset=["Datetime"]).sort_values(by=["StationId", "Datetime"])

# Create future targets safely using time-shifting merge
print("Engineering future AQI targets...")
df_target = df[["StationId", "Datetime", "AQI"]].copy()

# --- NEW: Create lagged features ---
print("Engineering lagged AQI features...")
for hours, label in zip([1, 2, 24], ["AQI_lag1", "AQI_lag2", "AQI_lag24"]):
    df_temp = df_target.copy()
    df_temp["Datetime"] = df_temp["Datetime"] + pd.Timedelta(hours=hours)
    df_temp = df_temp.rename(columns={"AQI": label})
    df_temp = df_temp.drop_duplicates(subset=["StationId", "Datetime"])
    df = pd.merge(df, df_temp, on=["StationId", "Datetime"], how="left")

# Drop rows missing past targets so we have clean features
df = df.dropna(subset=["AQI_lag1", "AQI_lag2", "AQI_lag24"])
print(f"After dropping missing lag features: {df.shape}")

for hours, label in zip([8, 24, 168], ["AQI_8h", "AQI_24h", "AQI_168h"]):
    df_temp = df_target.copy()
    df_temp["Datetime"] = df_temp["Datetime"] - pd.Timedelta(hours=hours)
    df_temp = df_temp.rename(columns={"AQI": label})
    df_temp = df_temp.drop_duplicates(subset=["StationId", "Datetime"])
    df = pd.merge(df, df_temp, on=["StationId", "Datetime"], how="left")

# Drop rows missing future targets so we have clean training labels
df = df.dropna(subset=["AQI_8h", "AQI_24h", "AQI_168h"])
print(f"After dropping missing future targets: {df.shape}")


df["hour"]    = df["Datetime"].dt.hour
df["month"]   = df["Datetime"].dt.month
df["day"]     = df["Datetime"].dt.dayofweek
df["quarter"] = df["Datetime"].dt.quarter
df["season"]  = df["month"].map({
    12: 0, 1: 0, 2: 0,   # winter
    3: 1,  4: 1, 5: 1,   # spring
    6: 2,  7: 2, 8: 2,   # summer
    9: 3, 10: 3, 11: 3   # autumn
})

# Cyclical encoding for hour and month
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ─────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────
df = df.drop(columns=["NO", "O3", "Benzene", "Xylene"], errors="ignore")

df["PM_ratio"]      = df["PM2.5"] / (df["PM10"] + 1e-5)
df["pollution_load"] = df["PM2.5"] + df["PM10"] + df["NO2"] + df["CO"]
df["NO_proxy"]      = df["NOx"] - df["NO2"]

# ─────────────────────────────────────────
# Drop unnecessary columns
# ─────────────────────────────────────────
drop_cols = ["StationId", "Datetime", "AQI_Bucket"]
df = df.drop(columns=drop_cols, errors="ignore")

# ─────────────────────────────────────────
# Drop remaining nulls
# ─────────────────────────────────────────
before = len(df)
df = df.dropna()
print(f"After dropping remaining nulls: {df.shape} (dropped {before - len(df)})")

# ─────────────────────────────────────────
# Print feature summary
# ─────────────────────────────────────────
print(f"\nFeatures: {list(df.drop('AQI', axis=1).columns)}")
print(f"AQI stats:\n{df['AQI'].describe()}")

# ─────────────────────────────────────────
# Train/test split — time based
# ─────────────────────────────────────────
split = int(len(df) * 0.8)
train = df.iloc[:split]
test  = df.iloc[split:]

train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)

print(f"\nTrain saved → {TRAIN_PATH} ({len(train)} rows)")
print(f"Test saved  → {TEST_PATH} ({len(test)} rows)")
print("Done.")
