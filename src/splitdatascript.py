import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("station_hour.csv")


# Remove rows where AQI is missing
df = df.dropna(subset=["AQI"])

# Features (inputs)
features = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]

X = df[features]
y = df["AQI"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # 20% test data
    random_state=42,
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# Combine features and target
train_df = X_train.copy()
train_df["AQI"] = y_train

test_df = X_test.copy()
test_df["AQI"] = y_test

# Save to files
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Training and testing files saved.")
