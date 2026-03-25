import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("original_data.csv")

# Drop rows where AQI is missing
df = df.dropna(subset=["AQI"])

# Select relevant columns
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
    "AQI",
]

df_selected = df[features]

# Compute correlation matrix
corr_matrix = df_selected.corr()

# Save correlation values as CSV (optional but useful)
corr_matrix.to_csv("correlation_values.csv")

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Matrix of Pollutants and AQI")
plt.tight_layout()

# Save image
plt.savefig("correlation_heatmap.png", dpi=300)

# Show plot
plt.show()

print("Correlation matrix saved as:")
print("- correlation_values.csv")
print("- correlation_heatmap.png")
