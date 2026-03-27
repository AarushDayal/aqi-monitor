# src/data_logger.py

import pandas as pd
import os

FILE_PATH = "src/data/aqi_log.csv"


def log_data(features, aqi):
    """
    Save current AQI + features to CSV
    """
    row = {**features, "aqi": aqi}

    df = pd.DataFrame([row])

    if not os.path.exists(FILE_PATH):
        df.to_csv(FILE_PATH, index=False)
    else:
        df.to_csv(FILE_PATH, mode="a", header=False, index=False)
