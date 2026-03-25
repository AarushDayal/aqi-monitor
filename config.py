import os


def get_api_key():
    return os.getenv("OPENAQ_API_KEY")


LOCATION_ID = 5633

RAW_DATA_PATH = "Data/raw/aqi_raw.csv"
PROCESSED_DATA_PATH = "Data/processed/aqi_processed.csv"
MODEL_PATH = "models/saved_model.pkl"
