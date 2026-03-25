import requests
import pandas as pd
import os
from config import get_api_key, RAW_DATA_PATH

LAT = 26.9164
LON = 75.7998


def fetch_data():

    url = f"https://api.openaq.org/v3/measurements/points?coordinates={LAT},{LON}&radius=5000&limit=100"
    headers = {"X-API-Key": get_api_key()}
    response = requests.get(url, headers=headers)
    data = response.json()
    print(data)  # keep temporarily
    records = []
    try:
        measurements = data["results"]

        for m in measurements:
            records.append(
                {
                    "location_id": m.get("locationId"),
                    "parameter": m["parameter"]["name"],
                    "value": m["value"],
                    "unit": m["parameter"]["units"],
                    "timestamp": m["dateTime"]["utc"],
                }
            )

        df = pd.DataFrame(records)

        file_exists = os.path.isfile(RAW_DATA_PATH)
        df.to_csv(RAW_DATA_PATH, mode="a", header=not file_exists, index=False)

        print("✅ Live AQI data fetched and stored.")

    except Exception as e:
        print("❌ Error parsing data:", e)


if __name__ == "__main__":
    fetch_data()
