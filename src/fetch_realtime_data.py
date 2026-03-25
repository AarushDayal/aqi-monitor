import requests
import pandas as pd
import numpy as np
from datetime import datetime

WAQI_TOKEN = "2c2bc76956b55ae1bc800b88b6db0f131a031dca"


def get_user_location():
    """Auto-detect user's city using IP geolocation."""
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=5)
        data = resp.json()
        return {
            "city": data.get("city", "Delhi"),
            "lat": data.get("latitude", 28.6139),
            "lon": data.get("longitude", 77.2090),
            "country": data.get("country_name", "India"),
        }
    except Exception:
        return {"city": "Delhi", "lat": 28.6139, "lon": 77.2090, "country": "India"}


def fetch_waqi_data(lat, lon):
    """Fetch real-time AQI data from WAQI API using lat/lon."""
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "ok":
        raise ValueError(f"WAQI API error: {data.get('data', 'unknown error')}")

    return data["data"]


def parse_waqi_to_features(waqi_data):
    """
    Convert raw WAQI response into a feature DataFrame
    that matches the training data schema exactly.
    """
    iaqi = waqi_data.get("iaqi", {})

    def get_val(key, default=0.0):
        return iaqi.get(key, {}).get("v", default)

    now = datetime.now()
    hour = now.hour
    month = now.month

    season_map = {
        12: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
        10: 3,
        11: 3,
    }

    row = {
        "PM2.5": get_val("pm25"),
        "PM10": get_val("pm10"),
        "NO2": get_val("no2"),
        "CO": get_val("co"),
        "NOx": get_val("no2"),  # proxy — WAQI doesn't expose NOx directly
        "SO2": get_val("so2"),
        "hour": hour,
        "month": month,
        "day": now.weekday(),
        "quarter": (month - 1) // 3 + 1,
        "season": season_map[month],
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "PM_ratio": get_val("pm25") / (get_val("pm10") + 1e-5),
        "pollution_load": get_val("pm25")
        + get_val("pm10")
        + get_val("no2")
        + get_val("co"),
        "NO_proxy": get_val("no2"),  # NOx - NO2 proxy
    }

    return pd.DataFrame([row])


def get_aqi_category(aqi):
    """Return AQI category label and color."""
    if aqi <= 50:
        return "Good", "#00e400"
    if aqi <= 100:
        return "Satisfactory", "#ffff00"
    if aqi <= 200:
        return "Moderate", "#ff7e00"
    if aqi <= 300:
        return "Poor", "#ff0000"
    if aqi <= 400:
        return "Very Poor", "#8f3f97"
    return "Severe", "#7e0023"


def fetch_and_prepare(lat=None, lon=None):
    """
    Full pipeline: detect location → fetch WAQI → parse features.
    Returns (features_df, raw_waqi_data, location_info)
    """
    if lat is None or lon is None:
        location = get_user_location()
        lat, lon = location["lat"], location["lon"]
    else:
        location = {"lat": lat, "lon": lon, "city": "Custom", "country": ""}

    raw = fetch_waqi_data(lat, lon)

    # Update location with station name if available
    location["station"] = raw.get("city", {}).get(
        "name", location.get("city", "Unknown")
    )
    location["aqi_reported"] = raw.get("aqi", "N/A")

    features_df = parse_waqi_to_features(raw)

    # ===== FORCE MODEL-COMPATIBLE FEATURES =====

    required_features = [
        "PM2.5",
        "PM10",
        "NO2",
        "NOx",
        "NH3",
        "CO",
        "SO2",
        "Toluene",
        "hour",
        "month",
        "day",
        "quarter",
        "season",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "PM_ratio",
        "pollution_load",
        "NO_proxy",
    ]

    # Add missing features
    for col in required_features:
        if col not in features_df.columns:
            features_df[col] = 0.0

    # ⚠️ THIS LINE IS THE MOST IMPORTANT
    features_df = features_df.reindex(columns=required_features)

    # Debug (VERY IMPORTANT — run once)
    print("\nEXPECTED:", required_features)
    print("ACTUAL:  ", features_df.columns.tolist())

    return features_df, raw, location
