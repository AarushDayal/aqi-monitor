# src/app.py

from flask import Flask, render_template, jsonify
import joblib
import numpy as np

from fetch_realtime_data import get_realtime_data
from forecasting_model import forecast_aqi


# OPTIONAL (safe logging)
try:
    from data_logger import log_data

    LOGGING_ENABLED = True
except:
    LOGGING_ENABLED = False

app = Flask(__name__)

# 🔹 Load ML model
model = joblib.load("../models/saved/stacking_ensemble.pkl")


# 🔹 Predict current AQI
def predict_current_aqi(features_dict):
    X = np.array(list(features_dict.values())).reshape(1, -1)
    return float(model.predict(X)[0])


# 🔹 AQI Category
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# 🔹 Home Route
@app.route("/")
def home():
    return render_template("index.html")


# 🔹 Prediction API
@app.route("/predict")
def predict():
    try:
        # ✅ Step 1: Get real-time data
        data = get_realtime_data()

        features = data["features"]
        pollutants = data["pollutants"]
        location = data["location"]

        # ✅ Step 2: Predict current AQI
        base_aqi = predict_current_aqi(features)

        # ✅ Step 3: Optional logging (safe)
        if LOGGING_ENABLED:
            try:
                log_data(features, base_aqi)
            except Exception as e:
                print("Logging failed:", e)

        # ✅ Step 4: Forecast (SAFE VERSION - no lag mismatch)
        future_preds = forecast_aqi(model=model, base_features=features, steps=3)

        aqi_8h = future_preds[0]
        aqi_24h = future_preds[1]
        aqi_7d = future_preds[2]

        # ✅ Step 5: Category
        category = get_aqi_category(base_aqi)

        # ✅ Step 6: Response
        return jsonify(
            {
                "aqi_now": round(base_aqi, 2),
                "aqi_8h": round(aqi_8h, 2),
                "aqi_24h": round(aqi_24h, 2),
                "aqi_7d": round(aqi_7d, 2),
                "category": category,
                "pollutants": pollutants,
                "location": location,
            }
        )

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})


# 🔹 Run App
if __name__ == "__main__":
    app.run(debug=True)
