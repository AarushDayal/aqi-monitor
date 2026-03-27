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
    log_pred = model.predict(X)[0]
    return float(np.clip(np.expm1(log_pred), 0, 500))


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
        data = get_realtime_data()

        features = data["features"]
        pollutants = data["pollutants"]
        location = data["location"]

        base_aqi = float(np.expm1(predict_current_aqi(features)))
        base_aqi = float(np.clip(base_aqi, 0, 500))

        if LOGGING_ENABLED:
            try:
                log_data(features, base_aqi)
            except Exception as e:
                print("Logging failed:", e)

        future_preds = forecast_aqi(model=model, base_features=features, steps=3)

        aqi_8h = future_preds[0]
        aqi_24h = future_preds[1]
        aqi_7d = future_preds[2]

        category = get_aqi_category(base_aqi)

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
