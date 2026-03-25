from flask import Flask, render_template, jsonify, request
import numpy as np
import joblib
import os
from fetch_realtime_data import fetch_and_prepare, get_aqi_category

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/saved/stacking_ensemble.pkl")
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


def predict_aqi(features_df):
    m = load_model()
    y_log = m.predict(features_df)
    y = np.expm1(y_log)
    y = np.clip(y, 0, 500)
    return round(float(y[0]), 1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict():
    """
    Auto-detects location from server IP.
    Optional query params: ?lat=28.6&lon=77.2 to override.
    """
    try:
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)

        features_df, raw_data, location = fetch_and_prepare(lat, lon)
        predicted_aqi = predict_aqi(features_df)
        category, color = get_aqi_category(predicted_aqi)

        pollutants = {
            "PM2.5": features_df["PM2.5"].iloc[0],
            "PM10":  features_df["PM10"].iloc[0],
            "NO2":   features_df["NO2"].iloc[0],
            "CO":    features_df["CO"].iloc[0],
            "SO2":   features_df["SO2"].iloc[0],
        }

        return jsonify({
            "status":         "ok",
            "predicted_aqi":  predicted_aqi,
            "reported_aqi":   location.get("aqi_reported"),
            "category":       category,
            "color":          color,
            "station":        location.get("station"),
            "city":           location.get("city"),
            "country":        location.get("country"),
            "lat":            location.get("lat"),
            "lon":            location.get("lon"),
            "pollutants":     {k: round(float(v), 2) for k, v in pollutants.items()},
            "timestamp":      raw_data.get("time", {}).get("s", "N/A"),
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
