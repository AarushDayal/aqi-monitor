import numpy as np
from datetime import datetime, timedelta


def forecast_aqi(model, base_features, steps=3):
    X = np.array(list(base_features.values())).reshape(1, -1)
    base_pred_log = model.predict(X)[0]
    base_pred = float(np.clip(np.expm1(base_pred_log), 0, 500))

    now = datetime.now()
    month = now.month

    # Season detection
    is_winter = month in [11, 12, 1, 2]
    is_summer = month in [4, 5, 6]

    def hour_factor(hour, weekday):
        factor = 1.0

        if is_winter:
            # Winter: night inversion is real — AQI goes UP at night
            if 0 <= hour <= 5:
                factor = 1.18  # temp inversion, worst air
            elif 6 <= hour <= 9:
                factor = 1.12  # morning rush + cold
            elif 10 <= hour <= 15:
                factor = 0.92  # sun disperses pollutants
            elif 16 <= hour <= 19:
                factor = 1.08  # evening rush
            elif 20 <= hour <= 23:
                factor = 1.14  # night inversion builds up
        elif is_summer:
            # Summer: hot afternoons disperse pollutants
            if 0 <= hour <= 5:
                factor = 0.90
            elif 6 <= hour <= 9:
                factor = 1.08  # morning rush
            elif 10 <= hour <= 16:
                factor = 0.82  # heat disperses
            elif 17 <= hour <= 20:
                factor = 1.05  # evening rush
            elif 21 <= hour <= 23:
                factor = 0.92
        else:
            # Monsoon / autumn — moderate
            if 7 <= hour <= 10:
                factor = 1.08
            elif 17 <= hour <= 20:
                factor = 1.05
            elif 0 <= hour <= 5:
                factor = 0.88
            else:
                factor = 0.95

        # Weekend — less traffic
        if weekday >= 5:
            factor *= 0.93

        return factor

    offsets = [8, 24, 168]  # 8h, 24h, 7d
    predictions = []

    for hours_ahead in offsets:
        future = now + timedelta(hours=hours_ahead)
        factor = hour_factor(future.hour, future.weekday())
        val = float(np.clip(base_pred * factor, 0, 500))
        predictions.append(round(val, 2))

    return predictions
