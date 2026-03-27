# src/forecast_model.py

import numpy as np


def forecast_aqi(model, base_features, steps=3):
    """
    Stable forecasting without lag/history
    """

    X = np.array(list(base_features.values())).reshape(1, -1)
    base_pred = model.predict(X)[0]

    predictions = []
    current = base_pred

    for _ in range(steps):
        current = 0.8 * current + 0.2 * base_pred
        predictions.append(float(current))

    return predictions
