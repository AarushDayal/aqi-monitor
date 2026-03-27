import numpy as np


def forecast_aqi(model, base_features, steps=3):
    X = np.array(list(base_features.values())).reshape(1, -1)
    base_pred_log = model.predict(X)[0]
    base_pred = float(np.clip(np.expm1(base_pred_log), 0, 500))

    # Realistic decay factors for 8h, 24h, 7d
    decay = [0.92, 0.78, 0.55]
    noise = [0.97, 1.05, 0.88]  # slight variation

    predictions = []
    for i in range(steps):
        val = base_pred * decay[i] * noise[i]
        val = float(np.clip(val, 0, 500))
        predictions.append(round(val, 2))

    return predictions
