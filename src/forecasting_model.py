import numpy as np

def forecast_aqi(model, base_features):
    X = np.array(list(base_features.values())).reshape(1, -1)
    
    # Model is a MultiOutputRegressor predicting log1p(AQI)
    preds_log = model.predict(X)[0]
    preds = np.expm1(preds_log)
    preds = np.clip(preds, 0, 500)
    
    # Returns [aqi_8h, aqi_24h, aqi_168h]
    return [round(float(p), 2) for p in preds]

