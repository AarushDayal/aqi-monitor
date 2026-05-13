import joblib
import os

model_path = 'models/saved/stacking_ensemble.pkl'
if os.path.exists(model_path):
    print(f"Loading {model_path}...")
    model = joblib.load(model_path)
    print("Compressing...")
    joblib.dump(model, model_path, compress=3)
    new_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Compressed size: {new_size:.2f} MB")
else:
    print("Model not found")
