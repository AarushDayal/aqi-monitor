# Air Quality Intelligence (AQI) Platform - Complete Project Overview

This document provides a comprehensive "in-and-out" breakdown of your entire AQI Forecasting project. It explains the purpose of every directory and file, how the machine learning architecture operates, and how the data flows from live APIs to the user interface.

## 🏗️ 1. High-Level Architecture

The project is an end-to-end Machine Learning web application designed to forecast air quality across multiple time horizons (8 hours, 24 hours, and 7 days). 

**The pipeline operates in three main stages:**
1. **Data Ingestion & Preprocessing:** Cleaning raw historical data, engineering advanced features (like trigonometric time encodings and time-lagged variables), and splitting it for training.
2. **Dual-Level Machine Learning:** 
   - **Level 1 (Stacking Ensemble):** Accurately deduces the current base AQI from raw weather/pollutant features to filter out sensor noise.
   - **Level 2 (Multi-Output XGBoost):** Takes the base state and predicts the exact AQI for the 8h, 24h, and 168h horizons simultaneously.
3. **Real-time Inference (Streamlit):** The user interface fetches live environmental data (via WAQI and Zippopotam APIs), engineers the features on the fly, passes them through the saved `.pkl` models, and displays the forecasts dynamically.

*(Note: The project was recently migrated from a Flask backend to a pure Streamlit architecture for better integration and UI/UX).*

---

## 📂 2. File & Directory Breakdown

Here is exactly what every file and folder in the project does:

### 🌟 Core Source Code (`src/`)
This is the heart of the application.

*   **`src/app.py`**: The main entry point for the Streamlit dashboard. It renders the UI, handles user PIN code inputs, loads the machine learning models (with an automatic fallback to download them from GitHub if Streamlit Cloud misses them), and visualizes the predictions using Plotly charts and premium CSS.
*   **`src/pages/About.py`**: A secondary Streamlit page that provides users with technical details about the project's architecture.
*   **`src/fetch_realtime_data.py`**: The bridge to the real world. It contains:
    *   `get_user_location()`: Determines the default location (recently updated to default to New Delhi).
    *   `geocode_pin_code()`: Connects to the Zippopotam API to convert Indian PIN codes into Latitude/Longitude.
    *   `fetch_waqi_data()`: Pulls live pollutant data from the World Air Quality Index (WAQI) API.
    *   `parse_waqi_to_features()`: Critically ensures the live API data perfectly matches the 23-dimensional feature matrix the models were trained on (handling missing features and mock time lags).
*   **`src/forecasting_model.py`**: A lightweight wrapper script that loads `multi_horizon_model.pkl` and processes the base features to return the future forecasts.
*   **`src/preprocess.py`**: The historical data factory. It takes raw CSV data and engineers it (removing outliers, calculating PM ratios, creating sine/cosine waves for time, and shifting targets to create `AQI_8h`, `AQI_24h`, etc.).
*   **`src/train_multi_horizon.py`**: The training script for the **Level 2** model. It trains an Extreme Gradient Boosting (XGBoost) model wrapped in a `MultiOutputRegressor` to predict the future horizons, and saves it as `multi_horizon_model.pkl`.
*   **`src/model3.py`**: The training script for the **Level 1** Base State Estimator. It trains a highly robust Stacking Ensemble (using Random Forest, Extra Trees, LightGBM, and a Ridge Regression meta-model) and saves it as `stacking_ensemble.pkl`.
*   **`src/model1.py` & `src/model2.py`**: Earlier/alternative iterations of predictive models used during the experimentation phase.
*   **`src/data_logger.py`**: A telemetry script designed to safely log real-time inference data (features and predicted AQI) for future model monitoring or retraining.
*   **`src/correlation_matrix_script.py` & `src/splitdatascript.py`**: Data science utility scripts used for Exploratory Data Analysis (EDA) to understand feature correlations and reliably split datasets into train/test groups.

### 🧠 Machine Learning Models (`models/`)
*   **`models/saved/stacking_ensemble.pkl`** (~33MB): The compiled Level 1 model.
*   **`models/saved/multi_horizon_model.pkl`** (~1.4MB): The compiled Level 2 XGBoost model.

### 🗄️ Data Storage (`Data/`)
*   **`Data/raw/`**: Where the initial, untouched dataset files are kept.
*   **`Data/processed/`**: Where `preprocess.py` dumps the cleaned, temporally-shifted data (e.g., `train_temporal.csv`, `test_temporal.csv`) ready for model training.
*   **`Data/live/`**: Used for temporary storage or logging of real-time pipeline batches.

### ⚙️ Root Configuration & Utilities
*   **`compress_models.py`**: A utility script used to compress the large `.pkl` files (using `joblib.dump(compress=3)`) to prevent GitHub from blocking the push due to their 100MB file size limits.
*   **`requirements.txt`**: Declares all Python dependencies (Streamlit, Plotly, XGBoost, LightGBM, Scikit-Learn, etc.) required for deployment.
*   **`project_documentation.md`**: The technical manifesto of the project. It deeply explains the mathematical reasoning behind the data preprocessing (e.g., cyclical trigonometric time encoding) and the architecture of the dual-level ML pipeline.
*   **`Dockerfile`**: Instructions for containerizing the application using Docker, ensuring it can run consistently on any cloud provider.
*   **`config.py`**: Centralized configuration variables (like base directories or API keys).
*   **`api/` & `dashboard/`**: Legacy folders that likely housed the old Flask backend and React/HTML frontend before the project was streamlined entirely into Streamlit.
*   **`notebooks/`**: Jupyter Notebooks used by data scientists for prototyping, testing hypotheses, and evaluating model metrics interactively.

---

## 🔄 3. How the Application Works "In and Out" (The Live Flow)

When a user visits the live Streamlit URL:

1. **Initialization:** `app.py` boots up and attempts to load the two `.pkl` models. If they are missing from the server (a common Streamlit Cloud glitch with large files), `app.py` directly hits the raw GitHub URL and downloads them dynamically into memory.
2. **User Input:** The user types a PIN code (e.g., `110001`) into the sidebar and clicks "Fetch Live Data". (If left blank, it defaults to New Delhi).
3. **Geocoding:** `fetch_realtime_data.py` sends the PIN code to Zippopotam (`api.zippopotam.us`) and gets the exact Latitude/Longitude.
4. **Data Fetching:** Those coordinates are sent to the WAQI API (`api.waqi.info`), which returns the real-time pollutants (PM2.5, PM10, NO2, etc.) for that exact location.
5. **Feature Alignment:** Because the ML models expect exactly 23 features (including advanced engineered ones like `PM_ratio`, `hour_sin`, and mock lags), `parse_waqi_to_features()` instantly processes the raw WAQI data into the exact mathematical format the models require.
6. **Inference (Level 1):** The 23 features are fed into `stacking_ensemble.pkl`, which returns the filtered, highly-accurate `Current AQI`.
7. **Inference (Level 2):** The base features are passed into `multi_horizon_model.pkl` (wrapped by `forecasting_model.py`), which shoots back three numbers: the predicted AQI for +8 Hours, +24 Hours, and +7 Days.
8. **Visualization:** `app.py` updates the premium CSS glassmorphic UI, displaying the numbers in cards and charting the predictive trend using an interactive Plotly area graph.
