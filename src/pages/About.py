import streamlit as st

st.set_page_config(
    page_title="About the Project",
    page_icon=None,
    layout="centered"
)

# Premium CSS Styling (Reused from app.py)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Air Quality Intelligence Architecture</h1>', unsafe_allow_html=True)

st.markdown("""
Welcome! This project demonstrates an end-to-end Machine Learning lifecycle applied to environmental forecasting.

### Machine Learning Engine
- **Models Used:** A Stacking Ensemble architecture combining Tree-based algorithms (XGBoost, LightGBM) and linear models to predict base Air Quality Index.
- **Multi-Horizon Forecasting:** A separate Multi-Output Regressor is used to forecast AQI across multiple time horizons (+8 Hours, +24 Hours, +7 Days).
- **Feature Engineering:** We engineered cyclical time features (Sine/Cosine for hour and month), rolling lags, and pollutant ratio features (e.g. PM2.5 / PM10 ratio) to capture temporal dependencies and complex non-linear relationships.

### Data Pipeline & Real-Time Inference
- **Live Ingestion:** Real-time environmental metrics are queried through the WAQI API based on precise auto-detected IP geolocation or user-provided Postal Codes.
- **Dynamic Preprocessing:** The application matches the incoming data to the training schema on the fly, filling missing values and recalculating engineered features synchronously.
- **Interactive UI:** The frontend is powered by Streamlit, utilizing custom HTML/CSS injections for a premium "glassmorphic" UI, dynamic state management, and real-time Plotly data visualizations.

### Tech Stack
- **Python** (Pandas, Numpy, Scikit-Learn, XGBoost)
- **Streamlit** for the frontend Application
- **Plotly** for interactive charts
- **Docker** ready for cloud deployment

---

*This project was built to showcase rigorous data engineering, predictive modeling, and polished product deployment.*
""")

if st.button("Back to Main Dashboard"):
    st.switch_page("app.py")
