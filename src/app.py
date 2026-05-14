import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import os
from datetime import datetime, timezone, timedelta 
from fetch_realtime_data import get_realtime_data
from forecasting_model import forecast_aqi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Optional safe logging
try:
    from data_logger import log_data
    LOGGING_ENABLED = True
except ImportError:
    LOGGING_ENABLED = False

# Configure the Streamlit page
st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #a1a1aa;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .metric-card {
        background: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin-top: 10px;
        letter-spacing: -1px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* AQI Colors */
    .good { color: #34d399; text-shadow: 0 0 20px rgba(52, 211, 153, 0.3); }
    .moderate { color: #fbbf24; text-shadow: 0 0 20px rgba(251, 191, 36, 0.3); }
    .unhealthy { color: #f87171; text-shadow: 0 0 20px rgba(248, 113, 113, 0.3); }
    .very-unhealthy { color: #c084fc; text-shadow: 0 0 20px rgba(192, 132, 252, 0.3); }
    .hazardous { color: #fda4af; text-shadow: 0 0 20px rgba(253, 164, 175, 0.3); }
    
    /* Status Banner */
    .status-banner {
        padding: 20px; 
        border-radius: 16px; 
        background: rgba(20, 20, 20, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 30px; 
        text-align: center;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }
    
    .pollutant-card {
        background: #18181b;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #27272a;
    }
    .pollutant-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    .pollutant-name {
        color: #a1a1aa;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    import urllib.request
    
    models_dir = os.path.join(BASE_DIR, "models", "saved")
    os.makedirs(models_dir, exist_ok=True)
    
    stacking_path = os.path.join(models_dir, "stacking_ensemble.pkl")
    multi_path = os.path.join(models_dir, "multi_horizon_model.pkl")
    
    # Download models if they don't exist
    base_url = "https://raw.githubusercontent.com/AarushDayal/aqi-monitor/main/models/saved/"
    
    if not os.path.exists(stacking_path):
        try:
            urllib.request.urlretrieve(base_url + "stacking_ensemble.pkl", stacking_path)
        except Exception as e:
            st.error(f"Failed to download stacking_ensemble.pkl: {e}")
            
    if not os.path.exists(multi_path):
        try:
            urllib.request.urlretrieve(base_url + "multi_horizon_model.pkl", multi_path)
        except Exception as e:
            st.warning(f"Failed to download multi_horizon_model.pkl: {e}")
    
    try:
        model = joblib.load(stacking_path)
    except Exception as e:
        st.error(f"Error loading stacking ensemble model: {e}")
        model = None
        
    try:
        multi_model = joblib.load(multi_path)
    except Exception as e:
        st.warning(f"Warning: could not load multi_horizon_model.pkl: {e}")
        multi_model = None
        
    return model, multi_model

model, multi_model = load_models()

def predict_current_aqi(features_dict, base_model):
    if base_model is None:
        return 0.0
    X = np.array(list(features_dict.values())).reshape(1, -1)
    log_pred = base_model.predict(X)[0]
    return float(np.clip(np.expm1(log_pred), 0, 500))

def get_aqi_category_and_color(aqi):
    if aqi <= 50:
        return "Good", "good"
    elif aqi <= 100:
        return "Moderate", "moderate"
    elif aqi <= 200:
        return "Unhealthy", "unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy", "very-unhealthy"
    else:
        return "Hazardous", "hazardous"

def render_metric_card(label, value, color_class=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_pollutant(name, value):
    st.markdown(f"""
    <div class="pollutant-card">
        <div class="pollutant-name">{name}</div>
        <div class="pollutant-val">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # --- Data Fetching Logic (Done early to populate session state) ---
    refresh = False
    pin_input = ""
    with st.sidebar:
        st.markdown("<h1>Control Panel</h1>", unsafe_allow_html=True)
        st.markdown("Refresh real-time environmental APIs and trigger ML inference.")
        
        pin_input = st.text_input("Custom PIN Code (Optional)", placeholder="e.g. 110001", help="Leave blank for default location (New Delhi, India)")
        
        refresh = st.button("Fetch Live Data", type="primary", use_container_width=True)
        st.divider()
        
    if refresh or "data" not in st.session_state:
        with st.spinner("Connecting to environmental APIs..."):
            try:
                # If pin_input is empty, we pass None
                data = get_realtime_data(pin_code=pin_input.strip() if pin_input.strip() else None)
                st.session_state["data"] = data
                st.session_state["last_updated"] = time.strftime("%I:%M %p, %b %d %Y")
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                return

    data = st.session_state["data"]
    features = data["features"]
    pollutants = data["pollutants"]
    
    # --- Sidebar Location Info ---
    with st.sidebar:
        loc = data["location"]
        st.success(f"**Location:**\n{loc}")
        st.caption(f"Last Sync: {datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%d %b %I:%M %p')} IST")
        st.divider()
        
        st.markdown("### About the Project")
        st.info("Learn more about the architecture and pipeline.")
        if st.button("Project Details", use_container_width=True):
            st.switch_page("pages/About.py")

    # --- Main Header ---
    st.markdown('<h1 class="main-header">Air Quality Forecasting</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced multi-horizon predictive modeling and real-time inference</p>', unsafe_allow_html=True)

    # --- ML Inference ---
    base_aqi = predict_current_aqi(features, model)
    base_aqi = float(np.clip(base_aqi, 0, 500))

    if LOGGING_ENABLED:
        try:
            log_data(features, base_aqi)
        except Exception as e:
            print("Logging failed:", e)

    if multi_model is not None:
        future_preds = forecast_aqi(model=multi_model, base_features=features)
    else:
        future_preds = [base_aqi] * 3

    aqi_8h, aqi_24h, aqi_7d = future_preds
    category, color_class = get_aqi_category_and_color(base_aqi)

    # --- Status Banner ---
    st.markdown(f"""
        <div class="status-banner">
            <h2 style="margin: 0; font-weight: 300;">Current Environment Status: 
                <span class="{color_class}" style="font-weight: 800;">{category}</span>
            </h2>
        </div>
    """, unsafe_allow_html=True)

    # --- Forecast Cards ---
    fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)
    with fc_col1:
        render_metric_card("Live Index", round(base_aqi, 1), color_class)
    with fc_col2:
        render_metric_card("8H Forecast", round(aqi_8h, 1),)
    with fc_col3:
        render_metric_card("24H Forecast", round(aqi_24h, 1),)
    with fc_col4:
        render_metric_card("7D Forecast", round(aqi_7d, 1),)

    st.divider()

    # --- Lower Section: Pollutants & Chart ---
    bot_col1, bot_col2 = st.columns([1, 2])
    
    with bot_col1:
        st.markdown("### Pollutant Levels")
        st.caption("Micrograms per cubic meter (μg/m³)")
        
        p1, p2 = st.columns(2)
        with p1: render_pollutant("PM2.5", pollutants.get("pm2_5", "--"))
        with p2: render_pollutant("PM10", pollutants.get("pm10", "--"))
        
        p3, p4 = st.columns(2)
        with p3: render_pollutant("NO2", pollutants.get("no2", "--"))
        with p4: render_pollutant("SO2", pollutants.get("so2", "--"))
        
        render_pollutant("Carbon Monoxide (CO)", pollutants.get("co", "--"))

    with bot_col2:
        st.markdown("### Predictive Trend Analysis")
        
        fig = go.Figure()
        
        # Area chart for a modern look
        x_vals = ['Current', '+8 Hours', '+24 Hours', '+7 Days']
        y_vals = [base_aqi, aqi_8h, aqi_24h, aqi_7d]
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, 
            mode='lines+markers', 
            fill='tozeroy',
            line=dict(color='#8b5cf6', width=4, shape='spline'),
            marker=dict(size=12, color='#ffffff', line=dict(width=3, color='#8b5cf6')),
            fillcolor='rgba(139, 92, 246, 0.1)'
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
            font=dict(color='#a1a1aa', family='Inter'),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            hovermode='x unified'
        )
        
        # Subtle Threshold Lines
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(52, 211, 153, 0.5)", annotation_text="Good", annotation_font_color="rgba(52, 211, 153, 0.8)")
        fig.add_hline(y=100, line_dash="dot", line_color="rgba(251, 191, 36, 0.5)", annotation_text="Moderate", annotation_font_color="rgba(251, 191, 36, 0.8)")
        fig.add_hline(y=200, line_dash="dot", line_color="rgba(248, 113, 113, 0.5)", annotation_text="Unhealthy", annotation_font_color="rgba(248, 113, 113, 0.8)")
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

if __name__ == "__main__":
    main()
