import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# Load trained model
model = joblib.load(r"models/random_forest_asteroid_model.joblib")

# --- Page Config ---
st.set_page_config(page_title="NASA NEO Hazard Prediction", layout="centered")

# --- Custom CSS for clean minimal look ---
st.markdown(
    """
    <style>
        #MainMenu, header, footer {visibility: hidden;}
        .block-container {padding-top: 2rem; max-width: 850px; margin: auto;}
        body {background-color: white; color: #111;}
        h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; font-weight: 600;}
        .stMetric {text-align: center;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.title("NASA NEO Hazard Prediction")
st.caption("Estimate the probability of an asteroid impacting Earth.")

# --- Input Section ---
st.subheader("Asteroid Parameters")

col1, col2 = st.columns(2)

with col1:
    abs_mag = st.number_input("Absolute Magnitude", value=22.0, step=0.1)
    diameter_min = st.number_input("Estimated Diameter Min (km)", value=0.1, step=0.01)
    diameter_max = st.number_input("Estimated Diameter Max (km)", value=0.2, step=0.01)
    orbiting_body = st.selectbox("Orbiting Body", ["Earth", "Mars", "Venus", "Jupiter", "Other"])
    orbiting_map = {"Earth": 0, "Jupiter": 1, "Mars": 2, "Venus": 3, "Other": 4}
    orbiting_encoded = orbiting_map.get(orbiting_body, 4)

with col2:
    velocity = st.number_input("Relative Velocity (km/s)", value=25.0, step=0.5)
    miss_distance = st.number_input("Miss Distance (km)", value=500000.0, step=1000.0)
    diameter_mean = st.number_input("Mean Diameter (km)", value=0.15, step=0.01)
    vel_diam_ratio = st.number_input("Velocity / Diameter Ratio", value=100.0, step=1.0)
    log_distance = st.number_input("Log Miss Distance", value=5.7, step=0.1)
    threat_score = st.number_input("Threat Score", value=50.0, step=1.0)

# --- Feature vector (EXACT training order) ---
features = np.array([[abs_mag, diameter_min, diameter_max,
                      orbiting_encoded, velocity, miss_distance,
                      diameter_mean, vel_diam_ratio,
                      log_distance, threat_score]])

# --- Prediction Button ---
if st.button("Predict"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]
    impact_percent = round(proba * 100, 2)

    st.subheader("Prediction Results")
    st.metric("Chance of Earth Impact", f"{impact_percent}%")

    # --- Radial Gauge ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=impact_percent,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if impact_percent > 50 else "green"},
            'steps': [
                {'range': [0, 25], 'color': "#d4f7d4"},
                {'range': [25, 50], 'color': "#f9f5b1"},
                {'range': [50, 75], 'color': "#fdd9a5"},
                {'range': [75, 100], 'color': "#f5b7b1"},
            ],
        }
    ))

    fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- Risk Message ---
    if prediction == 1:
        st.error("Hazardous Asteroid Detected")
    else:
        st.success("Safe Asteroid")
