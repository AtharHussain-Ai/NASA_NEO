import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load(r"models/random_forest_asteroid_model.joblib")

# --- Page Config ---
st.set_page_config(page_title="NASA NEO Prediction", layout="centered")

# --- Header ---
st.title("NASA NEO Hazard Prediction")
st.markdown(
    """
    Enter asteroid details to predict if it's Hazardous or Safe.  
    The model estimates the chance of Earth impact in real time.
    """
)

st.divider()

# --- Input Section ---
st.subheader("Asteroid Parameters")

col1, col2 = st.columns(2)

with col1:
    abs_mag = st.number_input("Absolute Magnitude", value=22.0, step=0.1)
    diameter_min = st.number_input("Estimated Diameter Min (km)", value=0.1, step=0.01)
    diameter_max = st.number_input("Estimated Diameter Max (km)", value=0.2, step=0.01)

with col2:
    orbiting_body = st.selectbox("Orbiting Body", ["Earth", "Mars", "Venus", "Jupiter", "Other"])
    orbiting_map = {"Earth": 0, "Jupiter": 1, "Mars": 2, "Venus": 3, "Other": 4}
    orbiting_encoded = orbiting_map.get(orbiting_body, 4)

    velocity = st.number_input("Relative Velocity (km/s)", value=25.0, step=0.5)
    miss_distance = st.number_input("Miss Distance (km)", value=500000.0, step=1000.0)

st.divider()

# --- Feature vector (match training order) ---
features = np.array([[abs_mag, diameter_min, diameter_max,
                      orbiting_encoded, velocity, miss_distance]])

# --- Prediction Button ---
if st.button("Predict Hazard"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]  # Probability hazardous

    impact_percent = round(proba * 100, 2)

    # --- Output ---
    st.subheader("Prediction Results")
    st.metric("Chance of Earth Impact", f"{impact_percent}%")
    st.progress(int(impact_percent))

    if prediction == 1:
        st.error("Hazardous Asteroid")
    else:
        st.success("Safe Asteroid")
