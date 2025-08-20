import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load(r"models/random_forest_asteroid_model.joblib")

st.title("🚀 NASA NEO Hazard Prediction")

st.markdown("Enter asteroid details to predict if it's **Hazardous** or **Safe**.")

# Collect inputs
abs_mag = st.number_input("Absolute Magnitude", value=22.0, step=0.1)
diameter_min = st.number_input("Estimated Diameter Min (km)", value=0.1, step=0.01)
diameter_max = st.number_input("Estimated Diameter Max (km)", value=0.2, step=0.01)

orbiting_body = st.selectbox("Orbiting Body", ["Earth", "Mars", "Venus", "Jupiter", "Other"])
# Encode orbiting_body (match training encoding order)
orbiting_map = {"Earth": 0, "Jupiter": 1, "Mars": 2, "Venus": 3, "Other": 4}
orbiting_encoded = orbiting_map.get(orbiting_body, 4)

velocity = st.number_input("Relative Velocity (km/s)", value=25.0, step=0.5)
miss_distance = st.number_input("Miss Distance (km)", value=500000.0, step=1000.0)
diameter_mean = st.number_input("Mean Diameter (km)", value=0.15, step=0.01)
vel_diam_ratio = st.number_input("Velocity / Diameter Ratio", value=100.0, step=1.0)
log_distance = st.number_input("Log Miss Distance", value=5.7, step=0.1)
threat_score = st.number_input("Threat Score", value=50.0, step=1.0)

# Put features in EXACT order used during training
features = np.array([[abs_mag, diameter_min, diameter_max,
                      orbiting_encoded, velocity, miss_distance,
                      diameter_mean, vel_diam_ratio,
                      log_distance, threat_score]])

# Predict
if st.button("Predict 🚀"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]  # Probability asteroid is hazardous

    impact_percent = round(proba * 100, 2)

    st.markdown(f"### 🌍 Chance of Earth Impact: **{impact_percent}%**")

    if prediction == 1:
        st.error("☢️ Hazardous Asteroid!")
    else:
        st.success("✅ Safe Asteroid")

