# Import necessary libraries
import streamlit as st  # For creating the web app
import numpy as np      # For working with arrays
import pickle           # For loading the saved model and scaler

# -----------------------------------------------------
# 1. Load the trained model and scaler (.pkl files)
# -----------------------------------------------------
# Make sure model.pkl and scaler.pkl are in the same folder as app.py
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------------------------------
# 2. Set the title of the Streamlit web app
# -----------------------------------------------------
st.title("🔬 Breast Cancer Prediction App")

st.write("""
This app predicts whether a breast tumor is **Benign (Harmless)** or **Malignant (Dangerous)**  
based on just 5 key diagnostic features.
""")

# -----------------------------------------------------
# 3. Create input fields for user to enter values
# -----------------------------------------------------
# The user will enter 5 feature values that were used during training
mean_radius = st.number_input("Mean Radius", min_value=0.0, format="%.4f")
mean_texture = st.number_input("Mean Texture", min_value=0.0, format="%.4f")
mean_area = st.number_input("Mean Area", min_value=0.0, format="%.4f")
mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, format="%.4f")
mean_compactness = st.number_input("Mean Compactness", min_value=0.0, format="%.4f")

# -----------------------------------------------------
# 4. Run prediction when user clicks "Predict"
# -----------------------------------------------------
if st.button("🔍 Predict"):
    # Step 1: Combine inputs into an array
    input_data = np.array([[mean_radius, mean_texture, mean_area, mean_smoothness, mean_compactness]])

    # Step 2: Scale input using the same scaler used during training
    scaled_input = scaler.transform(input_data)

    # Step 3: Predict using the loaded model
    prediction = model.predict(scaled_input)[0]

    # Step 4: Display result
    if prediction == 0:
        st.error("🔴 The tumor is **Malignant (Cancerous)**")
    else:
        st.success("🟢 The tumor is **Benign (Non-Cancerous)**")
