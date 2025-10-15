import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained XGB pipeline
model = pickle.load(open("model.pkl", "rb"))

st.title("📱 Mobile Price Prediction App")
st.write("Enter mobile specifications to predict its used price range:")

# -------------------------------
# 🧾 Input Fields
# -------------------------------
brand = st.selectbox(
    "Select Brand",
    [
        "Others", "Apple", "Samsung", "Huawei", "Realme", "LG", "Lenovo", "ZTE", "Xiaomi", "Oppo",
        "Asus", "Alcatel", "Vivo", "Micromax", "Honor", "HTC", "Nokia",
        "Motorola", "Sony", "Meizu", "Gionee", "Acer",
    ]
)
os = st.selectbox("Operating System", ["Android", "iOS", "Others"])
screen_size = st.number_input("Screen Size (in inches)", min_value=3.0, max_value=100.0, value=6.0, step=0.1)
_4g = st.selectbox("Supports 4G?", ["Yes", "No"])
_5g = st.selectbox("Supports 5G?", ["Yes", "No"])
rear_camera_mp = st.number_input("Rear Camera (MP)", min_value=1, max_value=200, value=48)
front_camera_mp = st.number_input("Front Camera (MP)", min_value=1, max_value=64, value=16)
internal_memory = st.selectbox("Internal Memory (GB)", [4, 8, 16, 32, 64, 128, 256, 512, 1024])
ram = st.selectbox("RAM (GB)", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
battery = st.number_input("Battery (mAh)", min_value=500, max_value=10000, value=4000)
weight = st.number_input("Weight (grams)", min_value=50, max_value=500, value=180)
release_year = st.number_input("Release Year", min_value=2000, max_value=2025, value=2023)
days_used = st.number_input("Days Used", min_value=0, max_value=3650, value=0)

# 🆕 User provides actual new price (we normalize it internally)
actual_new_price = st.number_input(
    "Enter Actual New Price (💲)", 
    min_value=1, 
    max_value=10000, 
    value=50, 
    step=50
)

# Convert to normalized (log scale, as used during training)
normalized_new_price = np.log(actual_new_price)

# -------------------------------
# 🧮 Prepare DataFrame for Model
# -------------------------------
input_dict = {
    'device_brand': [brand],
    'os': [os],
    'screen_size': [screen_size],
    '4g': [_4g],
    '5g': [_5g],
    'rear_camera_mp': [rear_camera_mp],
    'front_camera_mp': [front_camera_mp],
    'internal_memory': [internal_memory],
    'ram': [ram],
    'battery': [battery],
    'weight': [weight],
    'release_year': [release_year],
    'days_used': [days_used],
    'normalized_new_price': [normalized_new_price]
}

input_df = pd.DataFrame(input_dict)

# -------------------------------
# 🔮 Predict Normalized Used Price
# -------------------------------
if st.button("Predict Used Price Range"):
    pred_norm = model.predict(input_df)  # prediction from your XGB model
    
    # 10% confidence range (in normalized form)
    lower_norm = pred_norm[0] * 0.9
    upper_norm = pred_norm[0] * 1.1

    # Convert normalized predictions back to actual prices (inverse log)
    lower_actual = np.exp(lower_norm)
    upper_actual = np.exp(upper_norm)

    # 🧩 Safety rule: used price cannot exceed new price
    if upper_actual > actual_new_price:
        upper_actual = actual_new_price
    if lower_actual > upper_actual:
        lower_actual = upper_actual * 0.7 # 30% lower bound fallback

    st.success(f"💰 Predicted Used Price Range: 💲{lower_actual:,.0f} - 💲{upper_actual:,.0f}")

