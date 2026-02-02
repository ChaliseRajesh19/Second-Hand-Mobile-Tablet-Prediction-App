import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
st.markdown("""
    <style>
    /* Background Image */
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(135deg, rgba(102,126,234,0.03) 0%, rgba(118,75,162,0.03) 100%),
                          url('data:image/svg+xml,<svg xmlns="https://www.freepik.com/free-vector/gradient-ui-ux-background_17349978.htm" viewBox="0 0 60 60"><defs><pattern id="dot-pattern" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="2" fill="%23667eea" opacity="0.1"/></pattern><pattern id="grid-pattern" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M 20 0 L 0 0 0 20" fill="none" stroke="%23667eea" stroke-width="0.5" opacity="0.08"/></pattern></defs><rect width="60" height="60" fill="white"/><rect width="60" height="60" fill="url(%23grid-pattern)"/><rect width="60" height="60" fill="url(%23dot-pattern)"/></svg>');
        background-attachment: fixed;
        background-size: cover;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
            text-align: center;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(248,249,250,0.98) 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0;
        border: 3px solid #667eea;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
    }
    
    .info-box {
        background: rgba(240, 242, 246, 0.95);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #333;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load trained XGB pipeline
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ App Settings")
    
    theme_mode = st.radio(
        "Display Theme",
        ["Light", "Dark"],
        help="Choose your preferred theme"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **Mobile Price Predictor** uses advanced machine learning to estimate 
    the resale value of used mobile devices.
    
    The model analyzes:
    - Device specifications
    - Brand and release year
    - Usage duration
    - Market demand
    
    Predictions are for reference only.
    """)
    
    st.markdown("---")
    st.markdown("### 📞 Support & Follow Us")
    st.markdown("""
    **Connect with us on social media:**
    """)
    
    # Social Media Links
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <a href="https://www.facebook.com" target="_blank">
            <button style="width: 100%; padding: 10px; background-color: #1877F2; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 20px;">
                f
            </button>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href="https://www.instagram.com" target="_blank">
            <button style="width: 100%; padding: 10px; background-color: #E4405F; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 20px;">
                📷
            </button>
        </a>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <a href="https://twitter.com" target="_blank">
            <button style="width: 100%; padding: 10px; background-color: #1DA1F2; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 20px;">
                𝕏
            </button>
        </a>
        """, unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>📱 Mobile Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Get accurate resale price estimates for mobile devices</p>", unsafe_allow_html=True)

# Create two-column layout for inputs
col1, col2 = st.columns(2, gap="large")

# LEFT COLUMN - Device Information
with col1:
    st.markdown("<h3 class='section-header'>📋 Device Information</h3>", unsafe_allow_html=True)
    
    brand = st.selectbox(
        "Select Brand",
        [
            "Others", "Apple", "Samsung", "Huawei", "Realme", "LG", "Lenovo", "ZTE", "Xiaomi", "Oppo",
            "Asus", "Alcatel", "Vivo", "Micromax", "Honor", "HTC", "Nokia",
            "Motorola", "Sony", "Meizu", "Gionee", "Acer",
        ],
        help="Select the manufacturer of the mobile device"
    )
    
    os = st.selectbox("Operating System", ["Android", "iOS", "Others"], help="Choose the mobile OS")
    
    release_year = st.slider(
        "Release Year", 
        min_value=2000, 
        max_value=2025, 
        value=2023,
        help="Year the device was released"
    )
    
    days_used = st.number_input(
        "Days Used", 
        min_value=0, 
        max_value=3650, 
        value=0,
        step=30,
        help="How many days the device has been used (0 for new)"
    )

# RIGHT COLUMN - Technical Specifications
with col2:
    st.markdown("<h3 class='section-header'>⚙️ Technical Specifications</h3>", unsafe_allow_html=True)
    
    screen_size = st.number_input(
        "Screen Size (inches)", 
        min_value=3.0, 
        max_value=10.0, 
        value=6.5, 
        step=0.1,
        help="Display diagonal in inches"
    )
    
    ram = st.selectbox(
        "RAM (GB)", 
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        index=4,
        help="Random Access Memory"
    )
    
    internal_memory = st.selectbox(
        "Internal Storage (GB)", 
        [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        index=4,
        help="Built-in storage capacity"
    )
    
    battery = st.number_input(
        "Battery Capacity (mAh)", 
        min_value=500, 
        max_value=10000, 
        value=4000,
        step=100,
        help="Battery capacity in milliamp-hours"
    )

# CAMERA & CONNECTIVITY
st.markdown("<h3 class='section-header'>📷 Camera & Connectivity</h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    rear_camera_mp = st.number_input(
        "Rear Camera (MP)", 
        min_value=1, 
        max_value=200, 
        value=48,
        step=1,
        help="Main camera megapixels"
    )

with col2:
    front_camera_mp = st.number_input(
        "Front Camera (MP)", 
        min_value=1, 
        max_value=64, 
        value=16,
        step=1,
        help="Selfie camera megapixels"
    )

with col3:
    _4g = st.selectbox("4G Support", ["Yes", "No"], help="4G/LTE connectivity")

with col4:
    _5g = st.selectbox("5G Support", ["Yes", "No"], help="5G connectivity")

# WEIGHT & PRICE
st.markdown("<h3 class='section-header'>⚖️ Physical & Price Info</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    weight = st.number_input(
        "Weight (grams)", 
        min_value=50, 
        max_value=500, 
        value=180,
        step=5,
        help="Device weight in grams"
    )

with col2:
    actual_new_price = st.number_input(
        "New Phone Price (💲)", 
        min_value=1, 
        max_value=10000, 
        value=500, 
        step=50,
        help="Original retail price of the device"
    )


# Convert to normalized (log scale, as used during training)
normalized_new_price = np.log(actual_new_price)

# Prepare DataFrame for Model
input_dict = {
    'device_brand': [brand],
    'os': [os],
    '4g': [_4g],
    '5g': [_5g],
    'release_year': [release_year],      # ✅ ADD
    'screen_size': [screen_size],        # ✅ ADD
    'rear_camera_mp': [rear_camera_mp],
    'front_camera_mp': [front_camera_mp],
    'internal_memory': [internal_memory],
    'ram': [ram],
    'battery': [battery],
    'weight': [weight],
    'days_used': [days_used],
    'normalized_new_price': [normalized_new_price]
}

input_df = pd.DataFrame(input_dict)

# Prediction Section
st.markdown("<h3 class='section-header'>🔮 Price Estimation</h3>", unsafe_allow_html=True)

# Create button with custom styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "💰 Estimate Resale Price", 
        use_container_width=True,
        key="predict_btn"
    )

# Show prediction results
if predict_button:
    with st.spinner('Analyzing device specifications...'):
        pred_norm = model.predict(input_df)
        
        # 10% confidence range (in normalized form)
        lower_norm = pred_norm[0] * 0.9
        upper_norm = pred_norm[0] * 1.1

        # Convert normalized predictions back to actual prices (inverse log)
        lower_actual = np.exp(lower_norm)
        upper_actual = np.exp(upper_norm)

        # Safety rule: used price cannot exceed new price
        if upper_actual > actual_new_price:
            upper_actual = actual_new_price
        if lower_actual > upper_actual:
            lower_actual = upper_actual * 0.7

        # Display results in professional format
        st.markdown("<div class='prediction-box'>💲 {:.0f} - 💲 {:.0f}</div>".format(lower_actual, upper_actual), unsafe_allow_html=True)
        
        # Create result cards
        result_col1, result_col2, result_col3 = st.columns(3, gap="medium")
        
        with result_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='text-align: center;'>
                    <p style='color: #667eea; font-size: 0.9rem; margin: 0;'>MINIMUM PRICE</p>
                    <p style='font-size: 1.8rem; font-weight: 700; color: #333; margin: 0.5rem 0;'>💲{lower_actual:,.0f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            depreciation = ((actual_new_price - (lower_actual + upper_actual) / 2) / actual_new_price) * 100
            st.markdown(f"""
            <div class='metric-card'>
                <div style='text-align: center;'>
                    <p style='color: #667eea; font-size: 0.9rem; margin: 0;'>AVERAGE DEPRECIATION</p>
                    <p style='font-size: 1.8rem; font-weight: 700; color: #e74c3c; margin: 0.5rem 0;'>{depreciation:.1f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='text-align: center;'>
                    <p style='color: #667eea; font-size: 0.9rem; margin: 0;'>MAXIMUM PRICE</p>
                    <p style='font-size: 1.8rem; font-weight: 700; color: #333; margin: 0.5rem 0;'>💲{upper_actual:,.0f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional info
        with st.expander("📊 Estimation Details", expanded=True):
            st.markdown(f"""
            <div class='info-box'>
                <strong>Device Summary:</strong><br>
                <ul style='margin: 0.5rem 0;'>
                    <li><strong>Brand:</strong> {brand}</li>
                    <li><strong>Original Price:</strong> 💲{actual_new_price:,}</li>
                    <li><strong>Usage:</strong> {days_used} days ({days_used//365} years)</li>
                    <li><strong>Specs:</strong> {ram}GB RAM | {internal_memory}GB Storage | {screen_size}\" Screen</li>
                </ul>
                <br>
                <strong>Prediction Range:</strong> ±10% confidence interval<br>
                The estimated resale price accounts for device age, condition, and market demand.
            </div>
            """, unsafe_allow_html=True)

