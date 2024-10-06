# main.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import os
import requests

# Page configuration
st.set_page_config(
    page_title="ER Waiting Time Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTitle {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .stSubheader {
        font-size: 1.5rem;
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
    
# Load model and scaler
# Load model and scaler
@st.cache_resource
def load_prediction_resources():
    try:
        # GitHub raw URLs for your files
        model_url = "https://github.com/hhijry-coder/ER_predictor/raw/main/best_model%20(1).h5"
        scaler_url = "https://github.com/hhijry-coder/ER_predictor/raw/main/scaler%20(1).joblib"
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download and save model
            model_path = os.path.join(tmp_dir, 'best_model (1).h5')
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
            
            # Download and save scaler
            scaler_path = os.path.join(tmp_dir, 'scaler (1).joblib')
            response = requests.get(scaler_url)
            with open(scaler_path, 'wb') as f:
                f.write(response.content)
            
            # Load the files
            model = load_model(model_path, custom_objects={'mse': mse})
            scaler = joblib.load(scaler_path)
            
            return model, scaler
            
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

model, scaler = load_prediction_resources()

# Main content
st.title("🏥 ER Waiting Time Predictor")

# Sidebar for online input
with st.sidebar:
    st.subheader("Input Parameters")
    
    # Current timestamp
    current_time = datetime.datetime.now()
    
    # Date and time input
    date_input = st.date_input("Select Date", current_time.date())
    time_input = st.time_input("Select Time", current_time.time())
    
    # Combine date and time
    arrival_time = datetime.datetime.combine(date_input, time_input)
    
    # Extract features
    hour = arrival_time.hour
    minutes = arrival_time.minute
    day_of_week = arrival_time.weekday()
    
    # Other inputs
    x3 = st.number_input("X3 Value", min_value=0.0, max_value=100.0, value=50.0)
    waiting_people = st.number_input("Number of Waiting People", min_value=0, max_value=100, value=10)
    service_time = st.number_input("Service Time (minutes)", min_value=0, max_value=180, value=30)
    
    predict_button = st.button("Predict Waiting Time")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction Results")
    
    if predict_button:
        if model is not None and scaler is not None:
            # Prepare input data
            input_data = pd.DataFrame([[x3, hour, minutes, waiting_people, day_of_week, service_time]], 
                                    columns=['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime'])
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Reshape for LSTM if using LSTM model
            input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
            
            # Make prediction
            prediction = model.predict(input_reshaped)
            
            # Display prediction
            st.markdown(f"""
            ### Estimated Waiting Time:
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <h2 style='color: #1f77b4; text-align: center;'>{prediction[0][0]:.1f} minutes</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional context
            st.markdown("""
            #### Prediction Context:
            - This prediction takes into account current ER conditions and historical patterns
            - Actual waiting times may vary based on emergency cases and unforeseen circumstances
            """)
            
            # Confidence metrics
            st.markdown("#### Reliability Indicators:")
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            
            with col_metrics1:
                st.metric("Current Load", f"{waiting_people} patients")
            with col_metrics2:
                st.metric("Time of Day", f"{hour:02d}:{minutes:02d}")
            with col_metrics3:
                st.metric("Service Time", f"{service_time} min")

with col2:
    st.subheader("About the Prediction Model")
    st.markdown("""
    #### How it Works
    This prediction model uses advanced machine learning techniques to estimate ER waiting times based on:
    - Current time and day of week
    - Number of waiting patients
    - Historical service times
    - Other relevant factors (X3)
    
    #### Limitations
    - Predictions are estimates based on historical patterns
    - Emergency cases may affect actual waiting times
    - External factors might not be captured in the model
    - Regular model updates are required for optimal performance
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 ER Waiting Time Predictor v1.0 | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
