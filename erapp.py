import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tensorflow.keras.metrics import mean_squared_error as mse
import joblib
import folium
from streamlit_folium import folium_static
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import os

# Load the best model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        if os.path.exists('best_model.h5'):
            # Define custom objects
            custom_objects = {'mse': mse}
            model = load_model('best_model.h5', custom_objects=custom_objects)
        else:
            st.error("Model file 'best_model.h5' not found. Please ensure the file is in the correct location.")
            return None, None

        if os.path.exists('scaler.joblib'):
            scaler = joblib.load('scaler.joblib')
        else:
            st.error("Scaler file 'scaler.joblib' not found. Please ensure the file is in the correct location.")
            return None, None

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        st.error("If the error persists, you may need to retrain and save the model using the current version of Keras/TensorFlow.")
        return None, None

model, scaler = load_model_and_scaler()

# Streamlit app
st.title('Waiting Time Prediction and Hospital Finder App')

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Waiting Time Prediction", "Hospital Finder"])

if page == "Waiting Time Prediction":
    st.header("Waiting Time Prediction")
    
    if model is None or scaler is None:
        st.error("Unable to load the model or scaler. Please check the error messages above.")
    else:
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            data = load_data(uploaded_file)
            data = preprocess_data(data)
            
            st.write("Data Preview:")
            st.dataframe(data.head())
            
            # Feature selection and scaling
            X = data[features]
            y = data[target]
            X_scaled = scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled).flatten()
            
            # Evaluate model
            mae, mse, r2 = evaluate_model(y, y_pred, 'Best Model')
            
            st.write("Model Performance:")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"MSE: {mse:.2f}")
            st.write(f"R2: {r2:.2f}")
            
            # Visualizations
            st.write("Actual vs Predicted Plot:")
            fig_actual_vs_pred = plot_actual_vs_predicted(y, y_pred, 'Best Model')
            st.plotly_chart(fig_actual_vs_pred)
            
            st.write("Time Series Plot:")
            fig_time_series = plot_time_series(y, y_pred, data['timestamp'], 'Best Model')
            st.plotly_chart(fig_time_series)
            
            st.write("Residual Analysis:")
            fig_residuals = plot_residuals(y, y_pred, 'Best Model')
            st.plotly_chart(fig_residuals)
            
            st.write("Error Distribution:")
            fig_error_dist = plot_error_distribution(y, y_pred, 'Best Model')
            st.plotly_chart(fig_error_dist)

        else:
            st.write("Please upload a CSV or Excel file to begin the analysis.")

elif page == "Hospital Finder":
    st.header("Hospital Finder")
    
    city_name = st.text_input("Enter a city or town name:")
    radius = st.slider("Select search radius (in meters)", min_value=1000, max_value=10000, value=5000, step=1000)
    
    if st.button("Find Hospitals"):
        if city_name:
            with st.spinner("Searching for hospitals..."):
                hospital_map = create_hospital_map(city_name, radius)
            
            if hospital_map:
                st.success(f"Showing hospitals near {city_name} within a {radius}m radius")
                folium_static(hospital_map)
            else:
                st.error("Unable to find the specified location. Please try another city or town name.")
        else:
            st.warning("Please enter a city or town name.")
