import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import folium
from streamlit_folium import st_folium
import requests
import os
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from requests.exceptions import RequestException

# Paths for scaler and model
scaler_path = 'scaler.joblib'
model_path = 'best_model.h5'

# Function to read CSV or XLSX
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

def convert_datetime_to_numeric(X):
    X = X.copy()
    for col in X.select_dtypes(include=['datetime64']).columns:
        X[col] = X[col].astype(int) // 10**9
    return X

# Function to get hospitals using OpenStreetMap (Overpass API)
def get_nearby_hospitals(lat, lon, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    node
      [amenity=hospital]
      (around:{radius},{lat},{lon});
    out;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data['elements']

# Function to geocode with retry and caching
@st.cache_data(ttl=3600)  # Cache results for 1 hour
def geocode_with_retry(city_name, max_retries=3, initial_delay=1, backoff_factor=2):
    geolocator = Nominatim(user_agent="geoapi")
    
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(city_name, timeout=10)
            if location:
                return location.latitude, location.longitude
            else:
                st.error(f"City {city_name} not found. Please try another name.")
                return None
        except (GeocoderTimedOut, GeocoderServiceError, RequestException) as e:
            if attempt == max_retries - 1:
                st.error(f"Error geocoding city after {max_retries} attempts: {e}")
                return None
            else:
                delay = initial_delay * (backoff_factor ** attempt)
                time.sleep(delay)
    
    return None

# Function to create the folium map with hospitals
def create_map(city_name, radius=5000):
    location = geocode_with_retry(city_name)
    
    if location is None:
        return None
    
    lat, lon = location

    folium_map = folium.Map(location=[lat, lon], zoom_start=12)
    hospitals = get_nearby_hospitals(lat, lon, radius)

    for hospital in hospitals:
        hospital_name = hospital.get('tags', {}).get('name', 'Unnamed Hospital')
        folium.Marker(
            location=[hospital['lat'], hospital['lon']],
            popup=hospital_name,
            icon=folium.Icon(color='red', icon='plus-sign')
        ).add_to(folium_map)

    return folium_map

# Sidebar for Page Navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Select a page:", ["Data Upload", "Evaluation & Visualization", "Interactive Map"])

# Global variables for storing loaded data
uploaded_data = None
X_test, y_test = None, None
scaler = None
model = None

# Page 1: Data Upload
if pages == "Data Upload":
    st.title("Upload your Dataset")
    
    uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        uploaded_data = load_data(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(uploaded_data.head())
        
        st.write("Column Info:")
        st.write(uploaded_data.describe())

        # Sidebar to select target and features
        st.sidebar.write("Select target column")
        target_column = st.sidebar.selectbox("Target Column", uploaded_data.columns)
        
        st.sidebar.write("Select feature columns")
        feature_columns = st.sidebar.multiselect("Feature Columns", uploaded_data.columns, default=uploaded_data.columns[:-1].tolist())

        if st.button("Process Data"):
            # Preprocess Data
            X = uploaded_data[feature_columns]
            y = uploaded_data[target_column]
            
            X_test = convert_datetime_to_numeric(X)
            y_test = y

            # Load the scaler
            scaler = joblib.load(scaler_path)

            # Scale the data
            X_test_scaled = scaler.transform(X_test)

            st.session_state['X_test_scaled'] = X_test_scaled
            st.session_state['y_test'] = y_test

            st.success("Data processed and ready for evaluation!")

# Page 2: Evaluation & Visualization
elif pages == "Evaluation & Visualization":
    st.title("Model Evaluation & Visualization")

    if 'X_test_scaled' in st.session_state and 'y_test' in st.session_state:
        st.subheader("Model Performance")

        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Predict with loaded model
        y_pred = model.predict(st.session_state['X_test_scaled'])
        
        # Calculate performance metrics
        mae = mean_absolute_error(st.session_state['y_test'], y_pred)
        mse = mean_squared_error(st.session_state['y_test'], y_pred)
        r2 = r2_score(st.session_state['y_test'], y_pred)

        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**MSE**: {mse:.2f}")
        st.write(f"**R2**: {r2:.2f}")

        # Plot Actual vs Predicted
        fig = px.scatter(x=st.session_state['y_test'], y=y_pred.flatten(), labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
        st.plotly_chart(fig)

        # Plot Residuals
        residuals = st.session_state['y_test'] - y_pred.flatten()
        fig_res = px.scatter(x=y_pred.flatten(), y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, title="Residuals")
        st.plotly_chart(fig_res)
        
    else:
        st.warning("Please upload and process data first on the 'Data Upload' page.")

# Page 3: Interactive Map
elif pages == "Interactive Map":
    st.title("Find Nearby Hospitals")

    city_name = st.text_input("Enter the name of the city or town", value="New York")
    radius = st.slider("Select search radius (in meters)", min_value=1000, max_value=20000, step=1000, value=5000)

    if st.button("Find Hospitals"):
        with st.spinner("Searching for hospitals..."):
            folium_map = create_map(city_name, radius)
        
        if folium_map:
            st.write(f"Showing hospitals near **{city_name}** within **{radius}** meters.")
            st_folium(folium_map, width=700, height=500)
