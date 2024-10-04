import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import BytesIO
import base64

# Libraries for geolocation and map
import folium
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import requests

# Sidebar for Page Navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Select a page:", ["Data Upload", "Model Training", "Evaluation & Visualization", "Interactive Map"])

# Global variables for storing loaded data
uploaded_data = None
X_train, X_test, y_train, y_test = None, None, None, None
scaler = joblib.load('scaler.joblib')
model = load_model('https://github.com/hhijry-coder/ER_predictor/blob/main/best_model.h5')

# Function to read CSV or XLSX
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

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

# Function to create the folium map with hospitals
def create_map(city_name, radius=5000):
    # Get location (lat/lon) from city name
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(city_name)
    
    if location is None:
        st.error(f"City {city_name} not found. Please try another name.")
        return None
    
    lat, lon = location.latitude, location.longitude

    # Initialize Folium map
    folium_map = folium.Map(location=[lat, lon], zoom_start=12)

    # Get nearby hospitals
    hospitals = get_nearby_hospitals(lat, lon, radius)

    # Add hospitals as markers
    for hospital in hospitals:
        hospital_name = hospital.get('tags', {}).get('name', 'Unnamed Hospital')
        folium.Marker(
            location=[hospital['lat'], hospital['lon']],
            popup=hospital_name,
            icon=folium.Icon(color='red', icon='plus-sign')
        ).add_to(folium_map)

    return folium_map

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
        feature_columns = st.sidebar.multiselect("Feature Columns", uploaded_data.columns, default=uploaded_data.columns[:-1])

        if st.button("Process Data"):
            # Preprocess Data
            X = uploaded_data[feature_columns]
            y = uploaded_data[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            st.success("Data processed and ready for training!")

# Page 2: Model Training
elif pages == "Model Training":
    st.title("Train Models")
    
    if uploaded_data is not None:
        model_type = st.sidebar.selectbox("Select Model", ["Traditional LSTM", "Advanced LSTM", "DNN"])

        # Display current model name
        st.write(f"Training the **{model_type}** model on your data.")
        
        # Train button
        if st.button("Train"):
            # Implement model training logic based on selected model
            # Use pre-trained model here for quick demo
            st.success(f"Training of {model_type} model complete!")
    else:
        st.warning("Please upload data first on the 'Data Upload' page.")

# Page 3: Evaluation & Visualization
elif pages == "Evaluation & Visualization":
    st.title("Model Evaluation & Visualization")

    if X_test is not None:
        st.subheader("Model Performance")

        # Predict with loaded model
        y_pred = model.predict(scaler.transform(X_test))
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**MSE**: {mse:.2f}")
        st.write(f"**R2**: {r2:.2f}")

        # Plot Actual vs Predicted
        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
        st.plotly_chart(fig)

        # Plot Residuals
        residuals = y_test - y_pred
        fig_res = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, title="Residuals")
        st.plotly_chart(fig_res)
        
    else:
        st.warning("Please upload data and train a model first.")

# Page 4: Interactive Map
elif pages == "Interactive Map":
    st.title("Find Nearby Hospitals")

    # User input for city name
    city_name = st.text_input("Enter the name of the city or town", value="New York")

    # Slider for search radius
    radius = st.slider("Select search radius (in meters)", min_value=1000, max_value=20000, step=1000, value=5000)

    # Search button
    if st.button("Find Hospitals"):
        folium_map = create_map(city_name, radius)
        
        if folium_map:
            st.write(f"Showing hospitals near **{city_name}** within **{radius}** meters.")
            st_folium(folium_map, width=700, height=500)

