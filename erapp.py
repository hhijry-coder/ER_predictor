import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import folium
from streamlit_folium import st_folium
import requests
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from functools import lru_cache
import time
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Hospital Wait Time Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container {
        background: #fafafa;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Set style for seaborn plots
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-darkgrid")

@lru_cache(maxsize=100)
def _cached_geocoding(city_name):
    geolocator = Nominatim(user_agent="hospital_wait_time_predictor_1.0")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    return None

# Rate limiting decorator
def rate_limit(seconds):
    last_time = [0]  # Using list to make it mutable in closure
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if current_time - last_time[0] < seconds:
                time.sleep(seconds - (current_time - last_time[0]))
            last_time[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(1)  # Limit to 1 request per second
def get_city_coordinates(city_name):
    """
    Get coordinates for a city name using multiple geocoding attempts and error handling.
    
    Args:
        city_name (str): Name of the city to geocode
        
    Returns:
        tuple: (latitude, longitude) if successful, None if failed
    """
    if not city_name or not isinstance(city_name, str):
        st.error("Please enter a valid city name")
        return None
        
    # Clean up city name
    city_name = city_name.strip()
    
    try:
        # First try cached results
        coords = _cached_geocoding(city_name)
        if coords:
            return coords
            
        # If not in cache, try with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                geolocator = Nominatim(user_agent="hospital_wait_time_predictor_1.0")
                location = geolocator.geocode(city_name)
                
                if location:
                    return location.latitude, location.longitude
                    
            except GeocoderTimedOut:
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    time.sleep(retry_delay)
                continue
                
            except GeocoderUnavailable:
                st.error("Geocoding service is currently unavailable. Please try again later.")
                return None
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                return None
                
        st.warning(f"""
            Could not find coordinates for "{city_name}". 
            Try:
            1. Checking the spelling
            2. Adding the country name (e.g., "Paris, France")
            3. Using a more specific location
        """)
        return None
        
    except Exception as e:
        st.error(f"Error while geocoding: {str(e)}")
        return None

# Helper function to validate coordinates
def validate_coordinates(coords):
    """
    Validate that coordinates are within valid ranges.
    
    Args:
        coords (tuple): (latitude, longitude)
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not coords or not isinstance(coords, tuple) or len(coords) != 2:
        return False
        
    lat, lon = coords
    try:
        lat_float = float(lat)
        lon_float = float(lon)
        return -90 <= lat_float <= 90 and -180 <= lon_float <= 180
    except (TypeError, ValueError):
        return False

# Load the saved model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model('best_model.keras')
        with open('scaler_model.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model and scaler: {str(e)}")
        return None, None

# Function to preprocess new data
def preprocess_data(data, features, scaler):
    X_new = data[features]
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))
    return X_new_scaled

# Function to make predictions
def make_predictions(model, X_new_scaled):
    return model.predict(X_new_scaled).flatten()

# Function to display fancy predicted waiting time
def display_fancy_prediction(predicted_time):
    st.markdown(
        f"""
        <div style="background-color:#4CAF50;padding:20px;border-radius:10px">
        <h2 style="color:white;text-align:center;">Predicted Waiting Time: {predicted_time:.2f} minutes</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main application
def main():
    st.title("🏥 Hospital Wait Time Predictor")
    st.write("Predict hospital waiting times and explore nearby hospitals")

    # Load the model and scaler
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.error("Failed to load model and scaler. Please check the files.")
        return

    # Define the feature columns
    features = ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']
    target = 'waitingTime'

    # Create main columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.sidebar.header("Input Options")
        data_source = st.sidebar.radio("Choose the data source", ("Manual Input", "Upload CSV/XLSX File"))

        if data_source == "Manual Input":
            st.sidebar.subheader("Input values manually")
            X3 = st.sidebar.number_input("X3", min_value=0.0, value=10.0)
            hour = st.sidebar.slider("Hour", 0, 23, 12)
            minutes = st.sidebar.slider("Minutes", 0, 59, 30)
            waitingPeople = st.sidebar.number_input("Waiting People", min_value=0, value=5)
            dayOfWeek = st.sidebar.selectbox(
                "Day of the Week", 
                options=[0, 1, 2, 3, 4, 5, 6], 
                format_func=lambda x: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][x]
            )
            serviceTime = st.sidebar.number_input("Service Time", min_value=0.0, value=20.0)

            user_data = pd.DataFrame({
                'X3': [X3],
                'hour': [hour],
                'minutes': [minutes],
                'waitingPeople': [waitingPeople],
                'dayOfWeek': [dayOfWeek],
                'serviceTime': [serviceTime]
            })

            st.write("### Your Input Data")
            st.write(user_data)

            if st.button("Predict Waiting Time"):
                X_new_scaled = preprocess_data(user_data, features, scaler)
                predictions = make_predictions(model, X_new_scaled)
                display_fancy_prediction(predictions[0])

        else:
            uploaded_file = st.sidebar.file_uploader("Upload your CSV or XLSX file", type=["csv", "xlsx"])

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)

                    st.write("### Uploaded Data Preview")
                    st.write(data.head())

                    X_new_scaled = preprocess_data(data, features, scaler)
                    predictions = make_predictions(model, X_new_scaled)
                    data['Predicted_WaitingTime'] = predictions
                    
                    st.write("### Predictions")
                    st.write(data[['Predicted_WaitingTime']])
                    display_fancy_prediction(predictions[0])

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    with col2:
        st.subheader("Hospital Map")
        city_name = st.text_input("Enter a City Name to View Nearby Hospitals")
        
        if city_name:
            with st.spinner("Fetching city coordinates..."):
                city_coords = get_city_coordinates(city_name)
                
            if city_coords and validate_coordinates(city_coords):
                with st.spinner("Searching for nearby hospitals..."):
                    hospitals = get_hospitals_near_city(*city_coords)
                    
                if hospitals:
                    st.success(f"Found {len(hospitals)} hospitals near {city_name}")
                    
                    with st.spinner("Creating map..."):
                        city_map = display_hospital_map(hospitals, city_coords)
                        
                    if city_map is not None:
                        st_folium(city_map, width=800, height=500)
                        
                        with st.expander("View Hospital List"):
                            for idx, hospital in enumerate(hospitals, 1):
                                st.write(f"{idx}. {hospital['name']}")
                    else:
                        st.error("Failed to create map. Please try again.")
                else:
                    st.warning("""
                        No hospitals found in the area. 
                        Try:
                        1. Increasing the search radius
                        2. Checking a different location
                        3. Using a more specific address
                    """)
            else:
                st.error("""
                    Could not find the specified location. 
                    Please:
                    1. Check the spelling
                    2. Add the country name (e.g., "Paris, France")
                    3. Try a more specific location
                """)

        # Visualization section
        st.subheader("Data Visualizations")
        if data_source == "Upload CSV/XLSX File" and 'data' in locals():
            viz_type = st.selectbox(
                "Choose Visualization Type",
                ["Descriptive Analysis", "Box Plot", "Frequency Distribution", "Actual vs Predicted"]
            )

            if viz_type == "Descriptive Analysis":
                st.write("### Descriptive Statistics")
                st.write(data.describe())
            
            elif viz_type == "Box Plot":
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(data=data[features + [target]])
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            elif viz_type == "Frequency Distribution":
                n_features = len(features) + 1
                n_cols = 2
                n_rows = (n_features + 1) // 2
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                axes = axes.flatten()
                
                for i, feature in enumerate(features + [target]):
                    sns.histplot(data[feature], kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {feature}')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif viz_type == "Actual vs Predicted" and target in data.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=data, x=target, y='Predicted_WaitingTime')
                plt.plot([data[target].min(), data[target].max()], 
                        [data[target].min(), data[target].max()], 
                        'r--', lw=2)
                plt.xlabel("Actual Waiting Time")
                plt.ylabel("Predicted Waiting Time")
                plt.title("Actual vs Predicted Waiting Times")
                st.pyplot(fig)

if __name__ == "__main__":
    main()
