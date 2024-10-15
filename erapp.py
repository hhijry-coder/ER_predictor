import os
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import folium
from streamlit_folium import st_folium
import requests
from folium.plugins import MarkerCluster
from functools import lru_cache
import time
import streamlit as st

# Disable GPU usage to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set page config
st.set_page_config(
    page_title="HajjCare Flow Optimizer",
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
    div[data-testid="stSidebarNav"] {
        background-image: linear-gradient(#f0f2f6,#f0f2f6);
        padding: 1rem;
        border-radius: 10px;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache successful geocoding results
@lru_cache(maxsize=100)
def _cached_geocoding(city_name):
    geolocator = Nominatim(user_agent="hospital_wait_time_predictor_1.0")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    return None

# Rate limiting decorator
def rate_limit(seconds):
    last_time = [0]
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if current_time - last_time[0] < seconds:
                time.sleep(seconds - (current_time - last_time[0]))
            last_time[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(1)
def get_city_coordinates(city_name):
    if not city_name or not isinstance(city_name, str):
        st.error("Please enter a valid city name")
        return None
        
    city_name = city_name.strip()
    
    try:
        coords = _cached_geocoding(city_name)
        if coords:
            return coords
            
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                geolocator = Nominatim(user_agent="hospital_wait_time_predictor_1.0")
                location = geolocator.geocode(city_name)
                
                if location:
                    return location.latitude, location.longitude
                    
            except GeocoderTimedOut:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
                
            except GeocoderUnavailable:
                st.error("Geocoding service is currently unavailable. Please try again later.")
                return None
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                return None
                
        st.warning("""
            Could not find coordinates for the specified city. 
            Try:
            1. Checking the spelling
            2. Adding the country name (e.g., "Paris, France")
            3. Using a more specific location
        """)
        return None
        
    except Exception as e:
        st.error(f"Error while geocoding: {str(e)}")
        return None

def get_hospitals_near_city(lat, lon, radius=5000):
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter"
    ]
    
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius},{lat},{lon});
      way["amenity"="hospital"](around:{radius},{lat},{lon});
      relation["amenity"="hospital"](around:{radius},{lat},{lon});
    );
    out body center qt;
    """
    
    for endpoint in overpass_endpoints:
        try:
            response = requests.get(
                endpoint,
                params={'data': query},
                timeout=30,
                headers={'User-Agent': 'Hospital-Wait-Time-Predictor/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                hospitals = []
                
                for element in data.get('elements', []):
                    try:
                        if element['type'] == 'node':
                            lat = element['lat']
                            lon = element['lon']
                        else:
                            lat = element.get('center', {}).get('lat')
                            lon = element.get('center', {}).get('lon')
                            
                            if lat is None or lon is None:
                                continue
                        
                        name = element.get('tags', {}).get('name', 'Unnamed Hospital')
                        hospitals.append({
                            'name': name,
                            'lat': lat,
                            'lon': lon
                        })
                    except KeyError:
                        continue
                
                return hospitals
                
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            continue
            
    st.error(f"""
        Unable to fetch hospital data. Please try again later.
        Error: {last_error}
    """)
    return []

def is_valid_coordinates(lat, lon):
    try:
        return (
            isinstance(lat, (int, float)) and
            isinstance(lon, (int, float)) and
            -90 <= lat <= 90 and
            -180 <= lon <= 180
        )
    except:
        return False

def display_hospital_map(hospitals, city_coords):
    try:
        if not is_valid_coordinates(*city_coords):
            st.error("Invalid coordinates provided")
            return None
            
        m = folium.Map(location=city_coords, zoom_start=12)
        marker_cluster = MarkerCluster().add_to(m)
        
        for hospital in hospitals:
            if is_valid_coordinates(hospital['lat'], hospital['lon']):
                folium.Marker(
                    location=[hospital['lat'], hospital['lon']],
                    popup=hospital['name'],
                    icon=folium.Icon(color='red', icon='plus', prefix='fa')
                ).add_to(marker_cluster)
        
        folium.Marker(
            location=city_coords,
            popup='City Center',
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
        return m
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model('best_model.keras', compile=False)  # Disable optimizer warnings
        with open('scaler_model.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model and scaler: {str(e)}")
        return None, None

def preprocess_data(data, features, scaler):
    X_new = data[features]
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))
    return X_new_scaled

def make_predictions(model, X_new_scaled):
    try:
        return model.predict(X_new_scaled).flatten()
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def display_fancy_prediction(predicted_time):
    st.markdown(
        f"""
        <div style="background-color:#4CAF50;padding:20px;border-radius:10px">
        <h2 style="color:white;text-align:center;">Predicted Waiting Time: {predicted_time:.2f} minutes</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to plot based on user selection
def plot_visualization(data, predictions, selected_plot):
    if selected_plot == "Actual vs Predicted":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(predictions)), y=data['waitingTime'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=np.arange(len(predictions)), y=predictions, mode='lines+markers', name='Predicted'))
        fig.update_layout(title="Actual vs Predicted Waiting Time", xaxis_title="Instance", yaxis_title="Waiting Time (mins)")
        st.plotly_chart(fig)

    elif selected_plot == "Box Plot":
        fig = px.box(data, y="waitingTime", title="Box Plot of Waiting Times")
        st.plotly_chart(fig)

    elif selected_plot == "Histogram":
        fig = px.histogram(data, x="waitingTime", title="Histogram of Waiting Times")
        st.plotly_chart(fig)

def main():
    st.title("🏥 HajjCare Flow Optimizer")
    st.write("Predict hospital waiting times and explore nearby hospitals")

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.error("Failed to load model and scaler. Please check the files.")
        return

    # Features list
    features = ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']

    col1, col2 = st.columns([1, 2])

    with col1:
        st.sidebar.header("Input Options")
        data_source = st.sidebar.radio("Choose the data source", ("Manual Input", "Upload CSV/XLSX File"))

        if data_source == "Manual Input":
            st.sidebar.subheader("Input values manually")
            total_time = st.sidebar.number_input("Total Time (Waiting time + Service Time)", min_value=0.0, value=10.0)
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
                'X3': [total_time],
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
                if predictions is not None:
                    display_fancy_prediction(predictions[0])
                    st.write("### Select a Plot")
                    selected_plot = st.selectbox("Choose Plot", ["Actual vs Predicted", "Box Plot", "Histogram"])
                    plot_visualization(user_data, predictions, selected_plot)

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

                    if all(col in data.columns for col in features):
                        X_new_scaled = preprocess_data(data, features, scaler)
                        predictions = make_predictions(model, X_new_scaled)
                        if predictions is not None:
                            data['Predicted_WaitingTime'] = predictions
                            st.write("### Predictions")
                            st.write(data[['Predicted_WaitingTime']])
                            display_fancy_prediction(predictions[0])
                            st.write("### Select a Plot")
                            selected_plot = st.selectbox("Choose Plot", ["Actual vs Predicted", "Box Plot", "Histogram"])
                            plot_visualization(data, predictions, selected_plot)
                    else:
                        st.error("Uploaded data does not contain required features.")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    with col2:
        st.subheader("Hospital Map")
        city_name = st.text_input("Enter a City Name to View Nearby Hospitals")
        
        if city_name:
            with st.spinner("Fetching city coordinates..."):
                city_coords = get_city_coordinates(city_name)
                
            if city_coords and is_valid_coordinates(*city_coords):
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
                    st.warning("No hospitals found in the area. Try increasing the search radius or using a more specific address.")
            else:
                st.error("Could not find the specified location. Please check the spelling or try a more specific location.")

if __name__ == "__main__":
    main()
