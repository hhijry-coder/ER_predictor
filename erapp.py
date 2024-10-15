import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import folium
from streamlit_folium import st_folium
import requests
from folium.plugins import MarkerCluster
from functools import lru_cache
import time
import streamlit as st

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
    out center qt;
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
        
        Alternative options:
        1. Try a different city
        2. Refresh the page
        3. Check your internet connection
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
        model = load_model('best_model.keras')
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
    return model.predict(X_new_scaled).flatten()

def display_fancy_prediction(predicted_time):
    st.markdown(
        f"""
        <div style="background-color:#4CAF50;padding:20px;border-radius:10px">
        <h2 style="color:white;text-align:center;">Predicted Waiting Time: {predicted_time:.2f} minutes</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def visualize_manual_input(user_data, predicted_time):
    st.subheader("Visualizations")
    
    # Create subplots for individual feature visualizations
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(user_data.columns))
    
    for idx, feature in enumerate(user_data.columns):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        # Bar plot for each feature
        fig.add_trace(
            go.Bar(x=[feature], y=[user_data[feature].values[0]], name=feature),
            row=row, col=col
        )
        
        # Update y-axis range for better visibility
        fig.update_yaxes(range=[0, max(user_data[feature].values[0] * 1.2, 1)], row=row, col=col)
    
    fig.update_layout(height=600, showlegend=False, title_text="Input Feature Values")
    st.plotly_chart(fig, use_container_width=True)
    

def visualize_batch_data(data, predictions, target):
    st.subheader("Visualizations")
    
    # Calculate prediction errors
    data['Predicted_WaitingTime'] = predictions
    if target in data.columns:
        data['Error'] = data[target] - data['Predicted_WaitingTime']
    
    # Actual vs Predicted Scatter Plot with Best Fit Line
    if target in data.columns:
        fig_scatter = px.scatter(data, x=target, y='Predicted_WaitingTime', trendline="ols")
        fig_scatter.update_layout(title="Actual vs Predicted Waiting Time", 
                                  xaxis_title="Actual Waiting Time",
                                  yaxis_title="Predicted Waiting Time")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Individual feature distributions
    features = ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']
    fig_dist = make_subplots(rows=2, cols=3, subplot_titles=features)
    
    for idx, feature in enumerate(features):
        row = idx // 3 + 1
        col = idx % 3 + 1
        fig_dist.add_trace(
            go.Histogram(x=data[feature], name=feature),
            row=row, col=col
        )
    
    fig_dist.update_layout(height=800, showlegend=False, title_text="Feature Distributions")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Individual feature box plots
    fig_box = make_subplots(rows=2, cols=3, subplot_titles=features)
    
    for idx, feature in enumerate(features):
        row = idx // 3 + 1
        col = idx % 3 + 1
        fig_box.add_trace(
            go.Box(y=data[feature], name=feature),
            row=row, col=col
        )
    
    fig_box.update_layout(height=800, showlegend=False, title_text="Feature Box Plots")
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Prediction Error Histogram
    if target in data.columns:
        fig_error = px.histogram(data, x='Error', nbins=50)
        fig_error.update_layout(title="Prediction Error Distribution")
        st.plotly_chart(fig_error, use_container_width=True)

def main():
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.error("Failed to load model and scaler. Please check the files.")
        return

    # Features and target
    features = ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']
    target = 'waitingTime'

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Manual Input Visualization", "Batch Data Visualization", "Hospital Locator"])

    if page == "Manual Input Visualization":
        st.header("Manual Input Visualization")
        
        # Use columns to create a more efficient layout
        col1, col2, col3 = st.columns(3)

        with col1:
            total_time = st.number_input("Total Time (Waiting time + Service Time)", min_value=0.0, value=10.0)
            hour = st.slider("Hour", 0, 23, 12)
        
        with col2:
            minutes = st.slider("Minutes", 0, 59, 30)
            waiting_people = st.number_input("Waiting People", min_value=0, value=5)
        
        with col3:
            day_of_week = st.selectbox(
                "Day of the Week", 
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][x]
            )
            service_time = st.number_input("Service Time", min_value=0.0, value=20.0)

        if st.button("Predict Waiting Time"):
            user_data = pd.DataFrame({
                'X3': [total_time],
                'hour': [hour],
                'minutes': [minutes],
                'waitingPeople': [waiting_people],
                'dayOfWeek': [day_of_week],
                'serviceTime': [service_time]
            })

            X_new_scaled = preprocess_data(user_data, features, scaler)
            predictions = make_predictions(model, X_new_scaled)
            display_fancy_prediction(predictions[0])

            # Visualize manual input
            visualize_manual_input(user_data, predictions[0])

    elif page == "Batch Data Visualization":
        st.header("Batch Data Visualization")
        uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.write("### Uploaded Data Preview")
                st.write(data.head())

                # Check if required features are present
                if not set(features).issubset(data.columns):
                    st.error("Uploaded data does not contain all the required features.")
                    st.stop()

                # Check if target column exists for visualizations
                has_target = target in data.columns

                X_new_scaled = preprocess_data(data, features, scaler)
                predictions = make_predictions(model, X_new_scaled)
                data['Predicted_WaitingTime'] = predictions

                st.write("### Predictions")
                st.write(data[['Predicted_WaitingTime']])

                visualize_batch_data(data, predictions, target)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.info("Please upload a CSV or XLSX file to proceed.")
            

    elif page == "Hospital Locator":
        st.header("Hospital Locator")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Input City")
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
                        st.warning("""No hospitals found in the area. Try increasing the search radius or using a more specific address.""")
                else:
                    st.error("Could not find the specified location. Please check the spelling or try a more specific location.")

def visualize_manual_input(user_data, predicted_time):
    st.subheader("Visualizations")
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.write("### Feature Values")
        fig_features = go.Figure(data=[
            go.Bar(
                x=user_data.columns,
                y=user_data.iloc[0],
                marker_color='indianred'
            )
        ])
        fig_features.update_layout(title="Input Feature Values", yaxis_title="Value", xaxis_title="Features")
        st.plotly_chart(fig_features, use_container_width=True)
    
    with col2:
        st.write("### Predicted Waiting Time")
        fig_pred = go.Figure(data=[
            go.Indicator(
                mode="gauge+number",
                value=predicted_time,
                title={'text': "Predicted Waiting Time (minutes)"}
            )
        ])
        fig_pred.update_layout(height=300)
        st.plotly_chart(fig_pred, use_container_width=True)
    
    # Additional visualizations can be added here if needed

def visualize_batch_data(data, predictions, target):
    st.subheader("Visualizations")
    
    # Calculate prediction errors
    data['Predicted_WaitingTime'] = predictions
    if target in data.columns:
        data['Error'] = data[target] - data['Predicted_WaitingTime']
    
    # Create subplots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Actual vs Predicted", "Feature Box Plots", "Frequency Histograms", "Prediction Error Histogram"))

    # Actual vs Predicted Scatter Plot with Best Fit Line
    if target in data.columns:
        fig.add_trace(
            go.Scatter(x=data[target], y=data['Predicted_WaitingTime'],
                       mode='markers', name='Predicted vs Actual'),
            row=1, col=1
        )
        # Add best fit line
        if len(data[target]) > 1:
            fit = np.polyfit(data[target], data['Predicted_WaitingTime'], 1)
            fit_fn = np.poly1d(fit)
            fig.add_trace(
                go.Scatter(x=data[target], y=fit_fn(data[target]),
                           mode='lines', name='Best Fit Line'),
                row=1, col=1
            )

    # Box Plots for each feature
    for feature in ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']:
        fig.add_trace(
            go.Box(y=data[feature], name=feature),
            row=1, col=2
        )

    # Frequency Histograms for each feature
    for feature in ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']:
        fig.add_trace(
            go.Histogram(x=data[feature], name=feature, opacity=0.75),
            row=2, col=1
        )

    # Prediction Error Histogram
    if target in data.columns:
        fig.add_trace(
            go.Histogram(x=data['Error'], nbinsx=50, name='Error'),
            row=2, col=2
        )

    fig.update_layout(height=1000, showlegend=False,
                      title_text="Batch Data Visualizations",
                      template="plotly_white")
    fig.update_xaxes(title_text=target if target in data.columns else "", row=1, col=1)
    fig.update_yaxes(title_text="Waiting Time (minutes)", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
