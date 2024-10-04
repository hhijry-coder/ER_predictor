import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import folium
from streamlit_folium import folium_static
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Load the best model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model('best_model.h5')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

# Function to read CSV or XLSX
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

# Preprocessing function
def preprocess_data(data):
    data['Arrival time'] = pd.to_datetime(data['Arrival time'], format='%m/%d/%Y %H:%M')
    data['timestamp'] = data['Arrival time']
    data = data.sort_values('timestamp')
    data = data.drop(['X1', 'X2'], axis=1)
    return data

# Feature selection
features = ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']
target = 'waitingTime'

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Plotting functions
def plot_actual_vs_predicted(y_true, y_pred, model_name):
    fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual Waiting Time', 'y': 'Predicted Waiting Time'},
                     title=f'{model_name}: Actual vs Predicted')
    fig.add_trace(go.Scatter(x=y_true, y=y_true, mode='lines', name='Ideal'))
    return fig

def plot_time_series(actual, predicted, dates, model_name):
    df = pd.DataFrame({'Date': dates, 'Actual': actual, 'Predicted': predicted})
    df['Actual_Smooth'] = df['Actual'].rolling(window=24).mean()
    df['Predicted_Smooth'] = df['Predicted'].rolling(window=24).mean()
    
    fig = px.line(df, x='Date', y=['Actual', 'Predicted', 'Actual_Smooth', 'Predicted_Smooth'],
                  title=f'Time Series of Actual vs Predicted Waiting Times - {model_name}')
    return fig

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'},
                     title=f'Residual Analysis - {model_name}')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig

def plot_error_distribution(y_true, y_pred, model_name):
    errors = y_true - y_pred
    fig = px.histogram(errors, nbins=50, labels={'value': 'Error'},
                       title=f'{model_name} Error Distribution')
    return fig

# Hospital finder functions
def get_location(city_name):
    geolocator = Nominatim(user_agent="hospital_finder")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        else:
            return None
    except (GeocoderTimedOut, GeocoderServiceError):
        return None

def get_nearby_hospitals(lat, lon, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    node["amenity"="hospital"](around:{radius},{lat},{lon});
    out;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data['elements']

def create_hospital_map(city_name, radius=5000):
    location = get_location(city_name)
    if location:
        lat, lon = location
        hospitals = get_nearby_hospitals(lat, lon, radius)
        
        m = folium.Map(location=[lat, lon], zoom_start=12)
        
        for hospital in hospitals:
            folium.Marker(
                [hospital['lat'], hospital['lon']],
                popup=hospital.get('tags', {}).get('name', 'Unknown Hospital'),
                icon=folium.Icon(color='red', icon='plus-sign')
            ).add_to(m)
        
        return m
    else:
        return None

# Streamlit app
st.title('Waiting Time Prediction and Hospital Finder App')

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Waiting Time Prediction", "Hospital Finder"])

if page == "Waiting Time Prediction":
    st.header("Waiting Time Prediction")
    
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
