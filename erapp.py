import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error as sklearn_mse
import joblib
import os
import tensorflow as tf
from keras.saving import register_keras_serializable
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Register the custom MSE function with Keras
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load the best model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        if os.path.exists('best_model.h5'):
            custom_objects = {'mse': mse}
            model = load_model('best_model.h5', custom_objects=custom_objects)
        else:
            st.error("Model file 'best_model.h5' not found.")
            return None, None

        if os.path.exists('scaler.joblib'):
            scaler = joblib.load('scaler.joblib')
        else:
            st.error("Scaler file 'scaler.joblib' not found.")
            return None, None

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

# Define a function to load the uploaded data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Define a function to preprocess the data
def preprocess_data(data):
    try:
        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Rename columns to match the feature names the model was trained with
        expected_features = ['X1', 'X2', 'waitingTime', 'dayOfWeek']
        actual_features = list(data.columns)
        
        # Handling missing expected features
        for feature in expected_features:
            if feature not in actual_features:
                data[feature] = 0  # Placeholder value, adjust as needed

        # Reorder the columns to match the order during training
        data = data[expected_features]

        return data
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(y_true, y_pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y_true)), y=y_true, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_pred)), y=y_pred, mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Index', yaxis_title='Waiting Time')
    return fig

# Function to plot time series predictions
def plot_time_series(y_true, y_pred, timestamps, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=y_true, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=timestamps, y=y_pred, mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Timestamp', yaxis_title='Waiting Time')
    return fig

# Function to plot residuals
def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    fig = go.Figure(go.Scatter(x=np.arange(len(residuals)), y=residuals, mode='markers', name='Residuals'))
    fig.update_layout(title=title, xaxis_title='Index', yaxis_title='Residual')
    return fig

# Function to plot error distribution
def plot_error_distribution(y_true, y_pred, title):
    errors = y_true - y_pred
    fig = px.histogram(errors, nbins=30, title=title, labels={'value': 'Prediction Error'})
    return fig

# Function to create hospital map
def create_hospital_map(city_name, radius):
    geolocator = Nominatim(user_agent="hospital_finder")
    try:
        location = geolocator.geocode(city_name)
        if location:
            m = folium.Map(location=[location.latitude, location.longitude], zoom_start=12)
            folium.Marker([location.latitude, location.longitude], popup="Center").add_to(m)
            # Adding a placeholder circle for the search radius
            folium.Circle([location.latitude, location.longitude], radius=radius, color='blue', fill=True).add_to(m)
            return m
        else:
            return None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.error(f"Geocoding error: {str(e)}")
        return None

# Load the model and scaler once
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
            if data is not None:
                data = preprocess_data(data)
                if data is not None:
                    st.write("Data Preview:")
                    st.dataframe(data.head())
                    
                    # Feature selection
                    X = data.drop(columns=['waitingTime'])
                    y = data['waitingTime']
                    
                    # Scale features
                    X_scaled = scaler.transform(X)
                    
                    # Make predictions
                    y_pred = model.predict(X_scaled).flatten()
                    
                    # Evaluate model
                    mae = mean_absolute_error(y, y_pred)
                    mse = sklearn_mse(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    
                    st.write("Model Performance:")
                    st.write(f"MAE: {mae:.2f}")
                    st.write(f"MSE: {mse:.2f}")
                    st.write(f"R2: {r2:.2f}")
                    
                    # Visualizations
                    st.write("Actual vs Predicted Plot:")
                    fig_actual_vs_pred = plot_actual_vs_predicted(y, y_pred, 'Best Model')
                    st.plotly_chart(fig_actual_vs_pred)
                    
                    st.write("Time Series Plot:")
                    fig_time_series = plot_time_series(y, y_pred, data.index, 'Best Model')
                    st.plotly_chart(fig_time_series)
                    
                    st.write("Residual Analysis:")
                    fig_residuals = plot_residuals(y, y_pred, 'Best Model')
                    st.plotly_chart(fig_residuals)
                    
                    st.write("Error Distribution:")
                    fig_error_dist = plot_error_distribution(y, y_pred, 'Best Model')
                    st.plotly_chart(fig_error_dist)
                else:
                    st.error("Error during data preprocessing.")
            else:
                st.error("Error loading data.")
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
