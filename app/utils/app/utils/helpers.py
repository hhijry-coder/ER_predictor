import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import requests
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
from datetime import datetime
import io

def load_prediction_resources():
    """Load model and scaler from GitHub repository."""
    try:
        # GitHub raw URLs for your files
        model_url = "https://github.com/hhijry-coder/ER_predictor/raw/main/best_model%20(1).h5"
        scaler_url = "https://github.com/hhijry-coder/ER_predictor/raw/main/scaler%20(1).joblib"
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Download and save model
        model_path = os.path.join(temp_dir, 'best_model (1).h5')
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        
        # Download and save scaler
        scaler_path = os.path.join(temp_dir, 'scaler (1).joblib')
        response = requests.get(scaler_url)
        with open(scaler_path, 'wb') as f:
            f.write(response.content)
        
        # Load the files
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
            
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

def make_prediction(model, scaler, input_data):
    """Make predictions using the loaded model."""
    try:
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Reshape for LSTM
        input_reshaped = input_scaled.reshape((len(input_data), 1, input_scaled.shape[1]))
        
        # Make prediction
        predictions = model.predict(input_reshaped)
        
        return predictions.flatten()
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def process_batch_file(uploaded_file):
    """Process uploaded batch file (CSV or XLSX)."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:  # xlsx
            df = pd.read_excel(uploaded_file)
        
        required_columns = ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']
        
        # Check if all required columns are present
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
            
        return df[required_columns]
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_historical_plot(data, plot_type="line"):
    """Create various types of historical plots."""
    if plot_type == "line":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['hour'],
            y=data['waiting_time'],
            mode='lines+markers',
            name='Average Wait Time'
        ))
        fig.update_layout(
            title="Average Wait Times by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Wait Time (minutes)",
            hovermode='x unified'
        )
    elif plot_type == "heatmap":
        # Create hour vs day heatmap
        pivot_table = pd.pivot_table(
            data, 
            values='waiting_time', 
            index='dayOfWeek',
            columns='hour',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Wait Time Heatmap by Day and Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week"
        )
    
    return fig
