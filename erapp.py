import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from io import BytesIO
import base64
import folium
from streamlit_folium import st_folium
import requests
import os
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from requests.exceptions import RequestException

# Sidebar for Page Navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Select a page:", ["Data Upload", "Model Training", "Evaluation & Visualization", "Interactive Map"])

# Global variables for storing loaded data
uploaded_data = None
X_train, X_test, y_train, y_test = None, None, None, None
preprocessor = None
scaler = None
model = None

# Paths for model and scaler (use appropriate path for Streamlit Cloud)
preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
model_path = os.path.join(os.path.dirname(__file__), 'best_model.h5')

# Function to read CSV or XLSX
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

# Function to preprocess data
def preprocess_data(X):
    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    date_features = X.select_dtypes(include=['datetime64']).columns

    # If there are no datetime columns, convert potential string date columns
    if len(date_features) == 0:
        for col in categorical_features:
            try:
                X[col] = pd.to_datetime(X[col])
                date_features = date_features.append(col)
                categorical_features = categorical_features.drop(col)
            except:
                pass

    # Create preprocessing steps for each column type
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    date_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='1900-01-01')),
        ('date_to_num', FunctionTransformer(lambda x: x.astype(int) // 10**9))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('date', date_transformer, date_features)
        ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Create a new dataframe with processed data
    feature_names = (numeric_features.tolist() + 
                     preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_features).tolist() + 
                     date_features.tolist())
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    return X_processed_df, preprocessor

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
    # Get location (lat/lon) from city name
    location = geocode_with_retry(city_name)
    
    if location is None:
        return None
    
    lat, lon = location

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
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Preprocess the data
            X_train_processed, preprocessor = preprocess_data(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Apply StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_processed)
            X_test_scaled = scaler.transform(X_test_processed)

            # Save the preprocessor and scaler
            joblib.dump(preprocessor, preprocessor_path)
            joblib.dump(scaler, scaler_path)

            st.success("Data processed and ready for training!")

# Page 2: Model Training
elif pages == "Model Training":
    st.title("Train Models")
    
    if uploaded_data is not None and X_train is not None:
        model_type = st.sidebar.selectbox("Select Model", ["Traditional LSTM", "Advanced LSTM", "DNN"])

        # Display current model name
        st.write(f"Training the **{model_type}** model on your data.")
        
        # Train button
        if st.button("Train"):
            # Load preprocessor and scaler
            preprocessor = joblib.load(preprocessor_path)
            scaler = joblib.load(scaler_path)

            # Preprocess and scale the data
            X_train_processed = preprocessor.transform(X_train)
            X_train_scaled = scaler.transform(X_train_processed)

            # Implement model training logic based on selected model
            # For this example, we'll use a simple Dense Neural Network
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense

            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

            # Save the model
            model.save(model_path)

            st.success(f"Training of {model_type} model complete!")
    else:
        st.warning("Please upload data first on the 'Data Upload' page.")

# Page 3: Evaluation & Visualization
elif pages == "Evaluation & Visualization":
    st.title("Model Evaluation & Visualization")

    if X_test is not None and scaler is not None:
        st.subheader("Model Performance")

        # Load the preprocessor, scaler, and model
        try:
            preprocessor = joblib.load(preprocessor_path)
            scaler = joblib.load(scaler_path)
            model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        except Exception as e:
            st.error(f"Error loading model components: {e}")
            st.stop()

        # Preprocess and scale the test data
        X_test_processed = preprocessor.transform(X_test)
        X_test_scaled = scaler.transform(X_test_processed)

        # Predict with loaded model
        y_pred = model.predict(X_test_scaled)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**MSE**: {mse:.2f}")
        st.write(f"**R2**: {r2:.2f}")

        # Plot Actual vs Predicted
        fig = px.scatter(x=y_test, y=y_pred.flatten(), labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
        st.plotly_chart(fig)

        # Plot Residuals
        residuals = y_test - y_pred.flatten()
        fig_res = px.scatter(x=y_pred.flatten(), y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, title="Residuals")
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
        with st.spinner("Searching for hospitals..."):
            folium_map = create_map(city_name, radius)
        
        if folium_map:
            st.write(f"Showing hospitals near **{city_name}** within **{radius}** meters.")
            st_folium(folium_map, width=700, height=500)
