import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error as sklearn_mse
import joblib
import os
import tensorflow as tf
from keras.saving import register_keras_serializable

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

# Load the model and scaler once
model, scaler = load_model_and_scaler()

# Streamlit app
st.title('Waiting Time Prediction App')

if model is None or scaler is None:
    st.error("Unable to load the model or scaler.")
else:
    # Sidebar feature selection
    st.sidebar.header("Select Features for Prediction")
    
    # Allow users to select features they want to include in the model
    available_features = ['X3', 'hour', 'minutes', 'serviceTime', 'waitingPeople']
    selected_features = st.sidebar.multiselect("Choose features for prediction", available_features, default=available_features)

    # Upload file or input values for online prediction
    input_mode = st.sidebar.selectbox("Input Mode", ["Batch Upload", "Manual Input"])

    if input_mode == "Batch Upload":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

        if uploaded_file is not None:
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

            # Load data
            data = load_data(uploaded_file)
            if data is not None:
                st.write("Data Preview:")
                st.dataframe(data.head())

                # Preprocess the data
                def preprocess_data(data):
                    # Drop unnecessary features
                    data = data.drop(columns=['X1', 'X2'], errors='ignore')
                    
                    # Retrieve original feature names used during model training
                    original_feature_names = scaler.feature_names_in_
                    
                    # Ensure selected features are present
                    missing_features = set(original_feature_names) - set(data.columns)
                    for feature in missing_features:
                        data[feature] = 0  # Add missing features with default values

                    # Keep only features that were part of the original model
                    data = data[original_feature_names]

                    return data

                # Preprocess the data
                data = preprocess_data(data)

                # Scale the features
                try:
                    X_scaled = scaler.transform(data)
                except ValueError as e:
                    st.error(f"Scaling Error: {str(e)}")
                    st.stop()

                # Make predictions
                y_pred = model.predict(X_scaled).flatten()

                # Display predictions
                st.write("Predicted Waiting Times:")
                st.dataframe(pd.DataFrame({"Predicted WaitingTime": y_pred}))

    elif input_mode == "Manual Input":
        st.write("Manual Input for Online Prediction")
        
        # Collect feature values from user input
        input_data = {}
        for feature in selected_features:
            value = st.number_input(f"Enter value for {feature}", value=0)
            input_data[feature] = value
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Retrieve original feature names used during model training
        original_feature_names = scaler.feature_names_in_

        # Ensure all features are aligned with those expected by the scaler
        for feature in original_feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Add missing features with default values

        input_df = input_df[original_feature_names]  # Reorder the columns to match the training data

        # Scale the input features
        try:
            X_scaled = scaler.transform(input_df)
        except ValueError as e:
            st.error(f"Scaling Error: {str(e)}")
            st.stop()

        # Make predictions
        y_pred = model.predict(X_scaled).flatten()

        # Display the prediction
        st.write(f"Predicted Waiting Time: {y_pred[0]:.2f} minutes")
