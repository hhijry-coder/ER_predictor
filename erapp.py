import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error as sklearn_mse
from sklearn.model_selection import learning_curve
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
        if os.path.exists('best_model (1).h5'):
            custom_objects = {'mse': mse}
            model = load_model('best_model (1).h5', custom_objects=custom_objects)
        else:
            st.error("Model file 'best_model (1).h55' not found.")
            return None, None

        if os.path.exists('scaler (1).joblib'):
            scaler = joblib.load('scaler (1).joblib')
        else:
            st.error("Scaler file 'scaler (1).joblib' not found.")
            return None, None

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

# Load the model and scaler once
model, scaler = load_model_and_scaler()

def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    """Function to plot actual vs predicted values."""
    df = pd.DataFrame({'Sample': range(len(y_pred)), 'Predicted': y_pred})
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['Sample'], y=df['Predicted'], mode='lines', name='Predicted'))
    
    if y_true is not None:
        df['Actual'] = y_true
        fig.add_trace(go.Scatter(x=df['Sample'], y=df['Actual'], mode='lines', name='Actual'))
    
    fig.update_layout(title=title, xaxis_title="Sample Index", yaxis_title="Waiting Time")
    return fig

def plot_learning_curve(X, y, model, title="Learning Curve"):
    """Function to plot the learning curve."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers',
                             name='Training score', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean, mode='lines+markers',
                             name='Cross-validation score', line=dict(color='red')))
    fig.update_layout(title=title, xaxis_title="Training examples", yaxis_title="Score")
    return fig

def plot_error_histogram(y_true, y_pred, title="Prediction Error Histogram"):
    """Function to plot the prediction error histogram."""
    errors = y_pred - y_true
    fig = px.histogram(errors, nbins=30, title=title)
    fig.update_layout(xaxis_title="Prediction Error", yaxis_title="Frequency")
    return fig

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

                # Check if 'waitingTime' column exists for actual values
                if 'waitingTime' in data.columns:
                    y_true = data['waitingTime'].values
                    X = data.drop(columns=['waitingTime'])
                else:
                    y_true = None
                    X = data
                    st.warning("No 'waitingTime' column found in the uploaded data. Only predictions will be shown.")

                # Scale the features
                try:
                    X_scaled = scaler.transform(X)
                except ValueError as e:
                    st.error(f"Scaling Error: {str(e)}")
                    st.stop()

                # Reshape data to match the model's expected input shape (batch_size, 1, num_features)
                X_scaled_reshaped = np.expand_dims(X_scaled, axis=1)

                # Display progress bar while making predictions
                with st.spinner('Making predictions...'):
                    y_pred = model.predict(X_scaled_reshaped).flatten()

                # Visualize predictions
                st.write("Predicted Waiting Times:")
                st.dataframe(pd.DataFrame({"Predicted waitingTime": y_pred}))

                # Plot the predictions
                if y_true is not None:
                    fig_predictions = plot_predictions(y_true, y_pred, title="Predicted vs Actual Waiting Times (Batch)")
                    st.plotly_chart(fig_predictions)

                    # Calculate and display metrics
                    mae = mean_absolute_error(y_true, y_pred)
                    mse = sklearn_mse(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_true, y_pred)

                    st.write("Model Performance Metrics:")
                    st.write(f"Mean Absolute Error: {mae:.2f}")
                    st.write(f"Mean Squared Error: {mse:.2f}")
                    st.write(f"Root Mean Squared Error: {rmse:.2f}")
                    st.write(f"R-squared Score: {r2:.2f}")

                    # Plot learning curve
                    st.write("Learning Curve:")
                    fig_learning_curve = plot_learning_curve(X_scaled, y_true, model)
                    st.plotly_chart(fig_learning_curve)

                    # Plot error histogram
                    st.write("Prediction Error Histogram:")
                    fig_error_hist = plot_error_histogram(y_true, y_pred)
                    st.plotly_chart(fig_error_hist)
                else:
                    fig_predictions = plot_predictions(None, y_pred, title="Predicted Waiting Times (Batch)")
                    st.plotly_chart(fig_predictions)

                # Allow users to download predictions
                st.download_button(
                    label="Download Predictions",
                    data=pd.DataFrame({"Predicted waitingTime": y_pred}).to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )

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

        # Reshape data to match the model's expected input shape (batch_size, 1, num_features)
        X_scaled = np.expand_dims(X_scaled, axis=1)

        # Display progress bar while making predictions
        with st.spinner('Making prediction...'):
            y_pred = model.predict(X_scaled).flatten()

        # Display the prediction
        st.write(f"Predicted Waiting Time: {y_pred[0]:.2f} minutes")

        # Plot the prediction (even for a single point)
        fig = plot_predictions(None, y_pred, title="Manual Input Prediction")
        st.plotly_chart(fig)
