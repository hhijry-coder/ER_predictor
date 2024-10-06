import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
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

    def plot_actual_vs_predicted(y_true, y_pred):
        """Scatter plot of actual vs predicted values with best fit line."""
        # Scatter plot
        fig = px.scatter(x=y_pred, y=y_true, labels={'x': 'Predicted', 'y': 'Actual'}, title="Actual vs Predicted Scatter Plot")

        # Fit a linear regression model for the best fit line
        reg = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
        best_fit_line = reg.predict(y_pred.reshape(-1, 1))

        # Add best fit line to the scatter plot
        fig.add_trace(go.Scatter(x=y_pred, y=best_fit_line, mode='lines', name='Best Fit Line', line=dict(color='red')))
        return fig

    def plot_learning_curve(train_sizes, train_scores, test_scores):
        """Plot the learning curve (train and test scores over training size)."""
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Training Score', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=train_sizes, y=test_mean, mode='lines+markers', name='Validation Score', line=dict(color='orange')))
        fig.update_layout(title="Learning Curve", xaxis_title="Training Examples", yaxis_title="Score")
        return fig

    def plot_histogram(y_pred):
        """Plot histogram of predicted values."""
        fig = px.histogram(y_pred, title="Prediction Histogram")
        return fig

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

                # Reshape data to match the model's expected input shape (batch_size, 1, num_features)
                X_scaled = np.expand_dims(X_scaled, axis=1)

                # Display progress bar while making predictions
                with st.spinner('Making predictions...'):
                    y_pred = model.predict(X_scaled).flatten()

                # Assume that 'actual' values are available for plotting actual vs predicted
                # You can replace this with the correct actual values
                y_actual = np.random.rand(len(y_pred)) * 100  # Replace with actual target data

                # Plot actual vs predicted values with best fit line
                fig_actual_vs_pred = plot_actual_vs_predicted(y_actual, y_pred)
                st.plotly_chart(fig_actual_vs_pred)

                # Plot prediction histogram
                fig_histogram = plot_histogram(y_pred)
                st.plotly_chart(fig_histogram)

                # Learning curve (dummy implementation as example, replace with actual learning curve)
                train_sizes = np.linspace(1, len(X_scaled), 10, dtype=int)
                train_scores, test_scores = learning_curve(model, X_scaled, y_actual, train_sizes=train_sizes, cv=3)
                fig_learning_curve = plot_learning_curve(train_sizes, train_scores, test_scores)
                st.plotly_chart(fig_learning_curve)

                # Allow users to download predictions
                st.download_button(
                    label="Download Predictions",
                    data=pd.DataFrame({"Predicted WaitingTime": y_pred}).to_csv(index=False),
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

        # Plot histogram for manual input prediction (though it's just one prediction)
        fig_histogram = plot_histogram(y_pred)
        st.plotly_chart(fig_histogram)
