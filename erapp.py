import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tensorflow.keras.models import load_model
import joblib
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Waiting Time Predictor", layout="wide")


def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

@st.cache_resource
def load_model_and_scaler():
    # Define custom objects dictionary with 'mse' function
    custom_objects = {'mse': mse}
    
    # Load the model with custom objects
    model = load_model('best_model (1).h5', custom_objects=custom_objects)
    scaler = joblib.load('scaler (1).joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

@st.cache_data
def load_and_preprocess_data(file):
    if file.name.endswith('.xlsx'):
        data = pd.read_excel(file)
    elif file.name.endswith('.csv'):
        data = pd.read_csv(file)
    else:
        st.error("Unsupported file format. Please upload an XLSX or CSV file.")
        return None

    # Preprocess the data
    if 'Arrival time' in data.columns:
        data['Arrival time'] = pd.to_datetime(data['Arrival time'])
        data['timestamp'] = data['Arrival time']
    elif 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    else:
        st.error("The dataset must contain either 'Arrival time' or 'timestamp' column.")
        return None

    data = data.sort_values('timestamp')
    
    # Remove 'X1' and 'X2' if they exist
    data = data.drop(['X1', 'X2'], axis=1, errors='ignore')
    
    return data


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model('best_model (1).h5')
    scaler = joblib.load('scaler (1).joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Exploration", "Model Prediction", "Model Performance"])

# Data Upload
if page == "Data Upload":
    st.title("Data Upload")
    uploaded_file = st.file_uploader("Choose an XLSX or CSV file", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        data = load_and_preprocess_data(uploaded_file)
        if data is not None:
            st.session_state['data'] = data
            st.success("Data successfully loaded and preprocessed!")
            st.write(data.head())
    else:
        st.warning("Please upload a file to proceed.")

# Data Exploration
elif page == "Data Exploration":
    if 'data' not in st.session_state:
        st.warning("Please upload data in the 'Data Upload' page first.")
    else:
        data = st.session_state['data']
        st.title("Data Exploration")

        # Display raw data
        st.subheader("Raw Data")
        st.write(data.head())

        # Data statistics
        st.subheader("Data Statistics")
        st.write(data.describe())

        # Correlation matrix
        st.subheader("Correlation Matrix")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Time series plot
        st.subheader("Waiting Time Over Time")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(data['timestamp'], data['waitingTime'])
        ax.set_xlabel("Date")
        ax.set_ylabel("Waiting Time")
        st.pyplot(fig)

        # Seasonal Decomposition
        st.subheader("Seasonal Decomposition")
        ts = pd.Series(data['waitingTime'].values, index=data['timestamp'])
        result = seasonal_decompose(ts, model='additive', period=24)
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        result.observed.plot(ax=axes[0])
        axes[0].set_title('Observed')
        result.trend.plot(ax=axes[1])
        axes[1].set_title('Trend')
        result.seasonal.plot(ax=axes[2])
        axes[2].set_title('Seasonal')
        result.resid.plot(ax=axes[3])
        axes[3].set_title('Residual')
        st.pyplot(fig)

# Model Prediction
elif page == "Model Prediction":
    if 'data' not in st.session_state:
        st.warning("Please upload data in the 'Data Upload' page first.")
    else:
        st.title("Waiting Time Prediction")

        # User input for prediction
        st.subheader("Enter Values for Prediction")
        x3 = st.number_input("X3", min_value=0.0, max_value=1.0, value=0.5)
        hour = st.number_input("Hour", min_value=0, max_value=23, value=12)
        minutes = st.number_input("Minutes", min_value=0, max_value=59, value=30)
        waiting_people = st.number_input("Waiting People", min_value=0, value=10)
        day_of_week = st.number_input("Day of Week (0-6)", min_value=0, max_value=6, value=3)
        service_time = st.number_input("Service Time", min_value=0, value=5)

        # Make prediction
        if st.button("Predict Waiting Time"):
            input_data = np.array([[x3, hour, minutes, waiting_people, day_of_week, service_time]])
            input_scaled = scaler.transform(input_data)
            input_reshaped = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
            prediction = model.predict(input_reshaped)
            st.success(f"Predicted Waiting Time: {prediction[0][0]:.2f} minutes")

        # Batch prediction
        st.subheader("Batch Prediction")
        st.write("Use your uploaded data for batch prediction")
        if st.button("Run Batch Prediction"):
            data = st.session_state['data']
            X = data[['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']]
            X_scaled = scaler.transform(X)
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            batch_predictions = model.predict(X_reshaped).flatten()
            
            # Add predictions to the dataframe
            data['predicted_waitingTime'] = batch_predictions
            
            # Display results
            st.write(data[['timestamp', 'waitingTime', 'predicted_waitingTime']].head(20))
            
            # Download link for predictions
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

# Model Performance
elif page == "Model Performance":
    if 'data' not in st.session_state:
        st.warning("Please upload data in the 'Data Upload' page first.")
    else:
        data = st.session_state['data']
        st.title("Model Performance")

        # Use the entire dataset for evaluation
        X = data[['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']]
        y = data['waitingTime']
        X_scaled = scaler.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        y_pred = model.predict(X_reshaped).flatten()

        # Model metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.subheader("Model Metrics")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared Score: {r2:.2f}")

        # Actual vs Predicted Plot
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        # Residual Plot
        st.subheader("Residual Plot")
        residuals = y - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)

        # Error Distribution
        st.subheader("Error Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_xlabel("Error")
        st.pyplot(fig)

st.sidebar.info("This app predicts waiting times based on various features. Use the navigation menu to upload data, explore it, make predictions, and view model performance.")
