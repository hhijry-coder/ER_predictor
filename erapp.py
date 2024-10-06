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

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('Combined_Data.csv')
    data['Arrival time'] = pd.to_datetime(data['Arrival time'], format='%m/%d/%Y %H:%M')
    data['timestamp'] = data['Arrival time']
    data = data.sort_values('timestamp')
    data = data.drop(['X1', 'X2'], axis=1)
    return data

data = load_data()

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model('best_model.h5')
    scaler = joblib.load('scaler.jobkit')
    return model, scaler

model, scaler = load_model_and_scaler()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Prediction", "Model Performance"])

if page == "Data Exploration":
    st.title("Data Exploration")

    # Display raw data
    st.subheader("Raw Data")
    st.write(data.head())

    # Data statistics
    st.subheader("Data Statistics")
    st.write(data.describe())

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr = data[['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime', 'waitingTime']].corr()
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

elif page == "Model Prediction":
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

elif page == "Model Performance":
    st.title("Model Performance")

    # Assuming we have access to test data and predictions
    # You might need to adjust this part based on how you've stored your test data and predictions
    X_test = data[['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']].iloc[-1000:]
    y_test = data['waitingTime'].iloc[-1000:]
    X_test_scaled = scaler.transform(X_test)
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    y_pred = model.predict(X_test_reshaped).flatten()

    # Model metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Metrics")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared Score: {r2:.2f}")

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

    # Residual Plot
    st.subheader("Residual Plot")
    residuals = y_test - y_pred
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

st.sidebar.info("This app predicts waiting times based on various features. Use the navigation menu to explore the data, make predictions, and view model performance.")
