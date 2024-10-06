import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import io
import base64

# Load the best model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model('best_model (1).h5')
    scaler = joblib.load('scaler (1).joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

# Set page config
st.set_page_config(page_title="Waiting Time Prediction App", layout="wide")

# Title
st.title("Waiting Time Prediction App")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Create input fields for features
X3 = st.sidebar.number_input("X3", min_value=0.0, max_value=100.0, value=50.0)
hour = st.sidebar.number_input("Hour", min_value=0, max_value=23, value=12)
minutes = st.sidebar.number_input("Minutes", min_value=0, max_value=59, value=30)
waiting_people = st.sidebar.number_input("Waiting People", min_value=0, max_value=1000, value=50)
day_of_week = st.sidebar.number_input("Day of Week (0-6)", min_value=0, max_value=6, value=3)
service_time = st.sidebar.number_input("Service Time", min_value=0.0, max_value=1000.0, value=100.0)

# Create a dataframe from user inputs
input_data = pd.DataFrame({
    'X3': [X3],
    'hour': [hour],
    'minutes': [minutes],
    'waitingPeople': [waiting_people],
    'dayOfWeek': [day_of_week],
    'serviceTime': [service_time]
})

# Scale the input data
input_scaled = scaler.transform(input_data)

# Reshape input for LSTM (if the model expects 3D input)
input_reshaped = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

# Make prediction
prediction = model.predict(input_reshaped)

# Display prediction
st.header("Prediction Result")
st.write(f"Predicted Waiting Time: {prediction[0][0]:.2f} minutes")

# Visualization
st.header("Visualizations")

# Function to create download link for plot
def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Feature Importance Plot (using coefficients as a simple proxy for importance)
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance = np.abs(scaler.scale_)
feature_names = input_data.columns
sns.barplot(x=feature_importance, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Absolute Scaled Coefficient")
st.pyplot(fig)
st.markdown(get_image_download_link(fig, "feature_importance.png", "Download Feature Importance Plot"), unsafe_allow_html=True)

# Historical Data Plot (if available)
st.subheader("Historical Data Plot")
st.write("Note: This plot will be generated if historical data is available.")

# You can add code here to load and plot historical data if available

# Additional Information
st.header("Additional Information")
st.write("""
This app uses a machine learning model to predict waiting times based on various input parameters.
The model has been trained on historical data and uses features such as X3, hour, minutes, number of waiting people,
day of the week, and service time to make predictions.

Please note that the accuracy of predictions may vary, and this tool should be used as a general guide rather than
a definitive forecast.
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Your Name")
st.sidebar.write("© 2023 Your Company")
