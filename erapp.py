import os
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import streamlit as st

# Disable GPU usage to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set page config
st.set_page_config(
    page_title="HajjCare Flow Optimizer",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container {
        background: #fafafa;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    div[data-testid="stSidebarNav"] {
        background-image: linear-gradient(#f0f2f6,#f0f2f6);
        padding: 1rem;
        border-radius: 10px;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model('best_model.keras', compile=False)  # Disable optimizer warnings
        with open('scaler_model.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model and scaler: {str(e)}")
        return None, None

def preprocess_data(data, features, scaler):
    X_new = data[features]
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))
    return X_new_scaled

def make_predictions(model, X_new_scaled):
    try:
        return model.predict(X_new_scaled).flatten()
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def display_fancy_prediction(predicted_time):
    st.markdown(
        f"""
        <div style="background-color:#4CAF50;padding:20px;border-radius:10px">
        <h2 style="color:white;text-align:center;">Predicted Waiting Time: {predicted_time:.2f} minutes</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to plot the Actual vs Predicted with best-fit line
def plot_actual_vs_predicted(data, predictions):
    actual = data['waitingTime']
    
    # Scatter plot of actual vs predicted
    fig = go.Figure(go.Scatter(x=actual, y=predictions, mode='markers', name='Predicted vs Actual'))
    
    # Add best-fit line
    model = LinearRegression()
    model.fit(actual.values.reshape(-1, 1), predictions)
    best_fit = model.predict(actual.values.reshape(-1, 1))
    fig.add_trace(go.Scatter(x=actual, y=best_fit, mode='lines', name='Best Fit Line'))
    
    fig.update_layout(title="Predicted vs Actual Waiting Time with Best Fit Line", 
                      xaxis_title="Actual Waiting Time", 
                      yaxis_title="Predicted Waiting Time")
    st.plotly_chart(fig)

# Function to show box plot for all variables
def plot_box_all_variables(data, features):
    melted_data = pd.melt(data, value_vars=features, var_name='Variables', value_name='Values')
    fig = px.box(melted_data, x='Variables', y='Values', title="Box Plot of All Variables")
    st.plotly_chart(fig)

# Function to show histogram for all variables
def plot_histogram_all_variables(data, features):
    melted_data = pd.melt(data, value_vars=features, var_name='Variables', value_name='Values')
    fig = px.histogram(melted_data, x='Values', color='Variables', facet_col='Variables',
                       title="Histogram Frequency Distribution of All Variables")
    st.plotly_chart(fig)

def main():
    st.sidebar.title("HajjCare Flow Optimizer")

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.error("Failed to load model and scaler. Please check the files.")
        return

    # Features list
    features = ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']

    # Sidebar options for navigation
    page = st.sidebar.radio("Go to", ["Input Data", "Actual vs Predicted", "Box Plot", "Frequency Histogram"])

    if page == "Input Data":
        st.title("🏥 Input Data for Prediction")
        st.write("Predict hospital waiting times and explore nearby hospitals")

        # Sidebar input options
        data_source = st.sidebar.radio("Choose the data source", ("Manual Input", "Upload CSV/XLSX File"))

        if data_source == "Manual Input":
            st.sidebar.subheader("Input values manually")
            total_time = st.sidebar.number_input("Total Time (Waiting time + Service Time)", min_value=0.0, value=10.0)
            hour = st.sidebar.slider("Hour", 0, 23, 12)
            minutes = st.sidebar.slider("Minutes", 0, 59, 30)
            waitingPeople = st.sidebar.number_input("Waiting People", min_value=0, value=5)
            dayOfWeek = st.sidebar.selectbox(
                "Day of the Week", 
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][x]
            )
            serviceTime = st.sidebar.number_input("Service Time", min_value=0.0, value=20.0)

            user_data = pd.DataFrame({
                'X3': [total_time],
                'hour': [hour],
                'minutes': [minutes],
                'waitingPeople': [waitingPeople],
                'dayOfWeek': [dayOfWeek],
                'serviceTime': [serviceTime]
            })

            st.write("### Your Input Data")
            st.write(user_data)

            if st.button("Predict Waiting Time"):
                X_new_scaled = preprocess_data(user_data, features, scaler)
                predictions = make_predictions(model, X_new_scaled)
                if predictions is not None:
                    display_fancy_prediction(predictions[0])

        else:
            uploaded_file = st.sidebar.file_uploader("Upload your CSV or XLSX file", type=["csv", "xlsx"])

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)

                    st.write("### Uploaded Data Preview")
                    st.write(data.head())

                    if all(col in data.columns for col in features):
                        X_new_scaled = preprocess_data(data, features, scaler)
                        predictions = make_predictions(model, X_new_scaled)
                        if predictions is not None:
                            data['Predicted_WaitingTime'] = predictions
                            st.write("### Predictions")
                            st.write(data[['Predicted_WaitingTime']])
                            display_fancy_prediction(predictions[0])
                    else:
                        st.error("Uploaded data does not contain required features.")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    elif page == "Actual vs Predicted":
        st.title("📊 Actual vs Predicted Waiting Time")
        if 'Predicted_WaitingTime' in locals():
            plot_actual_vs_predicted(data, predictions)
        else:
            st.error("No data available for prediction. Please input data on the 'Input Data' page.")

    elif page == "Box Plot":
        st.title("📦 Box Plot for All Variables")
        if 'Predicted_WaitingTime' in locals():
            plot_box_all_variables(data, features)
        else:
            st.error("No data available for visualization. Please input data on the 'Input Data' page.")

    elif page == "Frequency Histogram":
        st.title("📊 Frequency Histogram for All Variables")
        if 'Predicted_WaitingTime' in locals():
            plot_histogram_all_variables(data, features)
        else:
            st.error("No data available for visualization. Please input data on the 'Input Data' page.")

if __name__ == "__main__":
    main()
