import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

import matplotlib.pyplot as plt
import seaborn as sns

# Set style for seaborn plots
sns.set_style("whitegrid")
plt.style.use("seaborn-dark")

# Load the saved model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model('best_model.keras')  # Load the best LSTM model
    with open('scaler_model.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)  # Load the saved scaler
    return model, scaler

# Function to preprocess new data
def preprocess_data(data, features, scaler):
    X_new = data[features]
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))  # Reshape for LSTM
    return X_new_scaled

# Function to make predictions
def make_predictions(model, X_new_scaled):
    return model.predict(X_new_scaled).flatten()

# Function to display fancy predicted waiting time
def display_fancy_prediction(predicted_time):
    st.markdown(
        f"""
        <div style="background-color:#4CAF50;padding:20px;border-radius:10px">
        <h2 style="color:white;text-align:center;">Predicted Waiting Time: {predicted_time:.2f} minutes</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to plot actual vs predicted results
def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(12, 8))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.xlabel('Actual Waiting Time', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Waiting Time', fontsize=14, fontweight='bold')
    plt.title(f'{model_name}: Actual vs Predicted', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

# Function to display descriptive statistics
def descriptive_stats(data):
    st.write("### Descriptive Statistics")
    st.write(data.describe())

# Visualization for Boxplot
def plot_boxplot(data, features, target):
    st.write("### Box Plot of Features and Target")
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=data[features + [target]])
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# Visualization for Frequency Distribution
def plot_histograms(data, features, target):
    st.write("### Frequency Distribution of Features and Target")
    n_plots = len(features) + 1
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features + [target]):
        sns.histplot(data[feature], kde=True, color='skyblue', edgecolor='black', ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}', fontsize=14)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    st.pyplot(fig)

# Load the model and scaler
model, scaler = load_model_and_scaler()

# Streamlit app title
st.title("Time Series Waiting Time Prediction App")

# Sidebar for file upload or manual entry
st.sidebar.header("Data Input Options")
data_source = st.sidebar.radio("Choose the data source", ("Upload CSV/XLSX File", "Manual Input"))

# Define the feature columns
features = ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']
target = 'waitingTime'

if data_source == "Upload CSV/XLSX File":
    # File upload section
    uploaded_file = st.sidebar.file_uploader("Upload your CSV or XLSX file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the uploaded file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Show a preview of the uploaded data
        st.write("### Uploaded Data Preview")
        st.write(data.head())

        # Preprocess and make predictions
        try:
            X_new_scaled = preprocess_data(data, features, scaler)

            # Make predictions
            predictions = make_predictions(model, X_new_scaled)

            # Display predictions
            st.write("### Predicted Waiting Times")
            data['Predicted_WaitingTime'] = predictions
            st.write(data[['Predicted_WaitingTime']])

            # Display fancy prediction card for the first prediction (or the average)
            display_fancy_prediction(predictions[0])  # You can adjust which prediction to display prominently

        except Exception as e:
            st.error(f"Error processing the file: {e}")

elif data_source == "Manual Input":
    st.sidebar.subheader("Input values manually")
    
    # Manual input for each feature
    X3 = st.sidebar.number_input("X3", min_value=0.0, value=10.0)
    hour = st.sidebar.slider("Hour", 0, 23, 12)
    minutes = st.sidebar.slider("Minutes", 0, 59, 30)
    waitingPeople = st.sidebar.number_input("Waiting People", min_value=0, value=5)
    dayOfWeek = st.sidebar.selectbox("Day of the Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][x])
    serviceTime = st.sidebar.number_input("Service Time", min_value=0.0, value=20.0)

    # Create DataFrame for manual input
    user_data = pd.DataFrame({
        'X3': [X3],
        'hour': [hour],
        'minutes': [minutes],
        'waitingPeople': [waitingPeople],
        'dayOfWeek': [dayOfWeek],
        'serviceTime': [serviceTime]
    })

    st.write("### Your Input Data")
    st.write(user_data)

    # Preprocess and predict
    X_new_scaled = preprocess_data(user_data, features, scaler)
    predictions = make_predictions(model, X_new_scaled)

    # Display fancy prediction card
    display_fancy_prediction(predictions[0])

# Sidebar Navigation for Visualizations
st.sidebar.header("Visualization Options")
visualization = st.sidebar.selectbox("Choose a Visualization", ("Descriptive Analysis", "Box Plot", "Frequency Distribution", "Actual vs Predicted"))

if uploaded_file is not None:
    # Visualization section
    if visualization == "Descriptive Analysis":
        descriptive_stats(data)
    elif visualization == "Box Plot":
        plot_boxplot(data, features, target)
    elif visualization == "Frequency Distribution":
        plot_histograms(data, features, target)
    elif visualization == "Actual vs Predicted" and 'waitingTime' in data.columns:
        plot_actual_vs_predicted(data['waitingTime'], data['Predicted_WaitingTime'], "Best Model")
    else:
        st.warning("Actual vs Predicted plot requires the 'waitingTime' column in the uploaded file.")
