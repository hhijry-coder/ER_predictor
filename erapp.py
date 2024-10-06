import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tensorflow.keras.models import load_model
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load model and scaler
model = load_model('best_model (1).h5')
scaler = joblib.load('scaler (1).joblib')

# Set style for seaborn plots
sns.set_style("whitegrid")
plt.style.use("seaborn-darkgrid")

# Streamlit app
def main():
    st.title("Time Series Analysis and Prediction")

    # File uploader for CSV and XLSX
    uploaded_file = st.file_uploader("Upload Combined Data (CSV or XLSX)", type=["csv", "xlsx"])
    data = None

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)

        data['Arrival time'] = pd.to_datetime(data['Arrival time'], format='%m/%d/%Y %H:%M')
        data['timestamp'] = data['Arrival time']
        data = data.sort_values('timestamp')

        # Drop X1 and X2
        data = data.drop(['X1', 'X2'], axis=1)

    # Sidebar for online input
    st.sidebar.header("Manual Input")
    X3 = st.sidebar.number_input("X3", value=0.0)
    hour = st.sidebar.number_input("Hour", min_value=0, max_value=23, value=12)
    minutes = st.sidebar.number_input("Minutes", min_value=0, max_value=59, value=30)
    waitingPeople = st.sidebar.number_input("Waiting People", min_value=0, value=0)
    dayOfWeek = st.sidebar.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
    serviceTime = st.sidebar.number_input("Service Time", value=0.0)

    # If no data is uploaded, use manual input
    if data is None:
        data = pd.DataFrame({
            'X3': [X3],
            'hour': [hour],
            'minutes': [minutes],
            'waitingPeople': [waitingPeople],
            'dayOfWeek': [dayOfWeek],
            'serviceTime': [serviceTime],
            'waitingTime': [0]  # Placeholder for target
        })

    # Feature selection
    features = ['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime']
    target = 'waitingTime'

    # Preprocessing visualizations
    if st.checkbox("Show Preprocessing Visualizations"):
        plot_preprocessing_visualizations(data, features, target)

    # Train/Test split
    train_size = int(len(data) * 0.8)
    X = data[features]
    y = data[target]
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape input for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Predictions
    y_pred = model.predict(X_test_lstm).flatten()

    # Evaluate model
    evaluate_model(y_test, y_pred, 'Loaded Model')

    # Visualizations
    if st.checkbox("Show Learning Curves"):
        plot_learning_curves(history, 'Loaded Model')

    if st.checkbox("Show Actual vs Predicted Plots"):
        plot_actual_vs_predicted(y_test, y_pred, 'Loaded Model')

    if st.checkbox("Show Time Series Plot"):
        test_dates = data['timestamp'].iloc[-len(y_test):]
        plot_time_series(y_test, y_pred, test_dates, 'Loaded Model')

    if st.checkbox("Show Seasonal Decomposition"):
        ts = pd.Series(data['waitingTime'].values, index=data['timestamp'])
        plot_seasonal_decomposition(ts)

    if st.checkbox("Show Residual Analysis"):
        plot_residuals(y_test, y_pred, 'Loaded Model')

    if st.checkbox("Show Error Distribution"):
        plot_error_distribution(y_test, y_pred, 'Loaded Model')

def plot_preprocessing_visualizations(data, features, target):
    # Box plots
    st.subheader("Box Plots of Features and Target")
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(data=data[features + [target]], ax=ax)
    st.pyplot(fig)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = data[features + [target]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
    st.pyplot(fig)

    # Frequency distribution
    st.subheader("Frequency Distribution")
    n_plots = len(features) + 1
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features + [target]):
        sns.histplot(data[feature], kde=True, color='skyblue', edgecolor='black', ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')

    st.pyplot(fig)

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    st.write(f'{model_name} - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}')

def plot_learning_curves(history, model_name):
    st.subheader(f'{model_name} Learning Curve')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.history['loss'], label='Training Loss', color='blue')
    ax.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax.set_title(f'{model_name} Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    st.pyplot(fig)

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    st.subheader(f'{model_name}: Actual vs Predicted')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    ax.set_xlabel('Actual Waiting Time')
    ax.set_ylabel('Predicted Waiting Time')
    st.pyplot(fig)

def plot_time_series(actual, predicted, dates, model_name):
    st.subheader(f'Time Series of Actual vs Predicted Waiting Times - {model_name}')
    fig, ax = plt.subplots(figsize=(15, 8))
    window = 24
    actual_smooth = pd.Series(actual).rolling(window=window).mean()
    predicted_smooth = pd.Series(predicted).rolling(window=window).mean()
    ax.plot(dates, actual, label='Actual', alpha=0.3, color='blue')
    ax.plot(dates, predicted, label='Predicted', alpha=0.3, color='red')
    ax.plot(dates, actual_smooth, label='Actual (Smoothed)', linewidth=2, color='darkblue')
    ax.plot(dates, predicted_smooth, label='Predicted (Smoothed)', linewidth=2, color='darkred')
    ax.set_xlabel('Date')
    ax.set_ylabel('Waiting Time')
    ax.legend()
    st.pyplot(fig)

def plot_seasonal_decomposition(ts):
    st.subheader("Seasonal Decomposition")
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

def plot_residuals(y_true, y_pred, model_name):
    st.subheader(f'Residual Analysis - {model_name}')
    fig, ax = plt.subplots(figsize=(12, 8))
    residuals = y_true - y_pred
    sns.regplot(x=y_pred, y=residuals, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.axhline(y=0, color='r', linestyle='--')
    st.pyplot(fig)

def plot_error_distribution(y_true, y_pred, model_name):
    st.subheader(f'{model_name} Error Distribution')
    fig, ax = plt.subplots(figsize=(12, 8))
    errors = y_true - y_pred
    sns.histplot(errors, kde=True, color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
