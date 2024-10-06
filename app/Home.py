import streamlit as st
from utils.helpers import load_prediction_resources

st.set_page_config(
    page_title="ER Waiting Time Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTitle {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .stSubheader {
        font-size: 1.5rem;
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for model and scaler
if 'model' not in st.session_state or 'scaler' not in st.session_state:
    st.session_state.model, st.session_state.scaler = load_prediction_resources()

# Main content
st.title("🏥 ER Waiting Time Predictor")
st.markdown("---")

# Welcome message
st.markdown("""
## Welcome to the ER Waiting Time Predictor

This application helps predict Emergency Room waiting times using advanced machine learning techniques. 
Choose from the following features in the navigation bar:

### 🔮 Predictions
- Make real-time predictions for individual cases
- Input current conditions and get estimated waiting times

### 📊 Historical Patterns
- View historical waiting time patterns
- Analyze trends with interactive visualizations
- Explore correlations between different factors

### 📑 Batch Predictions
- Upload CSV or Excel files for batch predictions
- Get predictions for multiple cases at once
- Download results for further analysis

### Getting Started
1. Select your desired feature from the navigation bar on the left
2. Follow the instructions on each page
3. Explore different visualizations and analysis options

""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 ER Waiting Time Predictor v1.1 | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
