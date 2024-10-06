# pages/01_batch_upload.py

import streamlit as st
import pandas as pd
from erapp import load_prediction_resources, navigation

st.set_page_config(page_title="Batch Upload", page_icon="📊", layout="wide")

# Display navigation
navigation()

st.title("Batch Data Upload and Analysis")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
    
    if st.button("Process Batch Data"):
        model, scaler = load_prediction_resources()
        
        if model is not None and scaler is not None:
            # Here you would process the batch data using your model
            st.write("Processing batch data...")
            # Example (you'll need to adjust this based on your actual model and data structure):
            # processed_data = scaler.transform(df)
            # predictions = model.predict(processed_data)
            # df['Predicted_Wait_Time'] = predictions
            # st.write(df)
            
            st.success("Batch processing complete!")
        else:
            st.error("Failed to load model or scaler. Please check your model files.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 ER Waiting Time Predictor v1.0 | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
