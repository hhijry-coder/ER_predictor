import streamlit as st
import pandas as pd
from utils.helpers import process_batch_file, make_prediction
import io

st.title("📑 Batch Predictions")

st.markdown("""
Upload a CSV or Excel file containing multiple cases for prediction. 
The file should contain the following columns:
- X3
- hour
- minutes
- waitingPeople
- dayOfWeek
- serviceTime
""")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Process the uploaded file
    input_data = process_batch_file(uploaded_file)
    
    if input_data is not None:
        st.success("File uploaded successfully!")
        
        # Show preview of the data
        st.subheader("Data Preview")
        st.dataframe(input_data.head())
        
        # Make predictions button
        if st.button("Generate Predictions"):
            if st.session_state.model is not None and st.session_state.scaler is not None:
                # Make predictions
                predictions = make_prediction(st.session_state.model, st.session_state.scaler, input_data)
                
                if predictions is not None:
                    # Add predictions to dataframe
                    results_df = input_data.copy()
                    results_df['predicted_waiting_time'] = predictions
                    
                    # Show results
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)
                    
                    # Download button
                    buffer = io.BytesIO()
                    results_df.to_csv(buffer, index=False)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="Download Predictions CSV",
                        data=buffer,
                        file_name="er_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Wait Time", 
                                f"{results_df['predicted_waiting_time'].mean():.1f} min")
                    with col2:
                        st.metric("Maximum Wait Time", 
                                f"{results_df['predicted_waiting_time'].max():.1f} min")
                    with col3:
                        st.metric("Minimum Wait Time", 
                                f"{results_df['predicted_waiting_time'].min():.1f} min")
                    
                    # Distribution plot
                    st.subheader("Distribution of Predicted Wait Times")
                    fig = px.histogram(results_df, x='predicted_waiting_time', 
                                     nbins=30, title='Distribution of Predicted Wait Times')
                    st.plotly_chart(fig, use_container_width=True)

# Display sample template
st.markdown("---")
st.subheader("Download Template")
st.markdown("""
If you need a template for your data, you can download one below:
""")

# Create sample template
sample_data = pd.DataFrame({
    'X3': [50.0],
    'hour': [14],
    'minutes': [30],
    'waitingPeople': [10],
    'dayOfWeek': [1],
    'serviceTime': [30]
})

# Download template button
buffer = io.BytesIO()
sample_data.to_csv(buffer, index=False)
buffer.seek(0)

st.download_button(
    label="Download Template CSV",
    data=buffer,
    file_name="er_prediction_template.csv",
    mime="text/csv"
)
