import streamlit as st
import pandas as pd
import datetime
from utils.helpers import make_prediction

st.title("🔮 Make Predictions")

# Input form
with st.form("prediction_form"):
    st.subheader("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date and time input
        date_input = st.date_input("Select Date", datetime.datetime.now().date())
        time_input = st.time_input("Select Time", datetime.datetime.now().time())
        
        # Combine date and time
        arrival_time = datetime.datetime.combine(date_input, time_input)
        
        # Extract features
        hour = arrival_time.hour
        minutes = arrival_time.minute
        day_of_week = arrival_time.weekday()
    
    with col2:
        # Other inputs
        x3 = st.number_input("X3 Value", min_value=0.0, max_value=100.0, value=50.0)
        waiting_people = st.number_input("Number of Waiting People", min_value=0, max_value=100, value=10)
        service_time = st.number_input("Service Time (minutes)", min_value=0, max_value=180, value=30)
    
    submit_button = st.form_submit_button("Predict Waiting Time")

# Make prediction when form is submitted
if submit_button:
    if st.session_state.model is not None and st.session_state.scaler is not None:
        # Prepare input data
        input_data = pd.DataFrame([[x3, hour, minutes, waiting_people, day_of_week, service_time]], 
                                columns=['X3', 'hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'serviceTime'])
        
        # Make prediction
        prediction = make_prediction(st.session_state.model, st.session_state.scaler, input_data)
        
        if prediction is not None:
            # Display prediction
            st.markdown("""
            ### Prediction Results
            """)
            
            col1, col2, col3 = st.columns([1,2,1])
            
            with col2:
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3>Estimated Waiting Time</h3>
                    <h2 style='color: #1f77b4;'>{prediction[0]:.1f} minutes</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional context
            st.markdown("""
            #### Prediction Context:
            - This prediction takes into account current ER conditions and historical patterns
            - Actual waiting times may vary based on emergency cases and unforeseen circumstances
            """)
            
            # Confidence metrics
            st.markdown("#### Input Summary:")
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            
            with col_metrics1:
                st.metric("Current Load", f"{waiting_people} patients")
            with col_metrics2:
                st.metric("Time of Day", f"{hour:02d}:{minutes:02d}")
            with col_metrics3:
                st.metric("Service Time", f"{service_time} min")
