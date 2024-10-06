# pages/02_visualization.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from main import navigation

st.set_page_config(page_title="Data Visualization", page_icon="📈", layout="wide")

# Display navigation
navigation()

st.title("Data Visualization")

# Example visualization using dummy data
# In a real application, you would use actual historical data
hours = list(range(24))
avg_wait_times = [30 + 15 * np.sin(h/4) + np.random.normal(0, 5) for h in hours]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hours,
    y=avg_wait_times,
    mode='lines+markers',
    name='Average Wait Time'
))
fig.update_layout(
    title="Average Wait Times by Hour",
    xaxis_title="Hour of Day",
    yaxis_title="Wait Time (minutes)",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# Additional visualizations can be added here

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 ER Waiting Time Predictor v1.0 | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
