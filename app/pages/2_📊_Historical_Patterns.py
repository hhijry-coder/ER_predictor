import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import create_historical_plot

st.title("📊 Historical Patterns")

# Sidebar for visualization options
st.sidebar.header("Visualization Options")
plot_type = st.sidebar.selectbox(
    "Select Plot Type",
    ["Line Plot", "Heatmap", "Box Plot"],
    key="plot_type"
)

# Generate sample historical data for demonstration
# In a real application, this would come from your database
dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
sample_data = pd.DataFrame({
    'timestamp': dates,
    'hour': dates.hour,
    'dayOfWeek': dates.dayofweek,
    'waiting_time': [30 + 15 * np.sin(h/4) + np.random.normal(0, 5) for h in range(len(dates))]
})

# Main content
st.markdown("""
Select different visualization types from the sidebar to explore patterns in ER waiting times.
You can hover over the plots for detailed information.
""")

# Create and display selected plot
if plot_type == "Line Plot":
    fig = create_historical_plot(sample_data, "line")
    st.plotly_chart(fig, use_container_width=True)
    
elif plot_type == "Heatmap":
    fig = create_historical_plot(sample_data, "heatmap")
    st.plotly_chart(fig, use_container_width=True)
    
elif plot_type == "Box Plot":
    fig = px.box(sample_data, x='hour', y='waiting_time', title="Wait Time Distribution by Hour")
    st.plotly_chart(fig, use_container_width=True)

# Additional analysis options
st.markdown("### Additional Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Summary Statistics")
    st.dataframe(sample_data['waiting_time'].describe())

with col2:
    st.markdown("#### Patterns by Day of Week")
    day_means = sample_data.groupby('dayOfWeek')['waiting_time'].mean()
    st.bar_chart(day_means)
