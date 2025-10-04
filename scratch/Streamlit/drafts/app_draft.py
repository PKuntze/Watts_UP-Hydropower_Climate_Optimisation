### Import libraries

import streamlit as st
import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys
from PIL import Image  # Pillow library

# LOAD Data: merged Table - ANN output + historical data --> @Bernd
# --- 1. Load dataset ---
data_path="data/Streamlit_Input.csv"
df= pd.read_csv(data_path)

# Ensure correct dtype
df["Date"] = pd.to_datetime(df["Date"])
df["Source"] = df["Source"].astype(str)  # safer for selectbox

### Section: Left Sidebar ###

# LOGO   --> @ Florencia
st.sidebar.image("images/logo_option_2.png", use_container_width=True)

# Select User ID (from list)    --> store this as userID 
selected_id = st.sidebar.selectbox("Select an ID:", df["Source"].unique())

# Select Date (from calendar)   --> stor this as selected_date
selected_date = st.sidebar.date_input(
    "Select a date", 
    value=df["Date"].min(),  # ensure default is also a date
    min_value=df["Date"].min(), 
    max_value=df["Date"].max()
)


# Define cutoff (as a date for easy comparison with selected_date)
cutoff_date = pd.to_datetime("2024-09-24")

#Decide historic vs forecast
is_forecast = pd.to_datetime(selected_date) >= cutoff_date

with st.sidebar.expander("ℹ️ About this app"):
    st.write("This tool helps the community forecast expected kWh consumption.")

# Weather of the day --> @Gozal


### Section: Main Page ###

# Let's give a title
st.title("⚡Watt’s Up, Kalam?⚡")

# Subtitle: "Kalam - Swat District of the Khyber Pakhtunkhwa province, Pakistan

# (Interactive) Show Pictures via sliders    --> @Noé 

#Visualize: Historical vs Prediction 




### Filter the df

filtered = df[
    (df["Date"] == pd.Timestamp(selected_date)) &
    (df["Source"] == selected_id)
]

# Display the prediction
if not filtered.empty:
    kwh_value = filtered["kwh"].values[0]

    if selected_date < cutoff_date:
        data_type = "Historic value"
        color = "#4CAF50"  # green
    else:
        data_type = "Forecast value"
        color = "#FF9800"  # orange

    # Nicely styled box
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center">
            <h2>{data_type}</h2>
            <h1>{kwh_value:.2f} kWh</h1>
            <p>for Source {selected_id} on {selected_date.date()}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("No data available for this combination.")

#Prediciton --> @Team

# Get prediction for selected day
highlight = filtered[filtered["Date"].dt.date == selected_date]

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered["Date"], filtered["kwh"], marker="o")

# Highlight selected point
if not highlight.empty:
    ax.scatter(highlight["Date"], highlight["kwh"], color="red", s=120, zorder=5)

ax.set_title(f"Energy prediction for {selected_id}")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted kWh")
ax.grid(True)
plt.xticks(rotation=45)

st.pyplot(fig)
