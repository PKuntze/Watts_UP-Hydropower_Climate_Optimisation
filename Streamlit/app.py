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
import os

# Streamlit theme configuration should be set in .streamlit/config.toml, not in Python code.

# LOAD Data: merged Table - ANN output + historical data --> @Bernd


### Section: Left Sidebar

# LOGO   --> @ Florencia

# Select User ID (from list)    --> store this as userID 
# Select Date (from calendar)   --> stor this as selected_date

# Weather of the day --> @Gozal
# ------------------------------
# LOAD DATA
# ------------------------------
data_path = "/Users/noeespinosa/Documents/10-Data-analysis/00-capstone-project/Watts_UP-Hydropower_Climate_Optimisation/Watts_UP-Hydropower_Climate_Optimisation/Streamlit/data/Streamlit_Input.csv" #depending on the unzipped csv file, path to be changed.
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip()

# Rename 'Source' to 'ID' for clarity
df.rename(columns={"Source": "ID"}, inplace=True)

# Ensure 'Date' column is datetime
df["Date"] = pd.to_datetime(df["Date"])
df["ds"] = df["Date"]

# ------------------------------
# SIDEBAR / SELECTIONS
# ------------------------------
st.sidebar.header("Select Options")

# Select ID (formerly Source)
selected_id = st.sidebar.selectbox("Select ID", df["ID"].unique())

# Filter df for the selected ID
df_id = df[df["ID"] == selected_id]

# Date selector
min_date = df_id["ds"].min().date()
max_date = df_id["ds"].max().date()
selected_date = st.sidebar.date_input(
    "Pick a date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

# Filter df for the selected date
df_selected = df_id[df_id["ds"].dt.date == selected_date]

# ------------------------------
# CONSUMPTION DISPLAY
# ------------------------------
if not df_selected.empty:
    consumption = df_selected["kwh"].values[0]
    st.sidebar.markdown(
        f"<p style='text-align:center; font-size:18px;'><b>Consumption:</b> {consumption:.2f} kWh</p>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        "<p style='text-align:center; font-size:18px;'><b>Consumption:</b> N/A</p>",
        unsafe_allow_html=True
    )

# ------------------------------
# WEATHER CARD
# ------------------------------
if not df_selected.empty:
    # Extract weather data (first row)
    data = {
        "Temperature_mean": df_selected["Temp_Mean"].values[0],
        "Temperature_min": df_selected["Temp_Min"].values[0],
        "Temperature_max": df_selected["Temp_Max"].values[0],
        "Dewpoint_mean": df_selected["Dewpoint_Mean"].values[0],
        "U_wind_mean": df_selected["U_Wind_Mean"].values[0],
        "V_wind_mean": df_selected["V_Wind_Mean"].values[0],
        "Precip_sum": df_selected["Precipitation_Sum"].values[0],
        "Snowfall_sum": df_selected["Snowfall_Sum"].values[0],
        "Snowcover_mean": df_selected["SnowCover_Mean"].values[0]
    }

    # Determine main weather icon and description
    if data["Snowfall_sum"] > 0 or data["Snowcover_mean"] > 20:
        main_icon = "❄️"
        weather_text = "Snowy"
    elif data["Precip_sum"] > 0.2:
        main_icon = "🌧️"
        weather_text = "Rainy"
    elif data["Temperature_max"] > 0 and data["Precip_sum"] < 0.2:
        main_icon = "☀️"
        weather_text = "Sunny"
    else:
        main_icon = "☁️"
        weather_text = "Cloudy"

    # Wind speed and direction
    wind_speed = round(math.sqrt(data["U_wind_mean"]**2 + data["V_wind_mean"]**2), 1)
    wind_dir = math.degrees(math.atan2(data["U_wind_mean"], data["V_wind_mean"])) % 360
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    wind_label = dirs[round(wind_dir / 45) % 8]

    # Display weather card in sidebar
    st.sidebar.markdown(
        f"""
        <div style="text-align:center; font-size:60px;">{main_icon}</div>
        <h3 style="text-align:center;">{weather_text}</h3>
        <p style="text-align:center; font-size:16px;">
        🌡️ {data["Temperature_min"]:.1f}°C – {data["Temperature_max"]:.1f}°C (avg {data["Temperature_mean"]:.1f}°C)<br>
        💧 Dew Point: {data["Dewpoint_mean"]:.1f}°C<br>
        🌬️ Wind: {wind_speed} m/s {wind_label}<br>
        🌧️ Precipitation: {data["Precip_sum"]:.2f} mm<br>
        ❄️ Snowfall: {data["Snowfall_sum"]:.2f} mm, Cover: {data["Snowcover_mean"]:.2f}%<br>
        </p>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.warning("No weather data available for the selected ID and date.")

# ------------------------------
# MAIN CONTENT
# ------------------------------
# --- HEADER: App Title ---
st.markdown("<h1 style='text-align: center; color: darkseagreen;'>⚡Watt’s Up, Kalam?⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Hydropower & Climate Optimization Dashboard</p>", unsafe_allow_html=True)
st.write("Select an ID and a date from the sidebar to see the power consumption and daily weather summary.")

# Subtitle: "Kalam - Swat District of the Khyber Pakhtunkhwa province, Pakistan

# (Interactive) Show Pictures via sliders    --> @Noé 
st.subheader("📸 Kalam Gallery: Photo Tour")

# Path to your images folder
image_folder = "Images"
image_files = ["hiking.jpg", "Kalam_day.jpg", "Kalam_sunset.jpg", "Kalam-Valley.jpg", 
               "kalam1.jpg", "lake.jpg", "lake1.jpg", "mountain1.jpg", "rapids.jpg", "flow_of_water.jpg", "micro_hydro_powerplat.jpg", "people_working.jpg"]  # Add more filenames as needed
image_paths = [os.path.join(image_folder, f) for f in image_files]
captions = [
    "Hiking view in Kalam",
    "Kalam during the day", 
    "Kalam during sunset",
    "Beautiful Kalam Valley Landscape",
    "Kalam mountains", 
    "Lake Mahodand",
    "Lake Mahodand", 
    "Mountain Falak Sar", 
    "Rapids in Kalam", 
    "flow_of_water.jpg", 
    "micro_hydro_powerplat.jpg", 
    "people_working.jpg"
]

# Initialize session state for image index
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0

# Function to go to next image
def next_image():
    st.session_state.image_index = (st.session_state.image_index + 1) % len(image_paths)

# Function to go to previous image
def prev_image():
    st.session_state.image_index = (st.session_state.image_index - 1) % len(image_paths)

# Get current image and caption
current_index = st.session_state.image_index
img = Image.open(image_paths[current_index])
caption = captions[current_index]

# Display image
st.image(img, caption=f"{caption} ({current_index + 1}/{len(image_paths)})", width=None)

# Navigation buttons
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    if st.button("⬅️"):
        prev_image()

with col3:
    if st.button("➡️"):
        next_image()


#Visualize: Historical vs Prediction 

#Prediciton --> @Team
