### Import libraries

import streamlit as st
import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys
import os
from PIL import Image  # Pillow library
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ------------------------------
# Set Page Layout
# ------------------------------

st.set_page_config(
    page_title="WattsUp – Micro-Hydro Forecasts",
    page_icon="⚡",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Make main content use more width */
    .block-container {
        max-width: 95%;   /* increase/decrease overall width */
        padding-left: 1rem;   /* tighter left margin */
        padding-right: 1rem;  /* tighter right margin */
    }

    /* Adjust sidebar width */
    [data-testid="stSidebar"] {
        min-width: 300px;   /* default ~250px, increase for readability */
        max-width: 350px;   /* prevent it from getting too wide */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Alternative:  Official command, can change between layout = "wide" or "centered" (default) 
#st.set_page_config(layout="wide")  # makes content span more of the screen


# ------------------------------
# LOAD DATA
# ------------------------------
data_path = "data/Streamlit_Input.csv"
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip()

#split Source into two new columns for selectors
df[["Device", "ID"]] = df["Source"].str.extract(r"^(consumer_device_\d+)_(data_user_\d+)$")

# Rename 'Source' to 'ID'
#df.rename(columns={"Source": "ID"}, inplace=True)

# Ensure 'Date' column is datetime
df["Date"] = pd.to_datetime(df["Date"])
df["ds"] = df["Date"]

# ------------------------------
# SIDEBAR / SELECTIONS
# ------------------------------


#Add Logo
st.sidebar.image("images/logo_option_4.png", use_container_width=True) 

st.sidebar.header("Check Your Energy Forecast")         #Previously "Select Options"

# Select ID
#selected_id = st.sidebar.selectbox("Select ID", df["ID"].unique())

# Sidebar selectors
selected_device = st.sidebar.selectbox("Select Device", sorted(df["Device"].unique()))
selected_id     = st.sidebar.selectbox("Select User", sorted(df["ID"].unique()))

selected_source = f"{selected_device}_{selected_id}"

# Filter df for selected ID
#df_id = df[df["ID"] == selected_id].sort_values("ds")
df_id = df[df["Source"] == selected_source].sort_values("ds")

# Select date
min_date = df_id["ds"].min().date()
max_date = df_id["ds"].max().date()
selected_date = st.sidebar.date_input(
    "Pick a date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

# If selected date missing, pick closest previous date
available_dates = df_id["ds"].dt.date.unique()
if selected_date not in available_dates:
    st.sidebar.warning(f"No data for {selected_date}. Showing closest previous available date.")
    selected_date = max(d for d in available_dates if d <= selected_date)

df_selected = df_id[df_id["ds"].dt.date == selected_date]

st.sidebar.markdown("---")  # Visual seperation from the next part

# ------------------------------
# WEATHER DISPLAY FOR SELECTED DATE
# ------------------------------

#st.sidebar.subheader("Weather on Your Selected Day")   #added

st.sidebar.markdown(
    "<div style='margin-left:10px; font-size:1.2em;'>Weather on Your Selected Day</div>",
    unsafe_allow_html=True
)    #Wraping  the subheader, to allign it with weather data

if not df_selected.empty:
    data = {
        "Temperature_mean": df_selected["Temp_Mean"].values[0],
        "Temperature_min": df_selected["Temp_Min"].values[0],
        "Temperature_max": df_selected["Temp_Max"].values[0],
        "Dewpoint_mean": df_selected["Dewpoint_Mean"].values[0],
        "U_wind_mean": df_selected["U_Wind_Mean"].values[0],
        "V_wind_mean": df_selected["V_Wind_Mean"].values[0],
        "Precip_sum": df_selected["Precipitation_Sum"].values[0],
        "Snowfall_sum": df_selected["Snowfall_Sum"].values[0],
        "Snowcover_mean": df_selected["SnowCover_Mean"].values[0],
        "Consumption": df_selected["kwh"].values[0]
    }

    # Weather icon
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

    # Wind info
    wind_speed = round(math.sqrt(data["U_wind_mean"]**2 + data["V_wind_mean"]**2), 1)
    wind_dir = math.degrees(math.atan2(data["U_wind_mean"], data["V_wind_mean"])) % 360
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    wind_label = dirs[round(wind_dir / 45) % 8]

    # Sidebar display
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
    st.sidebar.warning("No data available for selected ID and date.")

st.sidebar.markdown("---")  # Visual seperation from the next part

# ------------------------------
# About the App Info Text
# ------------------------------

with st.sidebar.expander("ℹ️ About this app", expanded=False):
    st.markdown(
        """
        This is a **prototype** application designed for members of the Kalam community.  
        It allows you to:

        - Check your household's **past hydropower supply** (kWh)  
        - View your **forecasted hydropower supply** for upcoming days  
        - See the **weather conditions** for the selected day  

        The goal is to help the community **understand and manage their energy use** more easily.  
        """
    )    

# ------------------------------
# 10-DAY FORECAST WINDOW
# ------------------------------
forecast_start = pd.to_datetime(selected_date)
forecast_end = forecast_start + timedelta(days=9)
df_forecast = df_id[(df_id["ds"] >= forecast_start) & (df_id["ds"] <= forecast_end)].copy()

# Pad missing dates to ensure 10-day continuity
all_dates = pd.date_range(forecast_start, forecast_end)
df_forecast = df_forecast.set_index("ds").reindex(all_dates).rename_axis("ds").reset_index()

# ------------------------------
# MAIN CONTENT
# ------------------------------

# ------------------------------
# Titel and Slideshow
# ------------------------------
st.markdown("<h1 style='text-align: center; color: #00D4FF;'>⚡Watt’s Up, Kalam?⚡</h1>", unsafe_allow_html=True)
#st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Hydropower & Climate Optimization Dashboard</p>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 25px;'>
        <h3 style='color:#00D4FF; font-size:28px; font-weight:600; display:inline-block; margin:0;'>
            Kalam Micro-Hydro Energy Insights Dashboard
        </h3>
        <div style='width:50%; height:3px; background-color:#00D4FF; margin: 5px auto 0 auto; border-radius:2px;'></div>
    </div>
    """,
    unsafe_allow_html=True
)



#Path to your images folder
image_folder = "images"
image_files = ["micro_hydro_powerplat.jpg","flow_of_water.jpg","rapids.jpg","people_working.jpg","hiking.jpg","kalam_day.jpg", "kalam_sunset.jpg",
               "mountain1.jpg", "lake.jpg", "kalam1.jpg"]  # Add more filenames as needed
image_paths = [os.path.join(image_folder, f) for f in image_files]
captions = [
    "Micro hydro-power plant.jpg",
    "Flow of water used in the MHP.jpg",
    "Rapids in Kalam",
    "Community members working on the MHP", 
    "Hiking view in Kalam",
    "Kalam during the day", 
    "Kalam during sunset",
    "Kalam mountains", 
    "Lake Mahodand", 
    "Mountain Falak Sar", 
]

#Initialize session state for image index
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0

#Function to go to next image
def next_image():
    st.session_state.image_index = (st.session_state.image_index + 1) % len(image_paths)

#Function to go to previous image
def prev_image():
    st.session_state.image_index = (st.session_state.image_index - 1) % len(image_paths)

#Get current image and caption
current_index = st.session_state.image_index
img = Image.open(image_paths[current_index])
caption = captions[current_index]

# center the image on main page 
# Display image centered using a column layout
col1, col2, col3 = st.columns([1.5, 7, 0.1])  # middle column wider
with col2:
    st.image(
        img,
        caption=f"{caption} ({current_index + 1}/{len(image_paths)})",
        width=1000  # fixed width
        # use_container_width=False  # not needed when width is specified
    )

#Alternative Display of images with automated wifth adjustment
#st.image(img, caption=f"{caption} ({current_index + 1}/{len(image_paths)})")    #, width=None

#Navigation buttons
col1, col2, col3 = st.columns([1, 7, 1])

with col1:
    if st.button("⬅️"):
        prev_image()

with col3:
    if st.button("➡️"):
        next_image()


# ------------------------------
# PREDICTION VALUE
# ------------------------------        

# Define cutoff date
cutoff_date = pd.to_datetime("2024-09-24").date()  # make it a date

if not df_selected.empty:
    consumption_value = data["Consumption"]

    # Determine if historic or forecast
    if selected_date < cutoff_date:
        data_type = "Historic value"
        bg_gradient = "linear-gradient(135deg, #4CAF50, #2E7D32)"  # green tones
        border_color = "#1B5E20"
    else:
        data_type = "Forecast value"
        bg_gradient = "linear-gradient(135deg, #00D4FF, #0077B6)"  # blue/cyan tones
        border_color = "#005B9C"

    st.markdown(
        f"""
        <div style='
            text-align:center; 
            background: {bg_gradient}; 
            padding:25px; 
            border-radius:20px; 
            border: 3px solid {border_color};
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
            margin-bottom:20px;
        '>
            <h3 style='color:#FFFFFF; font-size:24px; margin:0;'>{data_type}</h3>
            <h2 style='color:#FFFFFF; font-size:52px; margin:10px 0 0 0;'> {consumption_value:.2f} kWh </h2>
            <p style='color:#E0F7FA; font-size:22px; margin:10px 0 0 0;'>
                Power consumption for 
                <span style="color:#00FFFF; font-weight:bold; font-size:24px;">{selected_id}</span> 
                (Device: <span style="color:#00FFFF; font-weight:bold; font-size:24px;">{selected_device}</span>) 
                on <span style="color:#00FFFF; font-weight:bold; font-size:24px;">{selected_date}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("No hydropower supply data available for the selected date and ID.")

# ------------------------------
# PLOTS
# ------------------------------



tab1, tab2, tab3, tab4, tab5 = st.tabs(["Hydropower Supply", "Temperature", "Precipitation", "Snowfall", "Snow Cover"])

# --- Tab 1: kWh Consumption (historical vs predicted) ---
with tab1:
    if not df_forecast.empty:
        split_date = datetime(2024, 9, 24)
        df_forecast_sorted = df_forecast.sort_values("Date")

        # Historical data up to 23rd September
        df_hist = df_forecast_sorted[df_forecast_sorted["Date"] <= split_date - pd.Timedelta(days=1)]

        # Predicted data from 24th September onward
        df_pred = df_forecast_sorted[df_forecast_sorted["Date"] >= split_date].copy()

        # Add the last historical point to the predicted line for smooth connection
        if not df_hist.empty and not df_pred.empty:
            df_pred = pd.concat([
                df_hist.tail(1),  # last historical point
                df_pred
            ])

        fig_kwh = go.Figure()

        # Historical line
        fig_kwh.add_trace(go.Scatter(
            x=df_hist["Date"],
            y=df_hist["kwh"],
            mode='lines+markers',
            name='Historical',
            line=dict(color='green', width=4),   #blue
            marker=dict(size=8)                  # Bigger markers
        ))

        # Predicted dashed line
        fig_kwh.add_trace(go.Scatter(
            x=df_pred["Date"],
            y=df_pred["kwh"],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#00BFFF', dash='dot', width=4),
            marker=dict(size=8)                  # Bigger markers  
        ))

        fig_kwh.update_layout(
            title="Hydropower Supply (kWh)",
            xaxis_title="Date",
            yaxis_title="kWh",
            legend_title="Type",
            template="plotly_white"
        )

        st.plotly_chart(fig_kwh, use_container_width=True)
    else:
        st.info("No hydropower supply forecast data available.")

# --- Tab 2: Temperature ---
with tab2:
    if not df_forecast.empty:
        fig_temp = px.line(
            df_forecast,
            x="Date",
            y="Temp_Mean",
            markers=True,
            title="Temperature Forecast (°C)",
            labels={"Temp_Mean": "Temperature (°C)"}  # <-- updated label
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    else:
        st.info("No temperature forecast data available.")

# --- Tab 3: Precipitation ---
with tab3:
    if not df_forecast.empty:
        fig_precip = px.bar(df_forecast, x="Date", y="Precipitation_Sum", color="Precipitation_Sum",
                            labels={"Precipitation_Sum": "Precipitation (mm)"}, title="Precipitation Forecast (mm)")
        st.plotly_chart(fig_precip, use_container_width=True)
    else:
        st.info("No precipitation forecast data available.")

# --- Tab 4: Snowfall ---
with tab4:
    if not df_forecast.empty:
        fig_snow = px.line(
            df_forecast,
            x="Date",
            y="Snowfall_Sum",
            markers=True,
            title="Snowfall Forecast (mm)",
            labels={"Snowfall_Sum": "Snowfall (mm)"}  # <-- updated label
        )
        fig_snow.update_traces(line_color="lightblue", fill="tozeroy")
        st.plotly_chart(fig_snow, use_container_width=True)
    else:
        st.info("No snowfall forecast data available.")

# --- Tab 5: Snow Cover ---
with tab5:
    if not df_forecast.empty:
        fig_snow_cover = px.area(df_forecast, x="Date", y="SnowCover_Mean",
                                 labels={"SnowCover_Mean": "Snow Cover (%)"}, title="Snow Cover Forecast (%)")
        st.plotly_chart(fig_snow_cover, use_container_width=True)
    else:
        st.info("No snow cover forecast data available.")