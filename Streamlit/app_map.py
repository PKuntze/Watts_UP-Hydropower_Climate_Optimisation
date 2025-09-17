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
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import plotly.graph_objects as go



# LOAD Data: merged Table - ANN output + historical data --> @Bernd
data_path="Streamlit/Streamlit_Input.csv"
df= pd.read_csv(data_path)

### Section: Left Sidebar

# LOGO   --> @ Florencia

# Select User ID (from list)    --> store this as userID 
# Select Date (from calendar)   --> stor this as selected_date

# Weather of the day --> @Gozal
# ------------------------------
# LOAD DATA
# ------------------------------
data_path = "Streamlit/Streamlit_Input.csv"
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip()

# Rename 'Source' to 'ID'
df.rename(columns={"Source": "ID"}, inplace=True)

# Ensure 'Date' column is datetime
df["Date"] = pd.to_datetime(df["Date"])
df["ds"] = df["Date"]

# ------------------------------
# SIDEBAR / SELECTIONS
# ------------------------------
st.sidebar.header("Select Options")

# Select ID
selected_id = st.sidebar.selectbox("Select ID", df["ID"].unique())

# Filter df for selected ID
df_id = df[df["ID"] == selected_id].sort_values("ds")

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

# ------------------------------
# WEATHER DISPLAY FOR SELECTED DATE
# ------------------------------
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
        ⚡ Consumption: {data["Consumption"]:.2f} kWh
        </p>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.warning("No data available for selected ID and date.")

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
st.title("⚡Watt’s Up, Kalam?⚡")
st.write(f"Showing forecast for **ID: {selected_id}** starting from **{selected_date}**")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Consumption", "Temperature", "Precipitation", "Snowfall", "Snow Cover"])

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
            line=dict(color='blue')
        ))

        # Predicted dashed line
        fig_kwh.add_trace(go.Scatter(
            x=df_pred["Date"],
            y=df_pred["kwh"],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', dash='dot')
        ))

        fig_kwh.update_layout(
            title="Power Consumption (kWh)",
            xaxis_title="Date",
            yaxis_title="kWh",
            legend_title="Type",
            template="plotly_white"
        )

        st.plotly_chart(fig_kwh, use_container_width=True)
    else:
        st.info("No consumption forecast data available.")

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



