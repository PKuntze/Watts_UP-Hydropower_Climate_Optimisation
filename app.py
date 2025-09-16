### Import libraries

import math
from time import time
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys
from PIL import Image  # Pillow library
import numpy as np



# Let's give a title
st.title("Kalam - Swat District of the Khyber Pakhtunkhwa province, Pakistan")

# Show Picture
img = Image.open("images/hydroplant.jpg")
st.image(img, caption="Micro Hydroplant in Kalam", width=500)

# --- 1. Load dataset ---
#data_path="zindi_prophet_submission_all_sources.csv"
#df= pd.read_csv(data_path)
data_path="MySubmission_9_MLP.csv"
df= pd.read_csv(data_path)

# --- 2. Define preprocessing function ---
def split_id_column(df, id_col="ID", value_col="pred_kwh"):
    """
    Splits the ID column into 'date' and 'ID', 
    drops the original ID, 
    and reorders columns as ['date', 'ID', value_col].
    """
    # Split into 'date' and 'source'
    df[["date", "source"]] = df[id_col].str.split("_", n=1, expand=True)

    # Drop the original ID
    df = df.drop(columns=[id_col])

    # Rename 'source' back to 'ID'
    df = df.rename(columns={"source": "ID"})

    # Reorder columns
    df = df[["date", "ID", value_col]]
    
    # Convert date string to datetime (optional but often useful)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

def load_model(model_path):
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def animate_weather(df,date_val):
        # Example daily data
    data = {
        "Temperature_mean": df["Temperature (°C)_mean"].values[0],
        "Temperature_min": df["Temperature (°C)_min"].values[0],
        "Temperature_max": df["Temperature (°C)_max"].values[0],
        "Dewpoint_mean": df["Dewpoint Temperature (°C)_mean"].values[0],
        "U_wind_mean": df["U Wind Component (m/s)_mean"].values[0],   # west-east
        "V_wind_mean": df["V Wind Component (m/s)_mean"].values[0],    # south-north
        "Precip_sum": df["Total Precipitation (mm)_sum"].values[0],
        "Snowfall_sum": df["Snowfall (mm)_sum"].values[0],
        "Snowcover_mean": df["Snow Cover (%)_mean"].values[0]
    }
   

    # --- Determine main weather icon ---
    if data["Snowfall_sum"] > 0 or data["Snowcover_mean"] > 20:
        main_icon = "❄️"
        weather_text = "Snowy"
    elif data["Precip_sum"] > 0.2:
        main_icon = "🌧️"
        weather_text = "Rainy"
    elif data["Temperature_max"] > 0 and data["Precip_sum"] <0.2:
        main_icon = "☀️"
        weather_text = "Sunny"
    else:
        main_icon = "☁️"
        weather_text = "Cloudy"

    # --- Wind direction from U/V components ---
    wind_speed = round(math.sqrt(data["U_wind_mean"]**2 + data["V_wind_mean"]**2), 1)
    wind_dir = math.degrees(math.atan2(data["U_wind_mean"], data["V_wind_mean"]))
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    wind_label = dirs[int((wind_dir % 360)/45)]

    # --- Layout card ---
    st.markdown(
        f"""
        <div style="text-align:center; font-size:80px;">{main_icon}</div>
        <h2 style="text-align:center;">{weather_text}</h2>
        <p style="text-align:center; font-size:20px;">
        🌡️ {data["Temperature_min"]:.1f}°C – {data["Temperature_max"]:.1f}°C (avg {data["Temperature_mean"]:.1f}°C)<br>
        💧 Dew Point: {data["Dewpoint_mean"]:.1f}°C<br>
        🌬️ Wind: {wind_speed} m/s {wind_label}<br>
        🌧️ Precipitation: {data["Precip_sum"]:.2f} mm<br>
        ❄️ Snowfall: {data["Snowfall_sum"]:.2f} mm, Cover: {data["Snowcover_mean"]:.2f}%<br>
        </p>
        """,
        unsafe_allow_html=True
    )
    

# --- 3. Apply preprocessing ---
df = split_id_column(df)

# --- 4. Display data ---
st.header("Predicted Energy Consumption")



# --- 5. Check Prediction for specific date and user ---
df["date"] = pd.to_datetime(df["date"])
df["ID"] = df["ID"].astype(str)  # safer for selectbox

### Create widgets for user input

# Date Selector
selected_date = st.date_input(    #creates a calendar picker for dates
    "Select a date", 
    value=df["date"].min(), 
    min_value=df["date"].min(), 
    max_value=df["date"].max()
)

# ID selector
selected_id = st.selectbox(      #dropdown menu for IDs
    "Select an ID", 
    options=df["ID"].unique()
)

### Filter the df

filtered = df[
    (df["date"] == pd.Timestamp(selected_date)) &
    (df["ID"] == selected_id)
]
# Display the prediction
if not filtered.empty:
    pred_kwh = filtered["pred_kwh"].values[0]

    # Highlighted metric
    st.metric(
        label=f"Predicted energy consumption for ID {selected_id} on {selected_date}",
        value=f"{pred_kwh:.2f} kWh"
    )
else:
    st.write("No data available for this combination.")

#----SHOW WEATHER TRENDS ----based on the whole timeline
#load csv file test and future features
future_features = pd.read_csv('models/prophet_future_features.csv')
train_features = pd.read_csv('models/prophet_train_features.csv')
test_features= pd.read_csv('models/prophet_test_features.csv')

#check min and max date
future_features["ds"] = pd.to_datetime(future_features["date"])
min_date=min(pd.to_datetime([future_features["ds"].min(),test_features["ds"].min(),train_features["ds"].min()]))
max_date=max(pd.to_datetime([future_features["ds"].max(),test_features["ds"].max(),train_features["ds"].max()]))

# Date Selector
st.header("Daily Weather Summary")
selected_date_weather = st.date_input(    #creates a calendar picker for dates
    "Select a date", 
    value=df["date"].min(), 
    min_value=min_date, 
    max_value=max_date
)
#filter for date
dfs = {
    "df1": future_features,
    "df2": test_features,
    "df3": train_features
}
frame_found={}


def filter_dfs(date,dfs):
    for name, df in dfs.items():
        if pd.to_datetime(selected_date_weather) in pd.to_datetime(df["ds"]).values:
        #frame_found[name] = df[df["ds"] == selected_date_weather]
            return df
        
def plot_clima(df_selected,selected_date_weather,param,Farbe='blue'):
    mask = (pd.to_datetime(df_selected["ds"]) >= pd.to_datetime(selected_date_weather)) & (pd.to_datetime(df_selected["ds"]) < pd.to_datetime(selected_date_weather) + pd.Timedelta(days=7))
    week_df = df_selected.loc[mask]

    # Plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(week_df["ds"], week_df[param], marker="o", linestyle="-", color=Farbe)
    ax.set_title(f"{param} Trend for the Week Starting {selected_date_weather}")
    ax.set_xlabel("Date")
    ax.set_ylabel(param)
    ax.grid(True)

    st.pyplot(fig)

#filter the df based on selected date    
df_selected=filter_dfs(selected_date_weather,dfs)  
#print(df_selected)
animate_weather(df_selected[pd.to_datetime(df_selected['ds'])==pd.to_datetime(selected_date_weather)],selected_date_weather)
        
#Temperature Plot for 1 week starting from selected date 
# Filter 7-day window
#st.header('Weather forecast for the next week')
#plot_clima(df_selected,selected_date_weather,'Temperature (°C)_mean')

#plot_clima(df_selected,selected_date_weather,'Total Precipitation (mm)_sum','orange')

#plot_clima(df_selected,selected_date_weather,'Snowfall (mm)_sum', 'red')

#plot_clima(df_selected,selected_date_weather,'Snow Cover (%)_mean', 'purple')

import streamlit as st

st.header('Weather Forecast for the Next Week')

tab1, tab2, tab3, tab4 = st.tabs(["Temperature", "Precipitation", "Snowfall", "Snow Cover"])

with tab1:
   #plot_clima(df_selected, selected_date_weather, 'Temperature (°C)_mean')
   fig_temp = px.line(df_selected, x='date', y='Temperature (°C)_mean', markers=True, title="Temperature Trend")
   st.plotly_chart(fig_temp, use_container_width=True)
with tab2:
    #plot_clima(df_selected, selected_date_weather, 'Total Precipitation (mm)_sum', 'orange')
    with tab2:
     fig = px.bar(
        df_selected, 
        x='date', 
        y='Total Precipitation (mm)_sum', 
        color='Total Precipitation (mm)_sum', 
        labels={'Total Precipitation (mm)_sum':'Precipitation (mm)'},
        title='Daily Precipitation Forecast'
    )
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    #plot_clima(df_selected, selected_date_weather, 'Snowfall (mm)_sum', 'red')
    fig_snow = px.line(
        df_selected, 
        x='date', 
        y='Snowfall (mm)_sum',
        title="Daily Snowfall",
        markers=True
    )
    fig_snow.update_traces(line_color="lightblue", fill="tozeroy")  # soft area fill
    st.plotly_chart(fig_snow, use_container_width=True)
with tab4:
    #plot_clima(df_selected, selected_date_weather, 'Snow Cover (%)_mean', 'purple')
    fig_snow_cover = px.area(
        df_selected, 
        x='date', 
        y='Snow Cover (%)_mean',
        labels={'Snow Cover (%)_mean': 'Snow Cover (%)'},
        title="Snow Cover Trend",
        color_discrete_sequence=['#ADD8E6']
    )
    st.plotly_chart(fig_snow_cover, use_container_width=True)

