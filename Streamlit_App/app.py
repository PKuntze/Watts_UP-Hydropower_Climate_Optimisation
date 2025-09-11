### Import libraries

import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys
from PIL import Image  # Pillow library
# from functions import load_and_print, penguins_on_island, plot_penguins    

#st.set_option('deprecation.showPyplotGlobalUse', False)  
#let's one use matplotlib.pyplot() in global state (which is deprictted in streamlit) and suppresses the warning.
# Not necessary if code calls pyplot in the new style

# Let's give a title
st.title("⚡Watt’s Up, Kalam?⚡")

# Show Picture
img = Image.open("images/micro_hydroplant.jpeg")
st.image(img, caption="Micro Hydroplant in Kalam", use_container_width=True)

# --- 1. Load dataset ---
data_path="data/ANN_output.csv"
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


# --- 3. Apply preprocessing ---
df = split_id_column(df)

# --- 4. Display data ---
st.header("Predicted Energy Consumption")

#function to cache the rows whih will be displayed. Otherwise, everytime something is triggered on the app, new rows would be selected.
@st.cache_data    #caches the displayed rows
def get_sample(df, n=10):
    return df.sample(n=n, random_state=42)  # optional random_state for reproducibility
df_sample = get_sample(df)

st.dataframe(df_sample)
st.markdown("**This dataset contains daily predicted energy consumption (kWh) measured for each houshold in Kalam.**")


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


### Display the prediction
#if not filtered.empty:
#    pred_kwh = filtered["pred_kwh"].values[0]
#    st.write(f"Predicted energy consumption for ID {selected_id} on {selected_date}: **{pred_kwh:.2f} kWh**")
#else:
#    st.write("No data available for this combination.")

# please note: In Streamlit, the whole app.py script reruns every time a widget value changes (like your date or ID selection). 
# To prevent certain things to be affected by this, you have to cache them!   

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