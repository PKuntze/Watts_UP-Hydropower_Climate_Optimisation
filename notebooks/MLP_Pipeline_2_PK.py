"""
MLP_Pipeline.py

Full pipeline for hydropower electricity demand forecasting:
- Load & clean consumption and climate data
- Merge, reindex, feature engineering (lags, rolling stats, ratios)
- Train an MLP neural network (TensorFlow/Keras)
- Evaluate on train/test with RMSE & MAPE
- Predict an extra month (future climate data)
- Save submission file
- Plot results:
    * Extra month predictions (20 random sources in subplots + total demand)
    * Residuals (histograms, scatter plots)
    * Uncertainty bands with Monte Carlo Dropout
"""

import os
import time
#import random
import numpy as np
from scipy.stats import median_abs_deviation as mad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -------------------------------
# 1. Config
# -------------------------------

def model_id_and_folders(base_dir, model_id):
    """Creates new folders with an incremented suffix and update model_id."""
    i = 1
    model_id_i = f"{model_id}_{i}"
    
    while os.path.exists(f"{base_dir}/models/{model_id_i}"):
        i += 1
        model_id_i = f"{model_id}_{i}"
    os.makedirs(f"{base_dir}/models/{model_id_i}", exist_ok=True)
    os.makedirs(f"{base_dir}/plots/{model_id_i}")
    os.makedirs(f"{base_dir}/submissions/{model_id_i}")

    return model_id_i

BASEDIR = "/Users/patrickkuntze/Desktop/DS_bootcamp/Capstone/Watts_UP-Hydropower_Climate_Optimisation/"
MODELNAME = model_id_and_folders(BASEDIR, "FP_MonteCarlo_median")
MC_MODE = 'mean'
MC_MODE = 'median'
PREDICT_0_POWER = False

DATA_CONSUMPTION = BASEDIR+"data/Data.csv"
DATA_CLIMATE = BASEDIR+"data/Kalam_Climate_Data.xlsx"
DATA_SUBMISSION = BASEDIR+"data/SampleSubmission.csv"
OUTPUT_SUBMISSION = f"{BASEDIR}/submissions/{MODELNAME}/MySubmission_final_MLP{MODELNAME}.csv"
PLOTS_DIR = f"{BASEDIR}/plots/{MODELNAME}"
MODELS_DIR = f"{BASEDIR}/models/{MODELNAME}"

#os.makedirs(PLOTS_DIR, exist_ok=True)


# -------------------------------
# 2. Data Preparation
# -------------------------------
def load_climate(path=DATA_CLIMATE):
    """Load and aggregate daily climate data."""
    df = pd.read_excel(path)
    df["Date Time"] = pd.to_datetime(df["Date Time"])
    df["date"] = df["Date Time"].dt.date

    daily = df.groupby("date").agg({
        "Temperature (°C)": ["mean", "min", "max"],
        "Dewpoint Temperature (°C)": ["mean", "min", "max"],
        "U Wind Component (m/s)": "mean",
        "V Wind Component (m/s)": "mean",
        "Total Precipitation (mm)": "sum",
        "Snowfall (mm)": "sum",
        "Snow Cover (%)": "mean"
    })
    daily.columns = ["_".join(col).strip() for col in daily.columns.values]
    daily = daily.reset_index()

    rename = {
        "date": "Date",
        "Temperature (°C)_mean": "Temp_Mean",
        "Temperature (°C)_min": "Temp_Min",
        "Temperature (°C)_max": "Temp_Max",
        "Dewpoint Temperature (°C)_mean": "Dewpoint_Mean",
        "Dewpoint Temperature (°C)_min": "Dewpoint_Min",
        "Dewpoint Temperature (°C)_max": "Dewpoint_Max",
        "U Wind Component (m/s)_mean": "U_Wind_Mean",
        "V Wind Component (m/s)_mean": "V_Wind_Mean",
        "Total Precipitation (mm)_sum": "Precipitation_Sum",
        "Snowfall (mm)_sum": "Snowfall_Sum",
        "Snow Cover (%)_mean": "SnowCover_Mean"
    }
    return daily.rename(columns=rename)


def load_consumption(path=DATA_CONSUMPTION):
    """Load electricity consumption data and aggregate daily totals."""
    df = pd.read_csv(path)
    df.drop(columns=["consumer_device_9", "consumer_device_x", "v_red", "v_blue",
                     "v_yellow", "current", "power_factor"], inplace=True, errors="ignore")

    df["date_time"] = pd.to_datetime(df["date_time"])
    df["Date"] = df["date_time"].dt.date

    daily = df.groupby(["Source", "Date"]).agg({"kwh": "sum"}).reset_index()
    return daily


def merge_data(consumption, climate):
    """Merge consumption and climate data on date."""
    climate["Date"] = pd.to_datetime(climate["Date"])
    consumption["Date"] = pd.to_datetime(consumption["Date"])

    merged = consumption.merge(climate, on="Date", how="left")
    return merged


# -------------------------------
# 3. Feature Engineering
# -------------------------------
def add_features(df):
    df = df.copy()
    df["Temp_dew_diff"] = df["Temp_Mean"] - df["Dewpoint_Mean"]
    df["wind_speed"] = np.sqrt(df["U_Wind_Mean"]**2 + df["V_Wind_Mean"]**2)
    df["precip_snow_ratio"] = df["Precipitation_Sum"] / (df["Snowfall_Sum"] + 1e-6)
    return df


def add_lag_roll(df, group_col="Source", target_col="kwh",
                 lags=[1, 2, 7, 14], windows=[3, 7, 14]):
    """Create lag and rolling mean features."""
    df = df.sort_values([group_col, "Date"]).copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = df.groupby(group_col)[target_col].shift(1).rolling(w).mean()
    return df


def reindex_sources(df):
    """Ensure every source has full date coverage."""
    all_dates = pd.date_range(df["Date"].min(), df["Date"].max())
    all_sources = df["Source"].unique()
    idx = pd.MultiIndex.from_product([all_sources, all_dates], names=["Source", "Date"])

    df_full = df.set_index(["Source", "Date"]).reindex(idx).reset_index()
    df_full["kwh"] = df_full["kwh"].fillna(0)

    climate_cols = [c for c in df.columns if c not in ["Source", "kwh", "Date"]]
    for c in climate_cols:
        df_full[c] = df_full.groupby("Date")[c].transform("first")
    return df_full


# -------------------------------
# 4. Model
# -------------------------------
def build_mlp(input_dim, dropout_rate=0.2):
    """Build an MLP with dropout layers for Monte Carlo Dropout."""
    model = models.Sequential([
        layers.Dense(128, activation="relu", input_dim=input_dim),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="relu")  # enforce non-negative
    ])
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.Huber(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


# -------------------------------
# 5. Residual Plots
# -------------------------------
def plot_residuals(y_true, y_pred, dataset_name="Test", zoom_limit=500):
    residuals = y_true - y_pred

    # Full histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=100, kde=False)
    plt.title(f"{dataset_name} Residuals Histogram (Full Range)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"residuals_hist_full_{dataset_name}.png"))
    plt.close()

    # Zoomed histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=100, kde=False)
    plt.xlim(-zoom_limit, zoom_limit)
    plt.title(f"{dataset_name} Residuals Histogram (Zoom ±{zoom_limit})")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"residuals_hist_zoom_{dataset_name}.png"))
    plt.close()

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"{dataset_name} Residuals vs Predictions")
    plt.xlabel("Predicted kWh")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"residuals_scatter_{dataset_name}.png"))
    plt.close()


# -------------------------------
# 6. Monte Carlo Dropout
# -------------------------------

# --- Option 1
def mc_dropout_predictions(model, X, n_iter=50):
    """Run multiple stochastic forward passes to estimate uncertainty."""
    preds = []
    for _ in range(n_iter):
        preds.append(model(X, training=True).numpy().flatten())
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)

# --- 6. Monte Carlo Dropout ... Option 2
@tf.function
def dropout_forward(model, X):
    """Single dropout-enabled forward pass."""
    return model(X, training=True)

def mc_dropout_predictions_parallel(model, X, n_iter=50, batch_size=1, reduction='mean'):
    """
    Monte Carlo Dropout predictions with configurable uncertainty measure.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model with Dropout layers.
    X : tf.Tensor or np.ndarray
        Input batch.
    n_iter : int
        Number of stochastic forward passes.
    batch_size : int
        Number of MC samples to process per chunk (controls memory vs. speed).
    reduction : str
        "mean" → use mean + std
        "median" → use median + MAD

    Returns
    -------
    central : np.ndarray
        Mean or median predictions (shape = batch_size).
    spread : np.ndarray
        Std or MAD uncertainty (shape = batch_size).
    """
    preds = []

    for start in range(0, n_iter, batch_size):
        # Repeat inputs batch_size times
        X_tiled = tf.tile(tf.expand_dims(X, 0), [batch_size, 1, 1])
        X_tiled = tf.reshape(X_tiled, (-1,) + X.shape[1:])

        # Forward pass with dropout active
        preds_chunk = dropout_forward(model, X_tiled)

        # Reshape to (batch_size, batch_size)
        preds_chunk = tf.reshape(preds_chunk, (batch_size, X.shape[0]))
        preds.append(preds_chunk)

    # Combine all chunks → shape (n_iter, batch_size)
    preds = tf.concat(preds, axis=0)

    if reduction == "mean":
        central = tf.reduce_mean(preds, axis=0)
        spread = tf.math.reduce_std(preds, axis=0)

    elif reduction == "median":
        central = np.median(preds.numpy(), axis=0)
        spread = mad(preds.numpy(), axis=0)

    else:
        raise ValueError("reduction must be 'mean' or 'median'")

    return central.numpy() if isinstance(central, tf.Tensor) else central, \
           spread.numpy() if isinstance(spread, tf.Tensor) else spread

# -------------------------------
# 7. Pipeline
# -------------------------------
def run_pipeline():
    
    start_timer_proc = time.perf_counter()  # high-precision timer
    
    # --- Load data
    climate = load_climate()
    consumption = load_consumption()
    df = merge_data(consumption, climate)

    # --- Features
    df["Date"] = pd.to_datetime(df["Date"])
    df = reindex_sources(df)
    df = add_features(df)
    df = add_lag_roll(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "Temp_Mean", "Temp_Min", "Temp_Max", "Dewpoint_Mean", "Dewpoint_Min", "Dewpoint_Max",
        "U_Wind_Mean", "V_Wind_Mean", "Precipitation_Sum", "Snowfall_Sum", "SnowCover_Mean",
        "Temp_dew_diff", "wind_speed", "precip_snow_ratio"
    ] + [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_")]

    X, y = df[feature_cols], df["kwh"].values

    # --- Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    end_timer_proc = time.perf_counter()  # high-precision timer
    # --- Calculate data pre processing run time
    runtime = end_timer_proc - start_timer_proc
    with open(f'{MODELS_DIR}/model_{MODELNAME}_run_time.txt', 'w') as file:
        file.write(f'MLP_{MODELNAME} pre processing run time: ')
        file.write(f'{int(runtime//60)} min {int(runtime%60//1)} sec\n')
    file.close()
    start_timer_train = time.perf_counter()  # high-precision timer

    # --- Train model
    model = build_mlp(X_train.shape[1])
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
    lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0)

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=100, batch_size=256,
              callbacks=[es, lr], verbose=0)
    
    # --- Save model
    model.save(f'{MODELS_DIR}/model_{MODELNAME}.h5')

    # --- Evaluate
    y_pred_train = np.maximum(0, model.predict(X_train).flatten())
    y_pred_test = np.maximum(0, model.predict(X_test).flatten())

    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

    print(f"Train RMSE = {rmse_train:.3f}")
    print(f"Test RMSE = {rmse_test:.3f}, MAPE = {mape_test:.3f}")

    # --- Residual plots
    plot_residuals(y_train, y_pred_train, "Train", zoom_limit=500)
    plot_residuals(y_test, y_pred_test, "Test", zoom_limit=500)

    end_timer_train = time.perf_counter()  # high-precision timer
    # --- Calculate training run time
    runtime = end_timer_train - start_timer_train
    with open(f'{MODELS_DIR}/model_{MODELNAME}_run_time.txt', 'a') as file:
        file.write(f'MLP_{MODELNAME} training + validation run time: ')
        file.write(f'{int(runtime//60)} min {int(runtime%60//1)} sec\n')
    file.close()
    start_timer_pred = time.perf_counter()  # high-precision timer

    # --- Extra month prediction
    start_date, end_date = pd.to_datetime("2024-09-24"), pd.to_datetime("2024-10-24")
    extra_month = climate[(climate["Date"] >= start_date) & (climate["Date"] <= end_date)].copy()
    extra_month = add_features(extra_month)

    preds = []
    for src in df["Source"].unique():
        hist = df[df["Source"] == src].copy()
        future = extra_month.copy()
        future["Source"] = src

        lag_hist = hist.tail(30).reset_index(drop=True)
        rows = []

        for _, row in future.iterrows():
            r = row.to_dict()
            for lag in [1, 2, 7, 14]:
                r[f"lag_{lag}"] = lag_hist.loc[len(lag_hist) - lag, "kwh"] if len(lag_hist) >= lag else np.nan
            for w in [3, 7, 14]:
                r[f"roll_mean_{w}"] = lag_hist["kwh"].iloc[-w:].mean() if len(lag_hist) >= w else lag_hist["kwh"].mean()

            X_ex = pd.DataFrame([r])[feature_cols].fillna(0)
            X_ex = scaler.transform(X_ex)

            # Monte Carlo Dropout predictions
            #mc_mean, mc_std = mc_dropout_predictions(model, X_ex, n_iter=50)
            mc_mean, mc_std = mc_dropout_predictions_parallel(model, X_ex, n_iter=50, batch_size=10, reduction='median')
            r["pred_kwh"] = max(0, mc_mean[0])
            r["pred_std"] = mc_std[0]

            lag_hist = pd.concat([lag_hist, pd.DataFrame([{"kwh": r["pred_kwh"]}])], ignore_index=True)
            rows.append(r)
        preds.append(pd.DataFrame(rows))

    preds_df = pd.concat(preds)

    # --- Save submission
    sub = pd.read_csv(DATA_SUBMISSION)
    sub["Date"] = pd.to_datetime(sub["ID"].apply(lambda x: x.split("_")[0]))
    sub["Source"] = sub["ID"].apply(lambda x: "_".join(x.split("_")[1:]))

    submission = sub.merge(preds_df[["Date", "Source", "pred_kwh"]], on=["Date", "Source"], how="left")
    submission = submission[["ID", "pred_kwh"]]
    submission.to_csv(OUTPUT_SUBMISSION, index=False)
    print(f"✅ Submission saved: {OUTPUT_SUBMISSION}")

    end_timer_pred = time.perf_counter()  # high-precision timer
    # --- Calculate prediction run time
    runtime = end_timer_pred - start_timer_pred
    with open(f'{MODELS_DIR}/model_{MODELNAME}_run_time.txt', 'a') as file:
        file.write(f'MLP_{MODELNAME} prediction + submission run time: ')
        file.write(f'{int(runtime//60)} min {int(runtime%60//1)} sec\n')
    file.close()
    start_timer_plot = time.perf_counter()  # high-precision timer

    # --- Plots: 20 random sources (subplot with uncertainty bands)
    sample_sources = np.random.choice(df["Source"].unique(), 20, replace=False)

    fig, axes = plt.subplots(5, 4, figsize=(20, 15), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, src in enumerate(sample_sources):
        df_src = df[df["Source"] == src].copy()
        preds_src = preds_df[preds_df["Source"] == src].copy()

        ax = axes[i]
        ax.plot(df_src["Date"], df_src["kwh"], label="History", alpha=0.7)
        ax.plot(preds_src["Date"], preds_src["pred_kwh"], label="Forecast", linestyle="--", alpha=0.8)
        ax.fill_between(preds_src["Date"],
                        preds_src["pred_kwh"] - 2*preds_src["pred_std"],
                        preds_src["pred_kwh"] + 2*preds_src["pred_std"],
                        color="orange", alpha=0.3)
        ax.set_title(f"{src}", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=12)

    fig.suptitle("Extra Month Predictions for 20 Random Sources (with Uncertainty)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOTS_DIR, "forecast_20_sources.png"))
    plt.close()

    # --- Plot total demand
    df_total = df.groupby("Date")["kwh"].sum().reset_index()
    preds_total = preds_df.groupby("Date")[["pred_kwh", "pred_std"]].sum().reset_index()

    plt.figure(figsize=(12, 5))
    plt.plot(df_total["Date"], df_total["kwh"], label="Actual Total (history)", alpha=0.7)
    plt.plot(preds_total["Date"], preds_total["pred_kwh"], label="Predicted Total (extra month)", linestyle="--", alpha=0.8)
    plt.fill_between(preds_total["Date"],
                     preds_total["pred_kwh"] - 2*preds_total["pred_std"],
                     preds_total["pred_kwh"] + 2*preds_total["pred_std"],
                     color="orange", alpha=0.3)
    plt.title("Total Electricity Demand")
    plt.xlabel("Date"); plt.ylabel("Total kWh")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "forecast_total.png"))
    plt.close()

    end_timer_plot = time.perf_counter()  # high-precision timer
    # --- Calculate plotting run time
    runtime = end_timer_plot - start_timer_plot
    with open(f'{MODELS_DIR}/model_{MODELNAME}_run_time.txt', 'a') as file:
        file.write(f'MLP_{MODELNAME} plotting run time: ')
        file.write(f'{int(runtime//60)} min {int(runtime%60//1)} sec\n')
    file.close()


# -------------------------------
# 8. Run
# -------------------------------
if __name__ == "__main__":

    start_timer_all = time.perf_counter()  # high-precision timer
    
    run_pipeline()

    end_timer_all = time.perf_counter()  # high-precision timer
    # --- Calculate script run time
    runtime = end_timer_all - start_timer_all
    with open(f'{MODELS_DIR}/model_{MODELNAME}_run_time.txt', 'a') as file:
        file.write(f'MLP_{MODELNAME} script run time: ')
        file.write(f'{int(runtime//60)} min {int(runtime%60//1)} sec')
    file.close()