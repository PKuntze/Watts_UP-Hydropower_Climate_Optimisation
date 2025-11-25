"""
Python script pipeline.py for training an ANN.

Full pipeline for hydropower electricity demand forecasting:
- Load & clean consumption and climate data
- Merge, reindex, feature engineering (lags, rolling stats, ratios)
- Train an MLP neural network (TensorFlow/Keras)
- Evaluate on train/test with RMSE & MAPE
- Predict an extra month (future climate data)
- Save submission file
- Plot results:
    * Extra month predictions (20 random sources + total demand)
    * Residuals (histograms, scatter plots)
    * Feature importance (permutation importance)
"""
import os

import numpy as np
from scipy.stats import median_abs_deviation as mad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (root_mean_squared_error,
                             mean_absolute_percentage_error)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input


# -------------------------------
# 1. Config
# -------------------------------
def model_id_and_folders(base_dir, model_id):
    """Create new folders with an incremented suffix and update model_id."""
    i = 1
    model_id_i = f"{model_id}_{i}"

    while os.path.exists(f"{base_dir}/plots/{model_id_i}"):
        i += 1
        model_id_i = f"{model_id}_{i}"
    os.makedirs(f"{base_dir}/plots/{model_id_i}", exist_ok=True)
    os.makedirs(f"{base_dir}/submissions/{model_id_i}")

    return model_id_i


# Monte carlo mode
# MC_MODE = 'mean'
# MC_MODE = 'median_mad'
MC_MODE = 'median_pc'

# Parallelization
NITER = 1000
ITERBATCH = 500

# Names for files and directories
BASEDIR = ("/Users/patrickkuntze/01-Arbeit/Jobsuche/05_Weiterbildungen/"
           + "DS_bootcamp/Capstone/Watts_UP-Hydropower_Climate_Optimisation/"
           + "src/")
DATA_DIR = BASEDIR + "/data"
DATA_CONSUMPTION = DATA_DIR + "/Data.csv"
DATA_CLIMATE = DATA_DIR + "/Kalam Climate Data.xlsx"
DATA_SUBMISSION = DATA_DIR + "/SampleSubmission.csv"

MODELNAME = model_id_and_folders(BASEDIR,
                                 f"ANN_{MC_MODE}_niter{NITER}_batch{ITERBATCH}")

OUTPUT_SUBMISSION = (f"{BASEDIR}/submissions/{MODELNAME}/"
                     + f"MySubmission_final_MLP_{MODELNAME}.csv")
PLOTS_DIR = f"{BASEDIR}/plots/{MODELNAME}"
MODELS_DIR = f"{BASEDIR}/models/{MODELNAME}"


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
    df.drop(columns=["consumer_device_9", "consumer_device_x", "v_red",
            "v_blue", "v_yellow", "current", "power_factor"],
            inplace=True, errors="ignore")

    df["date_time"] = pd.to_datetime(df["date_time"])
    df["Date"] = df["date_time"].dt.date

    daily = df.groupby(["Source", "Date"]).agg({"kwh": "sum"}).reset_index()
    return daily


def merge_data(consumption, climate):
    """Merge consumption and climate data on date."""
    climate["Date"] = pd.to_datetime(climate["Date"])
    consumption["Date"] = pd.to_datetime(consumption["Date"])
    return consumption.merge(climate, on="Date", how="left")


# -------------------------------
# 3. Feature Engineering
# -------------------------------
def add_features(df):
    """Add new climate features per Source."""
    df = df.copy()
    df["Temp_dew_diff"] = df["Temp_Mean"] - df["Dewpoint_Mean"]
    df["wind_speed"] = np.sqrt(df["U_Wind_Mean"]**2 + df["V_Wind_Mean"]**2)
    df["precip_snow_ratio"] = (df["Precipitation_Sum"]
                               / (df["Snowfall_Sum"] + 1e-6))
    return df


def add_lag_roll(df, group_col="Source", target_col="kwh",
                 lags=None, windows=None):
    """Add lag and rolling mean features per Source."""
    if lags is None:
        lags = [1, 2, 3]
    if windows is None:
        windows = [3, 7, 14]
    df = df.sort_values([group_col, "Date"]).copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = (
            df.groupby(group_col)[target_col].shift(1).rolling(w).mean()
        )
    return df


def reindex_sources(df):
    """Ensure every source has full date coverage."""
    all_dates = pd.date_range(df["Date"].min(), df["Date"].max())
    all_sources = df["Source"].unique()
    idx = pd.MultiIndex.from_product(
        [all_sources, all_dates], names=["Source", "Date"]
    )

    df_full = df.set_index(["Source", "Date"]).reindex(idx).reset_index()
    df_full["kwh"] = df_full["kwh"].fillna(0)

    climate_cols = [c for c in df.columns if c not in ["Source", "kwh", "Date"]]
    for c in climate_cols:
        df_full[c] = df_full.groupby("Date")[c].transform("first")
    return df_full


# -------------------------------
# 4. Model
# -------------------------------
def build_mlp(input_dim):
    """Build simple MLP regressor."""
    model = models.Sequential([
        Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
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
    """Plot histogram + scatter for residuals."""
    residuals = y_true - y_pred

    # Full histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=100, kde=False)
    plt.title(f"{dataset_name} Residuals Histogram (Full Range)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"residuals_hist_full_{dataset_name}.png")
    )
    plt.close()

    # Zoomed histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=100, kde=False)
    plt.xlim(-zoom_limit, zoom_limit)
    plt.title(f"{dataset_name} Residuals Histogram (Zoom ±{zoom_limit})")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"residuals_hist_zoom_{dataset_name}.png")
    )
    plt.close()

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"{dataset_name} Residuals vs Predictions")
    plt.xlabel("Predicted kWh")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"residuals_scatter_{dataset_name}.png")
    )
    plt.close()


# -------------------------------
# 6. Feature Importance (Permutation)
# -------------------------------
def permutation_importance_manual(model, x_val, y_val, feature_names,
                                  n_repeats=5):
    """Compute permutation importance manually for a Keras model."""
    baseline = root_mean_squared_error(
        y_val, np.maximum(0, model.predict(x_val, verbose=0).flatten()),
    )
    importances = {}

    for i, col in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            x_permuted = x_val.copy()
            np.random.shuffle(x_permuted[:, i])  # shuffle one column
            y_pred = np.maximum(0, model.predict(x_permuted,
                                                 verbose=0).flatten())
            score = root_mean_squared_error(y_val, y_pred)
            scores.append(score - baseline)  # increase in RMSE
        importances[col] = np.mean(scores)

    return (
        pd.DataFrame(
            {"feature": list(importances.keys()),
             "importance": list(importances.values())}
        ).sort_values("importance", ascending=False)
    )


# -------------------------------
# 7. Monte Carlo Dropout
# -------------------------------
@tf.function
def dropout_forward(model, x):
    """Single dropout-enabled forward pass."""
    return model(x, training=True, verbose=0)


def mc_dropout_predictions_parallel(model, x, n_iter=50, batch_size=1,
                                    reduction='mean'):
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
        "median_mad" → use median + MAD
        "median_pc" → use median + 5th and 95th percentile

    Returns
    -------
    central : np.ndarray
        Mean or median predictions.
    spread, spread_low, spread_high : np.ndarrays
        Std or MAD uncertainty. Or lower and upper uncertainty estimates using
        percentiles, which avoids negative values.
    """
    preds = []

    for _batch_no in range(0, n_iter, batch_size):
        # Repeat inputs batch_size times
        x_tiled = tf.tile(tf.expand_dims(x, 0), [batch_size, 1, 1])
        x_tiled = tf.reshape(x_tiled, (-1,) + x.shape[1:])

        # Forward pass with dropout active
        preds_chunk = dropout_forward(model, x_tiled)

        # Reshape to (batch_size, batch_size)
        preds_chunk = tf.reshape(preds_chunk, (batch_size, x.shape[0]))
        preds.append(preds_chunk)

    # Combine all chunks → shape (n_iter, batch_size)
    preds = tf.concat(preds, axis=0)

    if reduction == "mean":
        central = tf.reduce_mean(preds.flatten(), axis=0)
        spread = tf.math.reduce_std(preds.flatten(), axis=0)
        return central.numpy() if isinstance(central, tf.Tensor) else central, \
            spread.numpy() if isinstance(spread, tf.Tensor) else spread

    elif reduction == "median_mad":
        central = np.median(preds.numpy().flatten(), axis=0)
        spread = mad(preds.numpy(), axis=0)
        return central.numpy() if isinstance(central, tf.Tensor) else central, \
            spread.numpy() if isinstance(spread, tf.Tensor) else spread

    elif reduction == "median_pc":
        central = np.median(preds.numpy().flatten(), axis=0)
        spread_low = np.percentile(preds.numpy().flatten(), 5, axis=0)
        spread_high = np.percentile(preds.numpy().flatten(), 95, axis=0)
        return central.numpy() if isinstance(central, tf.Tensor) else central, \
            spread_low.numpy() if isinstance(spread_low,
                                             tf.Tensor) else spread_low, \
            spread_high.numpy() if isinstance(spread_high,
                                              tf.Tensor) else spread_high

    else:
        raise ValueError("reduction must be 'mean', 'median_mad',"
                         + " or 'median_pc'")


# -------------------------------
# 8. Pipeline
# -------------------------------
def run_pipeline():
    """Run pipeline for ANN training, validation, and prediction."""
    # --- Load data
    climate = load_climate()
    consumption = load_consumption()
    df = merge_data(consumption, climate)

    # --- Features
    df["Date"] = pd.to_datetime(df["Date"])
    df = reindex_sources(df)
    df = add_features(df)
    df = add_lag_roll(df, lags=[1, 2, 7, 14], windows=[3, 7, 14])
    df = df.dropna().reset_index(drop=True)

    feature_cols = ([
        "Temp_Mean", "Temp_Min", "Temp_Max", "Dewpoint_Mean", "Dewpoint_Min",
        "Dewpoint_Max", "U_Wind_Mean", "V_Wind_Mean", "Precipitation_Sum",
        "Snowfall_Sum", "SnowCover_Mean", "Temp_dew_diff", "wind_speed",
        "precip_snow_ratio"] + [c for c in df.columns if
                                c.startswith("lag_") or c.startswith("roll_")]
    )

    x, y = df[feature_cols], df["kwh"].values

    # --- Scale & split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # --- Train model
    model = build_mlp(x_train.shape[1])
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                 restore_best_weights=True, verbose=0)
    lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=3, min_lr=1e-6, verbose=0)

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=100, batch_size=256, callbacks=[es, lr], verbose=0)

    # --- Evaluate
    y_pred_train = np.maximum(0, model.predict(x_train).flatten())
    y_pred_test = np.maximum(0, model.predict(x_test).flatten())

    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

    print(f"Train RMSE = {rmse_train:.3f}")
    print(f"Test RMSE = {rmse_test:.3f}, MAPE = {mape_test:.3f}")

    # --- Residual plots
    plot_residuals(y_train, y_pred_train, "Train", zoom_limit=500)
    plot_residuals(y_test, y_pred_test, "Test", zoom_limit=500)

    # --- Feature importance
    importances_df = permutation_importance_manual(model, x_test, y_test,
                                                   feature_cols, n_repeats=5)
    print(importances_df.head(20))

    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=importances_df.head(20),
                palette="viridis", hue="feature", legend=False)
    plt.title("Feature Importance (Permutation Importance)")
    plt.xlabel("Importance (Increase in RMSE)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
    plt.close()

    # --- Extra month prediction (future climate data)
    start_date, end_date = (pd.to_datetime("2024-09-24"),
                            pd.to_datetime("2024-10-24"))
    extra_month = climate[
        (climate["Date"] >= start_date) & (climate["Date"] <= end_date)
    ].copy()
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
                r[f"lag_{lag}"] = (
                    lag_hist.loc[
                        len(lag_hist) - lag, "kwh"
                    ] if len(lag_hist) >= lag else np.nan
                )
            for w in [3, 7, 14]:
                r[f"roll_mean_{w}"] = (
                    lag_hist["kwh"].iloc[-w:].mean()
                    if len(lag_hist) >= w else lag_hist["kwh"].mean()
                )

            x_ex = pd.DataFrame([r])[feature_cols].fillna(0)
            x_ex = scaler.transform(x_ex)

            # Monte Carlo Dropout predictions
            if MC_MODE == 'median_pc':
                r["pred_kwh"], r["pred_pc5"], r["pred_pc95"] = (
                    mc_dropout_predictions_parallel(model, x_ex, n_iter=NITER,
                                                    batch_size=ITERBATCH,
                                                    reduction=MC_MODE)
                )
            else:
                r["pred_kwh"], r["pred_std"] = (
                    mc_dropout_predictions_parallel(model, x_ex, n_iter=NITER,
                                                    batch_size=ITERBATCH,
                                                    reduction=MC_MODE)
                )

            lag_hist = pd.concat(
                [lag_hist, pd.DataFrame([{"kwh": r["pred_kwh"]}])],
                ignore_index=True
            )
            rows.append(r)
        preds.append(pd.DataFrame(rows))
    preds_df = pd.concat(preds)

    # --- Save submission
    sub = pd.read_csv(DATA_SUBMISSION)
    sub["Date"] = pd.to_datetime(sub["ID"].apply(lambda x: x.split("_")[0]))
    sub["Source"] = sub["ID"].apply(lambda x: "_".join(x.split("_")[1:]))

    submission = sub.merge(preds_df[["Date", "Source", "pred_kwh"]],
                           on=["Date", "Source"], how="left")
    submission = submission[["ID", "pred_kwh"]]
    submission.to_csv(OUTPUT_SUBMISSION, index=False)
    print(f"✅ Submission saved: {OUTPUT_SUBMISSION}")

    # --- Plots: 20 random sources (subplot with uncertainty bands)
    sample_sources = np.random.choice(df["Source"].unique(), 20, replace=False)

    fig, axes = plt.subplots(5, 4, figsize=(20, 15), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, src in enumerate(sample_sources):
        df_src = df[df["Source"] == src].copy()
        preds_src = preds_df[preds_df["Source"] == src].copy()

        ax = axes[i]
        ax.plot(df_src["Date"], df_src["kwh"], label="History", alpha=0.7)
        ax.plot(preds_src["Date"], preds_src["pred_kwh"], label="Forecast",
                linestyle="--", alpha=0.8)
        if MC_MODE == 'median_pc':
            ax.fill_between(preds_src["Date"],
                            preds_src["pred_pc5"],
                            preds_src["pred_pc95"],
                            color="orange", alpha=0.3)
        else:
            ax.fill_between(preds_src["Date"],
                            preds_src["pred_kwh"] - preds_src["pred_std"],
                            preds_src["pred_kwh"] + preds_src["pred_std"],
                            color="orange", alpha=0.3)
        ax.set_title(f"{src}", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=12)

    fig.suptitle(
        "Extra Month Predictions for 20 Random Sources (with Uncertainty)",
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOTS_DIR, f"forecast_20_sources_{MC_MODE}.png"))
    plt.close()

    for src in sample_sources:
        df_src = df[df["Source"] == src].copy()
        preds_src = preds_df[preds_df["Source"] == src].copy()

        plt.figure(figsize=(12, 5))
        plt.plot(df_src["Date"], df_src["kwh"], label="Actual (history)",
                 alpha=0.7)
        if MC_MODE == 'median_pc':
            plt.fill_between(preds_src["Date"],
                             preds_src["pred_pc5"],
                             preds_src["pred_pc95"],
                             color="orange", alpha=0.3)
        else:
            plt.fill_between(preds_src["Date"],
                             preds_src["pred_kwh"] - preds_src["pred_std"],
                             preds_src["pred_kwh"] + preds_src["pred_std"],
                             color="orange", alpha=0.3)
        plt.plot(preds_src["Date"], preds_src["pred_kwh"],
                 label="Predicted (extra month)", linestyle="--", alpha=0.8)
        plt.title(f"Electricity Consumption - device {src[16:18].strip('_')},"
                  + f" user {src[-2:].lstrip('_')}", fontsize='large')
        plt.xlabel("Date", fontsize='large')
        plt.ylabel("(kWh)", fontsize='large')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.legend(fontsize='large')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,
                                 f"forecast_device_{src[16:18].strip('_')}"
                                 + f"_user_{src[-2:].lstrip('_')}_"
                                 + f"{MC_MODE}.png"))
        plt.close()

    # --- Plot total demand
    df_total = df.groupby("Date")["kwh"].sum().reset_index()
    if MC_MODE == 'median_pc':
        preds_total = preds_df.groupby("Date")[["pred_kwh",
                                                "pred_pc95",
                                                "pred_pc5"]].sum().reset_index()
    else:
        preds_total = preds_df.groupby("Date")[["pred_kwh",
                                                "pred_std"]].sum().reset_index()

    plt.figure(figsize=(12, 5))
    plt.plot(df_total["Date"], df_total["kwh"], label="Actual Total (history)",
             alpha=0.7)
    if MC_MODE == 'median_pc':
        plt.fill_between(preds_total["Date"],
                         preds_total["pred_pc5"],
                         preds_total["pred_pc95"],
                         color="orange", alpha=0.3)
    else:
        plt.fill_between(preds_total["Date"],
                         preds_total["pred_kwh"] - preds_total["pred_std"],
                         preds_total["pred_kwh"] + preds_total["pred_std"],
                         color="orange", alpha=0.3)
    plt.plot(preds_total["Date"], preds_total["pred_kwh"],
             label="Predicted Total (extra month)", linestyle="--", alpha=0.8)
    plt.title("Total Electricity Consumption", fontsize='large')
    plt.xlabel("Date", fontsize='large')
    plt.ylabel("(kWh)", fontsize='large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"forecast_total_{MC_MODE}.png"))
    plt.close()

    # --- Plot total demand zoomed in around extra month
    plt.figure(figsize=(12, 5))
    plt.plot(df_total["Date"], df_total["kwh"], label="Actual Total (history)",
             alpha=0.7)
    if MC_MODE == 'median_pc':
        plt.fill_between(preds_total["Date"],
                         preds_total["pred_pc5"],
                         preds_total["pred_pc95"],
                         color="orange", alpha=0.3)
    else:
        plt.fill_between(preds_total["Date"],
                         preds_total["pred_kwh"] - preds_total["pred_std"],
                         preds_total["pred_kwh"] + preds_total["pred_std"],
                         color="orange", alpha=0.3)
    plt.plot(preds_total["Date"], preds_total["pred_kwh"],
             label="Predicted Total (extra month)", linestyle="--", alpha=0.8)
    plt.title("Total Electricity Consumption", fontsize='large')
    plt.xlabel("Date", fontsize='large')
    plt.ylabel("(kWh)", fontsize='large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.xlim(pd.to_datetime("2024-06-24"), pd.to_datetime("2024-10-24"))
    plt.savefig(os.path.join(PLOTS_DIR, f"forecast_total_window_{MC_MODE}.png"))
    plt.close()


# -------------------------------
# 8. Run
# -------------------------------
if __name__ == "__main__":

    run_pipeline()
