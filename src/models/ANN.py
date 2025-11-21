"""
pipeline.py

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
DATA_CONSUMPTION = "Data.csv"
DATA_CLIMATE = "Kalam Climate Data.xlsx"
DATA_SUBMISSION = "SampleSubmission.csv"
OUTPUT_SUBMISSION = "MySubmission_final_MLP.csv"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

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
    df = df.copy()
    df["Temp_dew_diff"] = df["Temp_Mean"] - df["Dewpoint_Mean"]
    df["wind_speed"] = np.sqrt(df["U_Wind_Mean"]**2 + df["V_Wind_Mean"]**2)
    df["precip_snow_ratio"] = (df["Precipitation_Sum"]
                               / (df["Snowfall_Sum"] + 1e-6))
    return df


def add_lag_roll(df, group_col="Source", target_col="kwh",
                 lags=[1, 2, 7, 14], windows=[3, 7, 14]):
    """Add lag and rolling mean features per Source."""
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
        layers.Dense(128, activation="relu", input_dim=input_dim),
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
def permutation_importance_manual(model, X_val, y_val, feature_names,
                                  n_repeats=5):
    """Compute permutation importance manually for a Keras model."""
    baseline = mean_squared_error(
        y_val, np.maximum(0, model.predict(X_val).flatten()), squared=False
        )
    importances = {}

    for i, col in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X_val.copy()
            np.random.shuffle(X_permuted[:, i])  # shuffle one column
            y_pred = np.maximum(0, model.predict(X_permuted).flatten())
            score = mean_squared_error(y_val, y_pred, squared=False)
            scores.append(score - baseline)  # increase in RMSE
        importances[col] = np.mean(scores)

    return (
        pd.DataFrame(
            {"feature": list(importances.keys()),
             "importance": list(importances.values())}
             ).sort_values("importance", ascending=False)
        )

# -------------------------------
# 7. Pipeline
# -------------------------------
def run_pipeline():
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

    feature_cols = ([
        "Temp_Mean", "Temp_Min", "Temp_Max", "Dewpoint_Mean", "Dewpoint_Min",
        "Dewpoint_Max", "U_Wind_Mean", "V_Wind_Mean", "Precipitation_Sum",
        "Snowfall_Sum", "SnowCover_Mean", "Temp_dew_diff", "wind_speed",
        "precip_snow_ratio"] + 
        [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_")]
        )

    X, y = df[feature_cols].values, df["kwh"].values

    # --- Scale & split
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

    # --- Train model
    model = build_mlp(X_train.shape[1])
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                 restore_best_weights=True, verbose=1)
    lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=100, batch_size=256,
                        callbacks=[es, lr], verbose=1)

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

    # --- Feature importance
    importances_df = permutation_importance_manual(model, X_test, y_test,
                                                   feature_cols, n_repeats=5)
    print(importances_df.head(20))

    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=importances_df.head(20),
                palette="viridis")
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

            X_ex = pd.DataFrame([r])[feature_cols].fillna(0)
            X_ex = scaler.transform(X_ex)
            r["pred_kwh"] = max(0, model.predict(X_ex)[0, 0])

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

    # --- Plots (extra month, 20 random sources + total)
    sample_sources = np.random.choice(df["Source"].unique(), 20, replace=False)
    for src in sample_sources:
        df_src = df[df["Source"] == src].copy()
        preds_src = preds_df[preds_df["Source"] == src].copy()

        plt.figure(figsize=(12, 5))
        plt.plot(df_src["Date"], df_src["kwh"], label="Actual (history)",
                 alpha=0.7)
        plt.plot(preds_src["Date"], preds_src["pred_kwh"],
                 label="Predicted (extra month)", linestyle="--", alpha=0.8)
        plt.title(f"Source: {src}")
        plt.xlabel("Date"); plt.ylabel("kWh")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"forecast_source_{src}.png"))
        plt.close()

    # Total demand
    df_total = df.groupby("Date")["kwh"].sum().reset_index()
    preds_total = preds_df.groupby("Date")["pred_kwh"].sum().reset_index()

    plt.figure(figsize=(12, 5))
    plt.plot(df_total["Date"], df_total["kwh"], label="Actual Total (history)",
             alpha=0.7)
    plt.plot(preds_total["Date"], preds_total["pred_kwh"],
             label="Predicted Total (extra month)", linestyle="--", alpha=0.8)
    plt.title("Total Electricity Demand")
    plt.xlabel("Date"); plt.ylabel("Total kWh")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "forecast_total.png"))
    plt.close()

# -------------------------------
# 8. Run
# -------------------------------
if __name__ == "__main__":
    run_pipeline()
