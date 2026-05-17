"""
Feature Engineering for Server Load Forecasting & Anomaly Detection.

- Time-based features (cyclical encoding for hour/dow)
- Lag & rolling statistics
- Domain-specific ratios (player density, efficiency)
- Prepares for Prophet (univariate) and LSTM (multivariate sequences)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encodings and flag columns to *df*."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 6)).astype(int)
    df["is_peak"] = ((df["hour"] >= 18) & (df["hour"] <= 23)).astype(int)
    return df


def add_lag_rolling_features(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Append lag and rolling-window statistics for *target_cols*.

    Args:
        df: Input dataframe sorted by timestamp on return.
        target_cols: Columns to build lag/rolling features from.
        lags: Lag offsets in rows (hours for hourly data).
        windows: Rolling window sizes in rows.

    Returns:
        DataFrame with new lag/rolling columns appended.
    """
    if target_cols is None:
        target_cols = ["concurrent_players", "cpu_usage_percent"]
    if lags is None:
        lags = [1, 6, 24, 48]
    if windows is None:
        windows = [6, 24, 48]

    df = df.sort_values("timestamp").copy()
    for col in target_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for w in windows:
            df[f"{col}_rollmean{w}"] = df[col].shift(1).rolling(w).mean()
            df[f"{col}_rollstd{w}"] = df[col].shift(1).rolling(w).std()
    return df


def add_domain_features(df: pd.DataFrame, max_capacity: int = 250) -> pd.DataFrame:
    """Add MMO-specific derived features (efficiency ratios, spike detection).

    Args:
        df: Input dataframe.
        max_capacity: Maximum players per server shard, used for density ratio.

    Returns:
        DataFrame with domain feature columns appended.
    """
    df = df.copy()
    df["player_density"] = df["concurrent_players"] / max_capacity
    df["cpu_per_player"] = df["cpu_usage_percent"] / (df["concurrent_players"] + 1)
    df["load_efficiency"] = df["tick_rate"] / (df["cpu_usage_percent"] + 1) * 10
    df["network_per_player"] = (df["network_in_mbps"] + df["network_out_mbps"]) / (
        df["concurrent_players"] + 1
    )
    # Z-score of CPU relative to its own recent 6h window — captures sudden spikes.
    df["cpu_spike"] = (
        df["cpu_usage_percent"] - df["cpu_usage_percent"].rolling(6).mean()
    ) / (df["cpu_usage_percent"].rolling(6).std() + 1e-6)
    return df


def prepare_prophet_df(
    df: pd.DataFrame,
    target: str = "concurrent_players",
) -> pd.DataFrame:
    """Format a dataframe for Prophet with required regressors.

    Args:
        df: Source dataframe containing timestamp, target and regressor columns.
        target: Name of the target column (mapped to Prophet's ``y``).

    Returns:
        Prophet-ready dataframe with columns ``ds``, ``y``, ``is_weekend``,
        ``hour_sin``, ``dow_sin``.
    """
    prophet_df = df[["timestamp", target]].rename(
        columns={"timestamp": "ds", target: "y"}
    )
    prophet_df["is_weekend"] = df["is_weekend"]
    prophet_df["hour_sin"] = df["hour_sin"]
    prophet_df["dow_sin"] = df["dow_sin"]
    return prophet_df.dropna()


def create_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "concurrent_players",
    seq_length: int = 48,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a dataframe into overlapping (X, y) windows for LSTM training.

    Args:
        df: Source dataframe sorted by timestamp.
        feature_cols: Input feature column names.
        target_col: Target column name.
        seq_length: Number of past timesteps fed into the model.
        horizon: Number of future timesteps to predict.

    Returns:
        Tuple of ``(X, y)`` arrays with shapes
        ``(n_samples, seq_length, n_features)`` and
        ``(n_samples, horizon)``.
    """
    df = df.sort_values("timestamp").copy()
    data = df[feature_cols + [target_col]].values
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i : i + seq_length, :-1])
        y.append(data[i + seq_length : i + seq_length + horizon, -1])
    return np.array(X), np.array(y)


def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Apply StandardScaler to *feature_cols* in-place.

    Args:
        df: DataFrame to scale (modified in-place).
        feature_cols: Columns to normalise.
        scaler: Pre-fitted scaler; created if ``None``.
        fit: Whether to fit the scaler on this data.

    Returns:
        Tuple of ``(df, scaler)``.
    """
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


if __name__ == "__main__":
    from data_ingestion import ingest_data

    sample_df = ingest_data("synthetic")
    sample_df = create_time_features(sample_df)
    sample_df = add_lag_rolling_features(sample_df)
    sample_df = add_domain_features(sample_df)
    print("Features created. Sample cols:", sample_df.columns.tolist()[:15])
    print(sample_df[["concurrent_players", "player_density", "cpu_per_player"]].describe())
