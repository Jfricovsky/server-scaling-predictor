"""
Forecasting Module: Prophet (univariate with seasonality) + PyTorch LSTM (multivariate).

Saves trained models and evaluation metadata to ``models/`` for dashboard inference.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not installed. Using LSTM only.")

from src.feature_engineering import (
    add_domain_features,
    add_lag_rolling_features,
    create_lstm_sequences,
    create_time_features,
    prepare_prophet_df,
)

_DEFAULT_LSTM_FEATURES: list[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "player_density",
    "cpu_usage_percent",
    "memory_usage_gb",
    "network_in_mbps",
    "tick_rate",
    "cpu_per_player",
]


class LSTMForecaster(nn.Module):
    """Two-layer LSTM with a linear head for scalar time-series forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_prophet(
    df: pd.DataFrame,
    config: dict,
    target: str = "concurrent_players",
    save_path: str = "models/prophet_model.pkl",
) -> tuple[object | None, dict]:
    """Fit a Prophet model with regressors and evaluate on a 7-day holdout.

    Args:
        df: Full historical dataframe.
        config: Parsed ``config.yaml`` dict.
        target: Column name to forecast.
        save_path: Where to persist the fitted model.

    Returns:
        Tuple of ``(prophet_model, metrics_dict)``.  Returns
        ``(None, {})`` if Prophet is not installed.
    """
    if not PROPHET_AVAILABLE:
        print("Skipping Prophet training (not installed).")
        return None, {}

    # Apply time features so prepare_prophet_df can access hour_sin / dow_sin
    # regardless of whether the caller already did feature engineering.
    df = create_time_features(df)
    prophet_df = prepare_prophet_df(df, target)
    train_df = prophet_df[
        prophet_df["ds"] < prophet_df["ds"].max() - timedelta(days=7)
    ]

    model = Prophet(
        seasonality_mode=config["models"]["prophet"]["seasonality_mode"],
        changepoint_prior_scale=config["models"]["prophet"]["changepoint_prior_scale"],
        seasonality_prior_scale=config["models"]["prophet"]["seasonality_prior_scale"],
        holidays_prior_scale=config["models"]["prophet"]["holidays_prior_scale"],
        interval_width=config["models"]["prophet"]["interval_width"],
        daily_seasonality=True,
        weekly_seasonality=True,
    )
    model.add_regressor("is_weekend")
    model.add_regressor("hour_sin")
    model.add_regressor("dow_sin")
    model.fit(train_df)

    future = model.make_future_dataframe(periods=24 * 7, freq="h")
    future["is_weekend"] = (future["ds"].dt.dayofweek >= 5).astype(int)
    future["hour_sin"] = np.sin(2 * np.pi * future["ds"].dt.hour / 24)
    future["dow_sin"] = np.sin(2 * np.pi * future["ds"].dt.dayofweek / 7)
    forecast = model.predict(future)

    # Robust evaluation: align on the last 168 samples of the held-out window.
    y_true = prophet_df["y"].tail(168).values
    y_pred = forecast["yhat"].tail(168).values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)

    metrics = {
        "prophet_mae": round(mae, 2),
        "prophet_rmse": round(rmse, 2),
        "prophet_mape": round(mape, 2),
    }
    print(f"Prophet metrics (7-day holdout): MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")

    joblib.dump(model, save_path)
    return model, metrics


def train_lstm(
    df: pd.DataFrame,
    config: dict,
    feature_cols: list[str] | None = None,
    target: str = "concurrent_players",
    save_path: str = "models/lstm_model.pt",
) -> tuple[LSTMForecaster, dict, list[str]]:
    """Train the LSTM forecaster and save the checkpoint.

    Args:
        df: Full historical dataframe.
        config: Parsed ``config.yaml`` dict.
        feature_cols: Override the default LSTM input features.
        target: Target column name.
        save_path: Where to persist the PyTorch checkpoint.

    Returns:
        Tuple of ``(model, metrics_dict, feature_cols_used)``.
    """
    if feature_cols is None:
        feature_cols = _DEFAULT_LSTM_FEATURES

    # Reproducibility
    seed: int = config["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = create_time_features(df)
    df = add_lag_rolling_features(df)
    df = add_domain_features(df)
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    # ── Feature + target scaling (fit on train rows only to prevent leakage) ─
    # Different features span very different ranges (hour_sin: −1..1 vs
    # cpu_usage_percent: 0..100 vs concurrent_players: ~80..300).  Without
    # normalisation the LSTM implicitly weights high-magnitude columns more.
    # The target is also scaled so MSE loss stays in a well-conditioned range
    # (~0–4) rather than thousands of squared player units; we inverse-transform
    # predictions before computing MAE so reported numbers stay interpretable.
    split_row = int(len(df) * config["data"]["train_test_split"])
    scaler = StandardScaler()
    target_scaler = StandardScaler()

    df_scaled = df.copy()
    # Cast to float64 to avoid pandas dtype-incompatibility FutureWarning
    df_scaled[feature_cols] = df_scaled[feature_cols].astype(float)
    df_scaled[target] = df_scaled[target].astype(float)

    df_scaled.loc[df_scaled.index[:split_row], feature_cols] = (
        scaler.fit_transform(df[feature_cols].iloc[:split_row])
    )
    df_scaled.loc[df_scaled.index[split_row:], feature_cols] = (
        scaler.transform(df[feature_cols].iloc[split_row:])
    )
    df_scaled.loc[df_scaled.index[:split_row], target] = (
        target_scaler.fit_transform(df[[target]].iloc[:split_row])
    )
    df_scaled.loc[df_scaled.index[split_row:], target] = (
        target_scaler.transform(df[[target]].iloc[split_row:])
    )

    seq_length: int = config["models"]["lstm"]["seq_length"]
    X, y = create_lstm_sequences(df_scaled, feature_cols, target, seq_length=seq_length, horizon=1)

    split = int(len(X) * config["data"]["train_test_split"])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # shuffle=False preserves temporal ordering — critical for time-series.
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=config["models"]["lstm"]["batch_size"],
        shuffle=False,
    )

    model = LSTMForecaster(
        input_size=X.shape[2],
        hidden_size=config["models"]["lstm"]["hidden_size"],
        num_layers=config["models"]["lstm"]["num_layers"],
        dropout=config["models"]["lstm"]["dropout"],
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["models"]["lstm"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    lstm_cfg = config["models"]["lstm"]
    patience: int = lstm_cfg.get("early_stopping_patience", 8)
    grad_clip: float = lstm_cfg.get("grad_clip", 1.0)
    max_epochs: int = lstm_cfg["epochs"]

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    model.train()
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_loss = criterion(model(batch_x), batch_y)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += batch_loss.item()

        # Validation pass (no grad)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test_t), y_test_t).item()
        model.train()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            avg_train = epoch_loss / len(train_loader)
            print(
                f"LSTM Epoch {epoch + 1}/{max_epochs}  "
                f"train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.5f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
            break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy().flatten()
        y_true_scaled = y_test_t.numpy().flatten()

    # Inverse-transform to player-count space for interpretable metrics
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {"lstm_mae": round(mae, 2), "lstm_rmse": round(rmse, 2)}
    print(f"LSTM Test metrics: MAE={mae:.2f}, RMSE={rmse:.2f}")

    torch.save(
        {
            "model_state": best_state,
            "input_size": X.shape[2],
            "feature_cols": feature_cols,
            "seq_length": seq_length,
            "metrics": metrics,
            "scaler": scaler,
            "target_scaler": target_scaler,
        },
        save_path,
    )
    return model, metrics, feature_cols


def load_models(
    models_dir: str = "models",
) -> tuple[object | None, dict | None]:
    """Load persisted Prophet and LSTM models from *models_dir*.

    Args:
        models_dir: Directory containing ``prophet_model.pkl`` and
            ``lstm_model.pt``.

    Returns:
        Tuple of ``(prophet_model, lstm_data_dict)``.  Either element is
        ``None`` if the corresponding artifact does not exist.
    """
    prophet_model: object | None = None
    prophet_path = os.path.join(models_dir, "prophet_model.pkl")
    if os.path.exists(prophet_path):
        prophet_model = joblib.load(prophet_path)

    lstm_data: dict | None = None
    lstm_path = os.path.join(models_dir, "lstm_model.pt")
    if os.path.exists(lstm_path):
        lstm_data = torch.load(lstm_path, map_location="cpu", weights_only=False)
        lstm_model = LSTMForecaster(lstm_data["input_size"])
        lstm_model.load_state_dict(lstm_data["model_state"])
        lstm_model.eval()
        lstm_data["model"] = lstm_model
        # scaler keys may be absent in older checkpoints — degrade gracefully
        lstm_data.setdefault("scaler", None)
        lstm_data.setdefault("target_scaler", None)

    return prophet_model, lstm_data


if __name__ == "__main__":
    from data_ingestion import ingest_data

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    df = ingest_data("synthetic")
    os.makedirs("models", exist_ok=True)

    p_model, p_metrics = train_prophet(df, config)
    l_model, l_metrics, feats = train_lstm(df, config)

    meta = {
        "training_date": pd.Timestamp.now().isoformat(),
        "prophet_metrics": p_metrics,
        "lstm_metrics": l_metrics,
        "n_samples": len(df),
    }
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Models trained and saved.")
