"""
Inference script for the AI-Powered Server Scaling Predictor.

Loads trained models, detects anomalies on the most recent window, runs
a Prophet / LSTM forecast, and prints prioritised recommendations to stdout.

Usage:
    python predict.py --hours 24 --model both
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_ingestion import ingest_data
from src.forecasting import load_models
from src.anomaly_detection import detect_anomalies
from src.recommendation import generate_recommendations


def build_prophet_future(
    prophet_model: object,
    hours: int,
) -> pd.DataFrame:
    """Build a regressor-filled future dataframe for Prophet prediction."""
    future: pd.DataFrame = prophet_model.make_future_dataframe(periods=hours, freq="h")
    future["is_weekend"] = (future["ds"].dt.dayofweek >= 5).astype(int)
    future["hour_sin"] = np.sin(2 * np.pi * future["ds"].dt.hour / 24)
    future["dow_sin"] = np.sin(2 * np.pi * future["ds"].dt.dayofweek / 7)
    return future


def print_forecast(forecast: pd.DataFrame, n_rows: int = 6) -> None:
    """Print the next *n_rows* Prophet predictions."""
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    print(forecast[cols].tail(n_rows).to_string(index=False))


def print_recommendations(recs: list[dict], top_n: int = 3) -> None:
    """Print the top *top_n* recommendations."""
    for rec in recs[:top_n]:
        reason_snippet = rec["reason"][:80]
        print(f"[{rec['priority']}] {rec['action']}: {reason_snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for the server scaling predictor.")
    parser.add_argument("--hours", type=int, default=24, help="Forecast horizon in hours (default: 24)")
    parser.add_argument(
        "--model",
        choices=["prophet", "lstm", "both"],
        default="both",
        help="Which forecasting model to use",
    )
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    df = ingest_data("synthetic")
    prophet_model, lstm_data = load_models()
    anom_df = detect_anomalies(df.tail(200))

    print("\n=== Latest Metrics ===")
    print(df.tail(3)[["timestamp", "concurrent_players", "cpu_usage_percent", "tick_rate"]])

    forecast_df: pd.DataFrame | None = None

    if args.model in ("prophet", "both") and prophet_model is not None:
        print(f"\n=== Prophet Forecast (next {args.hours}h) ===")
        future = build_prophet_future(prophet_model, args.hours)
        forecast_df = prophet_model.predict(future)
        print_forecast(forecast_df, n_rows=6)

    recs = generate_recommendations(df.tail(48), forecast_df, anom_df, config)
    print("\n=== Top Recommendations ===")
    print_recommendations(recs, top_n=3)


if __name__ == "__main__":
    main()
