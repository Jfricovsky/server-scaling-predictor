"""
Smoke and unit tests for the server scaling predictor pipeline.

These tests cover the core data-flow contracts (shapes, column names, types)
without requiring trained models on disk — every assertion can run on a fresh
clone in CI.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

# Ensure project root is importable (mirrors train.py / predict.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    from src.data_ingestion import ingest_data

    return ingest_data("synthetic")


@pytest.fixture(scope="session")
def featured_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    from src.feature_engineering import add_domain_features, create_time_features

    df = create_time_features(raw_df)
    df = add_domain_features(df)
    return df


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

class TestDataIngestion:
    def test_row_count(self, raw_df: pd.DataFrame, config: dict) -> None:
        expected_days = config["data"]["synthetic"]["n_days"]
        expected_rows = expected_days * 24
        assert len(raw_df) == expected_rows, (
            f"Expected {expected_rows} rows for {expected_days} days, got {len(raw_df)}"
        )

    def test_required_columns(self, raw_df: pd.DataFrame) -> None:
        required = {
            "timestamp",
            "concurrent_players",
            "cpu_usage_percent",
            "memory_usage_gb",
            "network_in_mbps",
            "network_out_mbps",
            "tick_rate",
            "anomaly",
        }
        missing = required - set(raw_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_nulls_in_core_columns(self, raw_df: pd.DataFrame) -> None:
        core = ["concurrent_players", "cpu_usage_percent", "timestamp"]
        null_counts = raw_df[core].isnull().sum()
        assert null_counts.sum() == 0, f"Unexpected nulls:\n{null_counts[null_counts > 0]}"

    def test_timestamp_monotonic(self, raw_df: pd.DataFrame) -> None:
        ts = pd.to_datetime(raw_df["timestamp"])
        assert ts.is_monotonic_increasing, "Timestamps are not monotonically increasing"

    def test_unknown_source_raises(self) -> None:
        from src.data_ingestion import ingest_data

        with pytest.raises(ValueError, match="Unknown source"):
            ingest_data("nonexistent_source")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_cyclical_features_range(self, featured_df: pd.DataFrame) -> None:
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert col in featured_df.columns, f"Missing column: {col}"
            assert featured_df[col].between(-1.0, 1.0).all(), (
                f"{col} has values outside [-1, 1]"
            )

    def test_domain_features_created(self, featured_df: pd.DataFrame) -> None:
        for col in ["player_density", "cpu_per_player", "load_efficiency", "cpu_spike"]:
            assert col in featured_df.columns, f"Missing domain feature: {col}"

    def test_player_density_range(self, featured_df: pd.DataFrame) -> None:
        max_capacity = 250
        density = featured_df["player_density"]
        assert (density >= 0).all(), "player_density has negative values"
        assert (density <= 2.0).all(), "player_density exceeds 2× capacity — check data"

    def test_lstm_sequences_shape(self, featured_df: pd.DataFrame) -> None:
        from src.feature_engineering import (
            add_lag_rolling_features,
            create_lstm_sequences,
        )

        df = add_lag_rolling_features(featured_df).dropna()
        feature_cols = ["hour_sin", "hour_cos", "player_density", "cpu_usage_percent"]
        seq_length, horizon = 24, 1
        X, y = create_lstm_sequences(df, feature_cols, seq_length=seq_length, horizon=horizon)

        assert X.ndim == 3, f"Expected 3D X, got shape {X.shape}"
        assert X.shape[1] == seq_length
        assert X.shape[2] == len(feature_cols)
        assert y.ndim == 2
        assert y.shape[1] == horizon


# ---------------------------------------------------------------------------
# Forecasting model
# ---------------------------------------------------------------------------

class TestLSTMForecaster:
    def test_forward_pass(self) -> None:
        import torch

        from src.forecasting import LSTMForecaster

        batch_size, seq_len, n_features = 4, 48, 10
        model = LSTMForecaster(input_size=n_features)
        x = torch.randn(batch_size, seq_len, n_features)
        out = model(x)
        assert out.shape == (batch_size, 1), f"Unexpected output shape: {out.shape}"

    def test_lstm_train_returns_metrics(
        self, featured_df: pd.DataFrame, config: dict
    ) -> None:
        import torch
        from src.feature_engineering import add_lag_rolling_features
        from src.forecasting import train_lstm

        df = add_lag_rolling_features(featured_df).dropna()
        # Use minimal epochs and disable early-stopping patience to keep CI fast.
        fast_config = {
            **config,
            "models": {
                **config["models"],
                "lstm": {
                    **config["models"]["lstm"],
                    "epochs": 3,
                    "early_stopping_patience": 99,
                },
            },
        }
        test_path = os.path.join(PROJECT_ROOT, "models", "lstm_test.pt")
        _, metrics, feat_cols = train_lstm(df, fast_config, save_path=test_path)
        assert "lstm_mae" in metrics
        assert "lstm_rmse" in metrics
        assert isinstance(metrics["lstm_mae"], float)
        assert metrics["lstm_mae"] >= 0

        # Verify both scalers are persisted in the checkpoint
        ckpt = torch.load(test_path, map_location="cpu", weights_only=False)
        assert "scaler" in ckpt, "Feature scaler not saved in LSTM checkpoint"
        assert ckpt["scaler"] is not None
        assert "target_scaler" in ckpt, "Target scaler not saved in LSTM checkpoint"
        assert ckpt["target_scaler"] is not None

        if os.path.exists(test_path):
            os.remove(test_path)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def test_train_and_detect(
        self, featured_df: pd.DataFrame, config: dict
    ) -> None:
        from src.anomaly_detection import detect_anomalies, train_anomaly_model

        save_path = os.path.join(PROJECT_ROOT, "models", "anomaly_test.pkl")
        model, feat_cols = train_anomaly_model(featured_df, config, save_path=save_path)
        assert model is not None
        assert len(feat_cols) > 0

        result = detect_anomalies(featured_df.tail(100), model_path=save_path)
        assert "final_anomaly" in result.columns
        assert "anomaly_score" in result.columns
        assert result["final_anomaly"].isin([0, 1]).all()

        if os.path.exists(save_path):
            os.remove(save_path)


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_returns_list(self, raw_df: pd.DataFrame, config: dict) -> None:
        from src.recommendation import format_recs_for_dashboard, generate_recommendations

        empty_anom = raw_df.tail(48).copy()
        empty_anom["final_anomaly"] = 0
        empty_anom["anomaly_score"] = 0.0
        empty_anom["anomaly_type"] = "none"
        empty_anom["rule_anomaly"] = 0

        recs = generate_recommendations(
            raw_df.tail(48), forecast_df=None, anomaly_df=empty_anom, config=config
        )
        assert isinstance(recs, list)

        df_recs = format_recs_for_dashboard(recs)
        assert isinstance(df_recs, pd.DataFrame)
        assert "action" in df_recs.columns

    def test_empty_recs_returns_all_clear(self, raw_df: pd.DataFrame, config: dict) -> None:
        from src.recommendation import format_recs_for_dashboard

        df = format_recs_for_dashboard([])
        assert len(df) == 1
        assert df.iloc[0]["action"] == "All systems nominal"


# ---------------------------------------------------------------------------
# Scaling executor
# ---------------------------------------------------------------------------

class TestScalingExecutor:
    def test_dry_run_does_not_execute(self) -> None:
        from src.scaling_executor import execute_recommendation

        result = execute_recommendation(
            {"action": "SCALE_UP", "priority": "HIGH", "reason": "test"}, dry_run=True
        )
        assert result["dry_run"] is True
        assert result["executed"] is False
        assert "DRY RUN" in result["message"]

    def test_get_scaling_status_keys(self) -> None:
        from src.scaling_executor import get_scaling_status

        status = get_scaling_status()
        for key in ("active_shards", "total_capacity", "current_load", "auto_scaling_enabled"):
            assert key in status, f"Missing key: {key}"
