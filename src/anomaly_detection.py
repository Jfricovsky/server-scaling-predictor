"""
Anomaly Detection Module.

- Isolation Forest (sklearn) for multivariate behavioural outliers
- Statistical rules for game-specific anomalies:
    - High CPU + low players → possible botting / exploit
    - Sudden >40 % player drop after high CPU → patch event or ban wave
- Can operate on forecast residuals or raw features
"""

from __future__ import annotations

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.feature_engineering import create_time_features, add_domain_features

_DEFAULT_FEATURE_COLS: list[str] = [
    "cpu_usage_percent",
    "concurrent_players",
    "memory_usage_gb",
    "network_in_mbps",
    "tick_rate",
    "player_density",
    "cpu_per_player",
    "cpu_spike",
]


def train_anomaly_model(
    df: pd.DataFrame,
    config: dict,
    feature_cols: list[str] | None = None,
    save_path: str = "models/anomaly_model.pkl",
) -> tuple[IsolationForest, list[str]]:
    """Fit an Isolation Forest on *df* and persist the bundle to *save_path*.

    Args:
        df: Training dataframe (pre-feature-engineering is fine; features are
            computed internally).
        config: Parsed ``config.yaml`` dict.
        feature_cols: Override the default set of anomaly features.
        save_path: Where to write the joblib bundle.

    Returns:
        Tuple of ``(fitted_model, feature_cols_used)``.
    """
    if feature_cols is None:
        feature_cols = _DEFAULT_FEATURE_COLS

    df = create_time_features(df)
    df = add_domain_features(df)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values
    contamination: float = config["models"]["anomaly"]["contamination"]

    model = IsolationForest(
        n_estimators=config["models"]["anomaly"]["n_estimators"],
        contamination=contamination,
        random_state=config["project"]["seed"],
        n_jobs=-1,
    )
    model.fit(X)

    preds = model.predict(X)

    if "anomaly" in df.columns:
        from sklearn.metrics import precision_recall_fscore_support

        y_true = df["anomaly"].values
        y_pred = (preds == -1).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        print(
            f"Anomaly Detection (synthetic labels): "
            f"Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}"
        )

    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "contamination": contamination},
        save_path,
    )
    return model, feature_cols


def detect_anomalies(
    df: pd.DataFrame,
    model_path: str = "models/anomaly_model.pkl",
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score rows in *df* and apply game-specific rule overrides.

    Args:
        df: Recent telemetry window to evaluate.
        model_path: Path to the persisted Isolation Forest bundle.
        threshold: Unused override slot (reserved for future score-based
            threshold tuning).

    Returns:
        *df* augmented with ``anomaly_score``, ``is_anomaly``,
        ``rule_anomaly``, and ``final_anomaly`` columns.
    """
    bundle = joblib.load(model_path)
    model: IsolationForest = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]

    df = create_time_features(df)
    df = add_domain_features(df)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values
    scores = model.decision_function(X)
    preds = model.predict(X)

    df["anomaly_score"] = scores
    df["is_anomaly"] = (preds == -1).astype(int)

    df["rule_anomaly"] = 0
    # Rule 1: High CPU + low players (possible bot farm or exploit)
    mask_bots = (df["cpu_usage_percent"] > 85) & (
        df["concurrent_players"] < df["concurrent_players"].quantile(0.3)
    )
    df.loc[mask_bots, "rule_anomaly"] = 1
    # Rule 2: Sudden player drop >40 % from previous hour with preceding high CPU
    df["player_drop"] = df["concurrent_players"].pct_change() < -0.4
    mask_drop = df["player_drop"] & (df["cpu_usage_percent"].shift(1) > 70)
    df.loc[mask_drop, "rule_anomaly"] = 1

    df["final_anomaly"] = ((df["is_anomaly"] == 1) | (df["rule_anomaly"] == 1)).astype(int)
    return df


if __name__ == "__main__":
    from data_ingestion import ingest_data

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    df = ingest_data("synthetic")
    os.makedirs("models", exist_ok=True)
    model, feats = train_anomaly_model(df, config)
    print("Anomaly model trained.")
