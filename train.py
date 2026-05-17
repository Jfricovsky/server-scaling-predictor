"""
One-command training pipeline for Server Scaling Predictor.
Generates synthetic data, engineers features, trains Prophet + LSTM + Anomaly models,
evaluates, and persists artifacts + metadata.
Run: python train.py
"""

import os
import sys
import yaml
import json
import pandas as pd
from datetime import datetime

# Ensure project root and src/ are importable (robust on Windows + venv)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.data_ingestion import ingest_data
from src.forecasting import train_prophet, train_lstm
from src.anomaly_detection import train_anomaly_model
from src.feature_engineering import create_time_features, add_domain_features

def main():
    print("=" * 60)
    print("AI-Powered Server Scaling Predictor - Training Pipeline")
    print("=" * 60)
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 1. Data
    print("\n[1/4] Generating/Loading synthetic telemetry...")
    df = ingest_data("synthetic", config_path="config.yaml")
    df.to_csv("data/synthetic_server_metrics.csv", index=False)
    print(f"    Saved {len(df)} records to data/synthetic_server_metrics.csv")
    
    # 2. Quick feature validation
    df = create_time_features(df)
    df = add_domain_features(df)
    print(f"    Engineered features. Sample: {df.columns.tolist()[:8]}...")
    
    # 3. Train models
    print("\n[2/4] Training Prophet (univariate player forecast)...")
    p_model, p_metrics = train_prophet(df, config)
    
    print("\n[3/4] Training LSTM (multivariate next-hour forecast)...")
    l_model, l_metrics, _ = train_lstm(df, config)
    
    print("\n[4/4] Training Anomaly Detector (Isolation Forest + rules)...")
    a_model, a_feats = train_anomaly_model(df, config)
    
    # Metadata
    meta = {
        "project": config['project']['name'],
        "trained_at": datetime.now().isoformat(),
        "n_records": len(df),
        "prophet": p_metrics or {"status": "skipped (prophet not installed)"},
        "lstm": l_metrics,
        "anomaly": {"contamination": config['models']['anomaly']['contamination'], "features": a_feats},
        "synthetic_params": config['data']['synthetic']
    }
    
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete! Artifacts saved to models/")
    print(json.dumps(meta, indent=2))
    print("\nNext: streamlit run dashboard/app.py")
    print("Scaling architecture ready (dry-run mode). See src/scaling_executor.py")
    print("=" * 60)

if __name__ == "__main__":
    main()