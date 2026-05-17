"""
Synthetic Server Metrics Generator for Sandbox MMO
Simulates realistic hourly telemetry for a persistent multiplayer server.
Includes trends, seasonality (daily/weekly), events, and injected anomalies (botting, DDoS, viral spikes).
Reproducible with seed.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_synthetic_metrics(n_days=90, freq="h", seed=42, config=None):
    if config is None:
        config = load_config()
    
    np.random.seed(seed)
    synth_cfg = config['data']['synthetic']
    
    n_hours = n_days * 24
    start_time = datetime(2026, 1, 1, 0, 0)
    timestamps = pd.date_range(start=start_time, periods=n_hours, freq=freq)
    
    # Base parameters
    base_players = synth_cfg['base_players']
    growth_rate = synth_cfg['growth_rate']  # daily growth
    max_cap = synth_cfg['max_capacity_per_server']
    anomaly_rate = synth_cfg['anomaly_rate']
    
    # Time features
    hours = np.arange(n_hours)
    hour_of_day = (timestamps.hour).values
    day_of_week = (timestamps.dayofweek).values
    is_weekend = (day_of_week >= 5).astype(int)
    
    # Trend: slow growth in player base
    trend = base_players + growth_rate * (hours / 24)
    
    # Daily seasonality: peak evenings 18-23h, low morning
    daily_pattern = 40 * np.sin(2 * np.pi * (hour_of_day - 20) / 24) + 20 * np.cos(2 * np.pi * hour_of_day / 24)
    daily_pattern = np.clip(daily_pattern, -30, 80)
    
    # Weekly: weekends +30%
    weekly_boost = 25 * is_weekend
    
    # Random walk noise for realism
    noise = np.cumsum(np.random.normal(0, 3, n_hours)) * 0.1
    noise = np.clip(noise, -15, 15)
    
    # Concurrent players
    concurrent_players = trend + daily_pattern + weekly_boost + noise
    concurrent_players = np.clip(concurrent_players, 20, max_cap * 1.1).astype(int)
    
    # CPU usage: strongly correlated with players + base + spikes
    cpu_base = 35 + (concurrent_players / max_cap) * 45
    cpu_noise = np.random.normal(0, 4, n_hours)
    cpu_usage = np.clip(cpu_base + cpu_noise, 10, 98)
    
    # Memory: slower growth, correlated
    memory_usage = 8 + (concurrent_players / max_cap) * 24 + np.random.normal(0, 1.5, n_hours)
    memory_usage = np.clip(memory_usage, 6, 48)
    
    # Network: in/out proportional to players + some variance
    network_in = (concurrent_players * 0.8 + np.random.normal(0, 20, n_hours)).clip(50, 800)
    network_out = (concurrent_players * 0.6 + np.random.normal(0, 15, n_hours)).clip(40, 600)
    
    # Tick rate (game server specific): drops under high load
    tick_rate = 60 - (cpu_usage - 60).clip(0, 30) * 0.8 + np.random.normal(0, 1.5, n_hours)
    tick_rate = np.clip(tick_rate, 20, 62).astype(int)
    
    # Create DF
    df = pd.DataFrame({
        'timestamp': timestamps,
        'concurrent_players': concurrent_players,
        'cpu_usage_percent': cpu_usage.round(1),
        'memory_usage_gb': memory_usage.round(1),
        'network_in_mbps': network_in.round(1),
        'network_out_mbps': network_out.round(1),
        'tick_rate': tick_rate,
        'hour': hour_of_day,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend
    })
    
    # Inject anomalies (3%)
    n_anom = int(n_hours * anomaly_rate)
    anom_indices = np.random.choice(n_hours, n_anom, replace=False)
    
    anomaly_types = []
    for idx in anom_indices:
        r = np.random.rand()
        if r < 0.35:  # DDoS / high load attack
            df.loc[idx, 'cpu_usage_percent'] = min(98, df.loc[idx, 'cpu_usage_percent'] + 35)
            df.loc[idx, 'network_in_mbps'] = df.loc[idx, 'network_in_mbps'] * 2.5
            df.loc[idx, 'tick_rate'] = max(15, df.loc[idx, 'tick_rate'] - 20)
            anomaly_types.append('ddos_spike')
        elif r < 0.6:  # Botting / high players low engagement (high cpu relative? or sudden player influx low quality)
            df.loc[idx, 'concurrent_players'] = min(int(max_cap * 1.05), int(df.loc[idx, 'concurrent_players'] * 1.4))
            df.loc[idx, 'cpu_usage_percent'] = min(95, df.loc[idx, 'cpu_usage_percent'] + 15)
            anomaly_types.append('bot_influx')
        elif r < 0.85:  # Viral event or unexpected peak
            df.loc[idx, 'concurrent_players'] = min(int(max_cap * 1.1), int(df.loc[idx, 'concurrent_players'] * 1.6 + 40))
            df.loc[idx, 'cpu_usage_percent'] = min(97, df.loc[idx, 'cpu_usage_percent'] + 25)
            anomaly_types.append('viral_spike')
        else:  # Churn / sudden drop (e.g. bad patch)
            df.loc[idx, 'concurrent_players'] = max(15, int(df.loc[idx, 'concurrent_players'] * 0.35))
            df.loc[idx, 'cpu_usage_percent'] = max(20, df.loc[idx, 'cpu_usage_percent'] - 25)
            anomaly_types.append('sudden_drop')
    
    df['anomaly'] = 0
    df.loc[anom_indices, 'anomaly'] = 1
    df['anomaly_type'] = 'normal'
    for i, idx in enumerate(anom_indices):
        df.loc[idx, 'anomaly_type'] = anomaly_types[i]
    
    # Add server_id for future multi-server (here single for MVP)
    df['server_id'] = 'realm1-shard1'
    df['realm'] = 'alpha-realm-1'
    
    return df

if __name__ == "__main__":
    config = load_config("../config.yaml")
    df = generate_synthetic_metrics(config=config)
    os.makedirs("../data", exist_ok=True)
    df.to_csv("../data/synthetic_server_metrics.csv", index=False)
    print(f"Generated {len(df)} hourly records with {df['anomaly'].sum()} anomalies.")
    print(df.head())
    print("\nAnomaly distribution:")
    print(df['anomaly_type'].value_counts())