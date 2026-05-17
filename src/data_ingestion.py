"""
Data Ingestion Module.

- Synthetic generator wrapper for reproducible offline development
- Placeholder branches for real VPS / Prometheus / Hetzner Cloud API ingestion
- Supports CSV, or live polling (future)
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_generator import generate_synthetic_metrics, load_config


def ingest_data(
    source: str = "synthetic",
    config_path: str = "config.yaml",
    seed: int | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Load server metrics from the requested source.

    Args:
        source: One of ``'synthetic'``, ``'csv'``, ``'hetzner'``,
            or ``'prometheus'``.
        config_path: Path to ``config.yaml`` (used by the synthetic generator).
        seed: Random seed for synthetic generation.  Defaults to the value of
            ``project.seed`` in ``config.yaml`` (42).  Override to generate a
            different telemetry timeline without changing the config file.
        **kwargs: Extra options forwarded to source-specific handlers.
            For ``'csv'``: ``path`` (str) overrides the default CSV location.

    Returns:
        DataFrame with server telemetry columns.

    Raises:
        ValueError: If *source* is not a recognised value.
    """
    config = load_config(config_path)

    if source == "synthetic":
        _seed = seed if seed is not None else config.get("project", {}).get("seed", 42)
        df = generate_synthetic_metrics(config=config, seed=_seed)
        print(f"[INGEST] Generated synthetic data: {len(df)} rows, {df['anomaly'].sum()} labeled anomalies (seed={_seed})")
        return df

    if source == "csv":
        path: str = kwargs.get("path", "data/synthetic_server_metrics.csv")  # type: ignore[assignment]
        df = pd.read_csv(path, parse_dates=["timestamp"])
        print(f"[INGEST] Loaded {len(df)} rows from {path}")
        return df

    if source == "hetzner":
        # Placeholder: In production, use Hetzner Cloud API + node exporter or a custom agent.
        # Example:
        #   import requests
        #   resp = requests.get(
        #       f"https://api.hetzner.cloud/v1/servers/{server_id}/metrics",
        #       params={"type": "cpu", "start": start, "end": end},
        #       headers={"Authorization": f"Bearer {os.environ['HETZNER_TOKEN']}"},
        #   )
        print("[INGEST] Hetzner API placeholder — implement with your token and server list")
        return generate_synthetic_metrics(config=config)

    if source == "prometheus":
        # Example query:
        #   from prometheus_api_client import PrometheusConnect
        #   prom = PrometheusConnect(url=os.environ["PROMETHEUS_URL"])
        #   prom.custom_query("avg_over_time(cpu_usage[5m])")
        print("[INGEST] Prometheus placeholder — connect to your game server Prometheus instance")
        return generate_synthetic_metrics(config=config)

    raise ValueError(f"Unknown source: {source!r}. Choose from: synthetic, csv, hetzner, prometheus")


def get_latest_metrics(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """Return the most recent *hours* rows for dashboard live view.

    Args:
        df: Full telemetry dataframe.
        hours: Number of most recent rows to return.

    Returns:
        Tail of *df* sorted by timestamp.
    """
    return df.sort_values("timestamp").tail(hours)


if __name__ == "__main__":
    sample = ingest_data("synthetic")
    print(sample[["timestamp", "concurrent_players", "cpu_usage_percent", "anomaly_type"]].tail(10))
