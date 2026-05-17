"""
Recommendation Engine.

Combines forecast + anomaly signals into actionable ops decisions for MMO
server scaling.  Prioritises cost-aware, low false-positive recommendations
suitable for a solo dev / small team.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def generate_recommendations(
    current_df: pd.DataFrame,
    forecast_df: pd.DataFrame | None,
    anomaly_df: pd.DataFrame,
    config: dict,
    max_capacity: int = 250,
) -> list[dict]:
    """Build a prioritised list of ops recommendations.

    Args:
        current_df: Latest 24–48 h of real server metrics.
        forecast_df: Prophet / LSTM future predictions with columns
            ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.  Pass
            ``None`` to skip forecast-based recommendations.
        anomaly_df: DataFrame with a ``final_anomaly`` flag column.
        config: Parsed ``config.yaml`` dict.
        max_capacity: Players per server shard (used for load %).

    Returns:
        List of recommendation dicts, each containing at minimum
        ``timestamp``, ``type``, ``action``, ``priority``, and ``reason``.
    """
    recs: list[dict] = []
    now = datetime.now()

    # 1. Scaling recommendation from forecast
    if forecast_df is not None and len(forecast_df) > 0:
        future_6h = forecast_df.head(6)
        max_pred: float = future_6h["yhat"].max()
        load_pct: float = (max_pred / max_capacity) * 100

        if load_pct > config["recommendation"]["scale_threshold"] * 100:
            hours_high: int = int(
                (future_6h["yhat"] > max_capacity * config["recommendation"]["scale_threshold"]).sum()
            )
            if hours_high >= config["recommendation"]["min_hours_high"]:
                recs.append({
                    "timestamp": now.isoformat(),
                    "type": "scaling",
                    "action": "SCALE_UP",
                    "priority": "HIGH",
                    "reason": (
                        f"Predicted peak load {load_pct:.0f}% for {hours_high}h. "
                        "Add 1 server shard (capacity +250 players)."
                    ),
                    "predicted_load_pct": round(load_pct, 1),
                    "estimated_monthly_cost_usd": 8,
                    "confidence": "high" if hours_high > 4 else "medium",
                })
        elif load_pct < 40:
            recs.append({
                "timestamp": now.isoformat(),
                "type": "scaling",
                "action": "MONITOR",
                "priority": "LOW",
                "reason": (
                    f"Low predicted load ({load_pct:.0f}%). "
                    "No scaling needed. Consider downsizing if sustained."
                ),
                "predicted_load_pct": round(load_pct, 1),
            })

    # 2. Anomaly alerts
    recent_anoms = anomaly_df[anomaly_df["final_anomaly"] == 1].tail(5)
    for _, row in recent_anoms.iterrows():
        atype: str = row.get("anomaly_type", "unknown")
        if atype == "ddos_spike":
            action, priority = "INVESTIGATE_DDOS", "CRITICAL"
            reason = (
                f"High network + CPU spike at {row['timestamp']}. "
                "Possible DDoS or exploit. Check firewall + player logs."
            )
        elif atype == "bot_influx" or row.get("rule_anomaly", 0) == 1:
            action, priority = "REVIEW_BOTS", "HIGH"
            reason = (
                f"Suspicious high CPU/low engagement pattern at {row['timestamp']}. "
                "Possible botting or macro abuse."
            )
        elif atype == "sudden_drop":
            action, priority = "CHECK_EVENT", "MEDIUM"
            reason = (
                f"Sudden player drop ({row['concurrent_players']}) at {row['timestamp']}. "
                "Investigate patch, event, or ban wave impact."
            )
        else:
            action, priority = "REVIEW_ANOMALY", "MEDIUM"
            reason = (
                f"Behavioural outlier detected (score {row['anomaly_score']:.2f}). "
                "Manual review recommended."
            )

        recs.append({
            "timestamp": (
                row["timestamp"].isoformat()
                if hasattr(row["timestamp"], "isoformat")
                else str(row["timestamp"])
            ),
            "type": "anomaly",
            "action": action,
            "priority": priority,
            "reason": reason,
            "player_count": int(row["concurrent_players"]),
            "cpu": round(row["cpu_usage_percent"], 1),
        })

    # 3. New realm / expansion trigger (disabled by default)
    # Enable via ``recommendation.enable_revenue_check: true`` in config.yaml
    # only once you have real revenue data (e.g. Stripe integration).
    if len(current_df) > 24 * 7:
        avg_daily: float = (
            current_df["concurrent_players"].resample("D", on="timestamp").mean().mean()
        )
        if (
            avg_daily > max_capacity * 0.75
            and config.get("recommendation", {}).get("enable_revenue_check", False)
        ):
            recs.append({
                "timestamp": now.isoformat(),
                "type": "expansion",
                "action": "OPEN_NEW_REALM",
                "priority": "MEDIUM",
                "reason": (
                    f"Avg daily players {avg_daily:.0f} nearing capacity. "
                    "Revenue check enabled — consider new realm if sales justify it."
                ),
                "avg_daily_players": round(avg_daily, 0),
            })

    return recs


def format_recs_for_dashboard(recs: list[dict]) -> pd.DataFrame:
    """Convert recommendations list to a display-friendly DataFrame.

    Args:
        recs: Output of :func:`generate_recommendations`.

    Returns:
        DataFrame ready for ``st.dataframe`` / iteration in the dashboard.
        Returns a single-row "all clear" frame when *recs* is empty.
    """
    if not recs:
        return pd.DataFrame([{
            "action": "All systems nominal",
            "priority": "INFO",
            "reason": "No scaling or anomalies detected in forecast window.",
        }])
    return pd.DataFrame(recs)


if __name__ == "__main__":
    print("Recommendation module ready. Run via dashboard or train.py")
