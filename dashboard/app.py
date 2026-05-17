"""
Streamlit Dashboard for AI-Powered Server Scaling Predictor
Run with: streamlit run dashboard/app.py
Features: Real-time metrics, interactive forecasts, anomaly alerts, actionable recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import yaml
import json

# Add project root to path so 'src' package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion import ingest_data, get_latest_metrics
from src.forecasting import load_models, train_prophet, train_lstm
from src.anomaly_detection import detect_anomalies, train_anomaly_model
from src.recommendation import generate_recommendations, format_recs_for_dashboard
from src.feature_engineering import create_time_features, add_domain_features
from src.scaling_executor import execute_recommendation, get_scaling_status

st.set_page_config(page_title="Server Scaling Predictor | MMO Ops", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Dark theme friendly
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .metric-card { background-color: #1e2533; padding: 1rem; border-radius: 0.5rem; }
    .high { color: #ff4b4b; font-weight: bold; }
    .medium { color: #ffa500; }
    .low { color: #00c853; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)

@st.cache_data(ttl=3600)
def load_all_data(seed: int = 42) -> pd.DataFrame:
    return ingest_data("synthetic", seed=seed)

@st.cache_resource
def get_trained_models():
    config = load_config()
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    missing_prophet = not os.path.exists(os.path.join(models_dir, "prophet_model.pkl"))
    missing_lstm = not os.path.exists(os.path.join(models_dir, "lstm_model.pt"))
    missing_anomaly = not os.path.exists(os.path.join(models_dir, "anomaly_model.pkl"))

    if missing_prophet or missing_lstm or missing_anomaly:
        st.info("Training models on first run... (takes 1-2 min)")
        df = load_all_data()
        if missing_prophet or missing_lstm:
            train_prophet(df, config, save_path=os.path.join(models_dir, "prophet_model.pkl"))
            train_lstm(df, config, save_path=os.path.join(models_dir, "lstm_model.pt"))
        if missing_anomaly:
            train_anomaly_model(df, config, save_path=os.path.join(models_dir, "anomaly_model.pkl"))

    prophet_model, lstm_data = load_models(models_dir)
    return prophet_model, lstm_data

def plot_historical(df, metric="concurrent_players"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[metric], mode='lines', name=metric.replace('_', ' ').title(), line=dict(color='#00b4d8', width=2)))
    
    # Add anomaly highlights
    anom = df[df['final_anomaly'] == 1] if 'final_anomaly' in df.columns else pd.DataFrame()
    if not anom.empty:
        fig.add_trace(go.Scatter(x=anom['timestamp'], y=anom[metric], mode='markers', name='Anomalies',
                                 marker=dict(color='red', size=8, symbol='x')))
    
    fig.update_layout(title=f"{metric.replace('_', ' ').title()} Over Time", template="plotly_dark", height=400,
                      xaxis_title="Time", yaxis_title=metric)
    return fig

def plot_forecast(prophet_model, df, hours_ahead=48):
    if prophet_model is None:
        return go.Figure().update_layout(title="Prophet model not available (install prophet)", template="plotly_dark")
    
    future = prophet_model.make_future_dataframe(periods=hours_ahead, freq='h')
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    future['hour_sin'] = np.sin(2 * np.pi * future['ds'].dt.hour / 24)
    future['dow_sin'] = np.sin(2 * np.pi * future['ds'].dt.dayofweek / 7)
    
    forecast = prophet_model.predict(future)
    
    fig = go.Figure()
    # Historical
    hist = df.tail(24*7)  # last week
    fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['concurrent_players'], mode='lines', name='Historical', line=dict(color='#00b4d8')))
    # Forecast
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#ff6b6b', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='rgba(255,107,107,0.3)')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', fill='tonexty', line=dict(color='rgba(255,107,107,0.3)')))
    
    fig.update_layout(title="Player Load Forecast (Next 48h) - Prophet", template="plotly_dark", height=450)
    return fig, forecast

def main():
    # ── Session state defaults ────────────────────────────────────────────────
    if "live_mode" not in st.session_state:
        st.session_state.live_mode = False
    if "sim_seed" not in st.session_state:
        st.session_state.sim_seed = 42

    st.title("📈 AI Server Scaling Predictor")
    st.caption("For Persistent Sandbox MMO | Solo Dev Project | May 2026")

    config = load_config()
    current_seed = st.session_state.sim_seed if st.session_state.live_mode else 42
    df = load_all_data(seed=current_seed)
    prophet_model, lstm_data = get_trained_models()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        metric = st.selectbox("Primary Metric", ["concurrent_players", "cpu_usage_percent", "tick_rate", "memory_usage_gb"])
        hours_back = st.slider("History Window (hours)", 24, 24*14, 24*7)
        show_anom = st.checkbox("Highlight Anomalies", True)

        st.divider()

        # ── Data mode toggle ─────────────────────────────────────────────────
        st.subheader("Data Mode")
        mode_choice = st.radio(
            "Mode",
            ["🔒 Reproducible", "🎲 Live Simulation"],
            index=1 if st.session_state.live_mode else 0,
            help=(
                "Reproducible: seed=42, same data every run (portfolio default).\n"
                "Live Simulation: random seed each refresh — different players, CPU, anomalies."
            ),
            label_visibility="collapsed",
        )
        new_live = mode_choice == "🎲 Live Simulation"
        if new_live != st.session_state.live_mode:
            st.session_state.live_mode = new_live
            if new_live:
                st.session_state.sim_seed = np.random.randint(1, 99999)
            st.cache_data.clear()
            st.rerun()

        if st.session_state.live_mode:
            st.caption(f"Active seed: `{st.session_state.sim_seed}`")
            if st.button("🔄 New Simulation", help="Roll a new random seed and regenerate data"):
                st.session_state.sim_seed = np.random.randint(1, 99999)
                st.cache_data.clear()
                st.rerun()
        else:
            st.caption("Seed: `42` — fully reproducible")

        st.divider()
        st.subheader("Model Status")
        st.success("Prophet: Ready" if prophet_model else "Prophet: Not installed")
        st.success("LSTM: Ready" if lstm_data else "LSTM: Not trained")
        if lstm_data:
            mae = lstm_data.get("metrics", {}).get("lstm_mae", "—")
            st.caption(f"LSTM MAE: {mae} players")

        if st.button("Retrain Models", help="Delete saved artifacts and retrain all models from scratch"):
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            for fname in ("prophet_model.pkl", "lstm_model.pt", "anomaly_model.pkl"):
                fpath = os.path.join(models_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
    
    # KPI Cards with Tooltips
    latest = df.tail(1).iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Players", f"{int(latest['concurrent_players'])}", 
                  delta=f"{int(latest['concurrent_players'] - df.iloc[-25]['concurrent_players'])} last hour",
                  help="Number of players currently online. Higher = more load on servers.")
    
    with col2:
        st.metric("CPU Load", f"{latest['cpu_usage_percent']:.1f}%", 
                  delta_color="inverse" if latest['cpu_usage_percent'] > 80 else "normal",
                  help="Average CPU usage across all servers. Above 80% = time to consider scaling.")
    
    with col3:
        st.metric("Tick Rate", f"{int(latest['tick_rate'])} TPS", 
                  delta=f"{int(latest['tick_rate'] - df.iloc[-25]['tick_rate'])}" if 'tick_rate' in df.columns else "",
                  help="Game server performance (Ticks Per Second). Lower = possible performance issues.")
    
    with col4:
        st.metric("Anomalies (24h)", f"{df.tail(24)['anomaly'].sum() if 'anomaly' in df.columns else 0}", 
                  delta="Investigate" if df.tail(24)['anomaly'].sum() > 2 else "Normal",
                  help="Number of unusual patterns detected. Could be bots, DDoS, or sudden churn.")

    # Advanced Settings (Optional Features)
    with st.expander("⚙️ Advanced Settings — Revenue & Auto-Deploy (Optional)"):
        st.caption("These are disabled by default. Enable only when you have real revenue data and admin access.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            enable_revenue = st.toggle("Enable Revenue Check for New Realms", value=False,
                                       help="Requires Stripe API or revenue database integration")
        with col_b:
            auto_deploy = st.toggle("Allow Automatic Server Deployment", value=False,
                                    help="Requires admin authentication in production")
        
        if enable_revenue:
            st.info("Revenue-based realm expansion enabled. Connect to Stripe for live checks.")
        if auto_deploy:
            st.warning("Auto-deploy mode is ON. Real usage requires secure admin login.")

    # Demo Scenario Buttons
    st.markdown("### Quick Demo Scenarios")
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        if st.button("🚀 Simulate Viral Spike"):
            st.success("**Simulation:** Sudden player surge detected! System recommends immediate SCALE UP.")
            st.info("In production: Would automatically provision 1-2 new server shards.")
    
    with demo_col2:
        if st.button("🛡️ Simulate DDoS Attack"):
            st.error("**Simulation:** Extreme network + CPU spike with low player engagement. Possible DDoS detected!")
            st.warning("System would flag this as CRITICAL and send immediate alert to ops team.")
    
    with demo_col3:
        if st.button("📉 Simulate Player Drop"):
            st.warning("**Simulation:** Sudden 45% drop in players after high CPU period. Possible patch issue or ban wave.")
            st.info("System recommends investigating recent changes and monitoring retention.")

    # ========== Cost & Revenue Projection Chart ==========
    st.markdown("### 30-Day Cost & Revenue Projection")
    st.caption("Example projection. Infrastructure cost ($8/250 players) and revenue are placeholders. In production, connect to cloud billing API + Stripe/revenue DB for real numbers.")
    
    # Simple projection data
    days = list(range(1, 31))
    base_players = latest['concurrent_players']
    projected_players = [base_players * (1 + 0.008 * d) for d in days]  # ~0.8% daily growth
    infrastructure_cost = [round(p / 250 * 8, 1) for p in projected_players]  # $8 per 250 players
    projected_revenue = [650 * (1 + 0.005 * d) for d in days]  # slow revenue growth (default $650/mo)
    
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(x=days, y=projected_players, name="Projected Players", 
                                  line=dict(color="#00b4d8", width=2), yaxis="y1"))
    fig_cost.add_trace(go.Scatter(x=days, y=infrastructure_cost, name="Est. Infrastructure Cost ($/day)", 
                                  line=dict(color="#ff6b6b", width=2, dash="dash"), yaxis="y2"))
    fig_cost.add_trace(go.Scatter(x=days, y=projected_revenue, name="Projected Revenue ($/day)", 
                                  line=dict(color="#00c853", width=2, dash="dot"), yaxis="y2"))
    
    fig_cost.update_layout(
        title="30-Day Infrastructure Cost vs Revenue Projection",
        template="plotly_dark",
        height=380,
        xaxis_title="Days from Now",
        yaxis=dict(title="Players", side="left"),
        yaxis2=dict(title="USD ($)", side="right", overlaying="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Trends", 
        "🔮 Forecast", 
        "🚨 Anomalies", 
        "✅ Recommendations",
        "❓ How It Works",
        "🧪 What-If Simulator"
    ])
    
    with tab1:
        st.subheader("Historical Server Health")
        hist_df = df.tail(hours_back).copy()
        if show_anom:
            hist_df = create_time_features(hist_df)
            hist_df = add_domain_features(hist_df)
            hist_df = detect_anomalies(hist_df)  # re-detect for viz
        fig = plot_historical(hist_df, metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-metric overview
        st.subheader("All Metrics Heatmap (last 48h)")
        recent = df.tail(48)
        fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                             subplot_titles=("Players & CPU", "Network & Memory", "Tick Rate"))
        fig2.add_trace(go.Scatter(x=recent['timestamp'], y=recent['concurrent_players'], name='Players'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=recent['timestamp'], y=recent['cpu_usage_percent'], name='CPU %'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=recent['timestamp'], y=recent['network_in_mbps'], name='Net In'), row=2, col=1)
        fig2.add_trace(go.Scatter(x=recent['timestamp'], y=recent['memory_usage_gb'], name='Mem GB'), row=2, col=1)
        fig2.add_trace(go.Scatter(x=recent['timestamp'], y=recent['tick_rate'], name='Tick Rate'), row=3, col=1)
        fig2.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Player Load Forecasting")
        if prophet_model:
            forecast_fig, forecast_df = plot_forecast(prophet_model, df)
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            if lstm_data:
                lstm_mae = lstm_data.get("metrics", {}).get("lstm_mae", "—")
                st.caption(
                    f"LSTM next-hour prediction (multivariate): MAE = {lstm_mae} players on holdout test set"
                )
        else:
            st.warning("Install Prophet for full forecast: `pip install prophet` (see README)")
            # Fallback simple forecast
            st.line_chart(df.set_index('timestamp')['concurrent_players'].tail(168))
    
    with tab3:
        st.subheader("Anomaly Detection Results")
        anom_df = detect_anomalies(df.tail(500))  # recent window
        st.dataframe(anom_df[['timestamp', 'concurrent_players', 'cpu_usage_percent', 'anomaly_score', 'final_anomaly', 'anomaly_type']].tail(20),
                     use_container_width=True, hide_index=True)
        
        # Friendly Anomaly Explanation Cards
        st.markdown("### What These Anomalies Mean")
        
        with st.container():
            st.markdown("**High CPU + Low Players**")
            st.caption("Often indicates botting, macro usage, or exploit attempts. These can damage the in-game economy and frustrate real players.")
        
        with st.container():
            st.markdown("**Sudden Player Drop**")
            st.caption("Could be caused by a bad patch, mass ban wave, or server issues. Important to investigate quickly to prevent further churn.")
        
        with st.container():
            st.markdown("**Network / CPU Spikes**")
            st.caption("Classic sign of DDoS attacks or large-scale cheating tools. These require immediate attention to protect server stability.")
        
        # Anomaly distribution
        if 'anomaly_type' in anom_df.columns:
            st.bar_chart(anom_df['anomaly_type'].value_counts())
        
        st.info("Red flags: High CPU + Low Players (botting) | Sudden Drops (churn/patch) | Network Spikes (DDoS)")
    
    with tab4:
        st.subheader("Ops Recommendations")
        # Generate recs
        latest_df = get_latest_metrics(df, 48)
        anom_recent = detect_anomalies(latest_df)
        forecast_placeholder = None
        if prophet_model:
            _, forecast_placeholder = plot_forecast(prophet_model, df, 24)  # just for recs
        
        recs = generate_recommendations(latest_df, forecast_placeholder, anom_recent, config)
        rec_df = format_recs_for_dashboard(recs)
        
        # Show current scaling status
        status = get_scaling_status()
        st.caption(f"**Current Status:** {status['active_shards']} shards | {status['current_load']}% load | Auto-scaling: {'Enabled' if status['auto_scaling_enabled'] else 'Dry Run Mode'}")
        
        for i, row in rec_df.iterrows():
            color = "high" if row.get('priority') in ['HIGH', 'CRITICAL'] else ("medium" if row.get('priority') == 'MEDIUM' else "low")
            with st.container():
                st.markdown(f"**{row['action']}** <span class='{color}'>[{row.get('priority', 'INFO')}]</span>", unsafe_allow_html=True)
                st.caption(row['reason'])
                
                # Add execution button for each recommendation
                if st.button(f"Execute: {row['action']}", key=f"exec_{i}"):
                    result = execute_recommendation(row.to_dict(), dry_run=True)
                    if result.get("executed"):
                        st.success(result["message"])
                    else:
                        st.info(result["message"])
        
        if st.button("Export Recommendations as CSV"):
            rec_df.to_csv("recommendations_export.csv", index=False)
            st.success("Exported to recommendations_export.csv")

    # ========== NEW TAB 5: How It Works ==========
    with tab5:
        st.header("How This System Works")
        st.markdown("""
        This dashboard is powered by a complete machine learning pipeline designed for real game operations.
        
        ### The Full Pipeline (Simplified)
        
        **1. Data Collection**  
        We collect hourly metrics from game servers (player count, CPU, memory, tick rate, network usage).
        
        **2. Forecasting**  
        Two models work together:
        - **Prophet** predicts load 1–48 hours ahead (great for planning)
        - **LSTM** predicts the next hour with high accuracy (great for immediate response)
        
        **3. Anomaly Detection**  
        The system watches for unusual patterns:
        - Sudden high CPU with low players → possible botting
        - Extreme network spikes → possible DDoS attack
        - Sudden player drops → possible patch issues or mass churn
        
        **4. Smart Recommendations**  
        Based on forecasts and anomalies, the system suggests actions like:
        - "Add 1 more server shard"
        - "Investigate suspicious activity"
        - "Consider opening a new realm"
        
        **5. Safe Execution**  
        Everything runs in **Dry-Run Mode** by default. In production, approved actions can automatically create new servers via cloud APIs (Hetzner, AWS, etc.).
        
        ---
        **Why this matters:** Instead of guessing when to scale, you get data-driven, cost-aware recommendations that protect both player experience and your budget.
        """)

    # ========== NEW TAB 6: What-If Simulator ==========
    with tab6:
        st.header("What-If Simulator")
        st.caption("Adjust the sliders below to simulate different scenarios and see how the system would react.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sim_players = st.slider("Current Concurrent Players", 50, 400, 180, step=10)
            sim_growth = st.slider("Expected Growth (players/day)", -20, 50, 8, step=2)
            sim_revenue = st.slider("Monthly Private Server Revenue ($)", 0, 3000, 650, step=50)
        
        with col2:
            st.write("**Simulated Conditions:**")
            st.write(f"- Current Load: ~{int(sim_players / 250 * 100)}% of capacity")
            st.write(f"- Projected Peak (24h): ~{int(sim_players * 1.3 + sim_growth * 3)} players")
            st.write(f"- Revenue vs New Realm Cost: {'✅ Justified' if sim_revenue > 1200 else '⚠️ Marginal'}")
        
        if st.button("Run What-If Simulation", type="primary"):
            # Simple simulation logic
            if sim_players > 210 and sim_growth > 10:
                st.success("**Recommendation:** SCALE UP immediately. High load + strong growth detected.")
                st.info("System would suggest adding 1 new server shard (estimated $8–12/month).")
            elif sim_players > 180 and sim_revenue > 1200:
                st.warning("**Recommendation:** Consider opening a new realm. Revenue supports expansion.")
            elif sim_growth < -10:
                st.error("**Recommendation:** Monitor closely. Declining player base detected.")
            else:
                st.info("**Recommendation:** System is stable. No immediate action needed.")
            
            st.caption("Note: This is a simplified simulation. The real system uses Prophet + LSTM for more accurate forecasting.")
    
    # Footer
    st.divider()
    st.caption("Server Scaling Predictor • Production-Ready Architecture • Dry-Run Mode Active")
    st.caption("Models trained on 90 days synthetic telemetry | Last retrained: " + datetime.now().strftime("%Y-%m-%d %H:%M"))

if __name__ == "__main__":
    main()