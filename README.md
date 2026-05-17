# AI-Powered Server Scaling Predictor

**End-to-End Time-Series Forecasting, Anomaly Detection & Automated Scaling System for Multiplayer Games**

*Production-Ready ML Pipeline | May 2026*

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![CI](https://github.com/yourusername/server-scaling-predictor-mmo/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/server-scaling-predictor-mmo/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](./Dockerfile)

---

## 🎯 Project Overview

This production-ready ML system predicts server load for persistent multiplayer games, detects anomalous behavior (DDoS, botting, viral spikes, sudden churn), and generates actionable recommendations for scaling servers or opening new realms. It includes a complete architecture for automatic infrastructure scaling via cloud APIs.

**Why this matters for game ops:**
- Persistent worlds run 6–24 months. Manual capacity planning fails at 200+ concurrent.
- Early detection of anomalies protects economy and player experience.
- Cost-aware recommendations (scale only when revenue justifies) are critical for solo/small-team monetization via private servers.

**Key Features:**
- Full ML lifecycle: synthetic data generation → feature engineering → ensemble forecasting (Prophet + LSTM) → anomaly detection → intelligent recommendation engine → interactive dashboard.
- **Production Scaling Architecture**: Includes commented production code for automatic server scaling via Hetzner Cloud API (easily adaptable to AWS, GCP, or Kubernetes).
- Real-world impact: Directly solves capacity planning and cost optimization for long-running multiplayer games.
- Clean, modular, documented code ready for integration with live game telemetry (Prometheus, custom agents, etc.).

**Primary Outcomes (on synthetic 90-day dataset):**
- Prophet 7-day holdout: **MAE = 13.85 players**, MAPE = 7.1%
- LSTM next-hour: **MAE = 5.29 players** (early-stopped at epoch 40/100)
- Anomaly detection: **Precision 0.71 / Recall 0.72 / F1 0.71** on injected 3% anomalies
- Dashboard loads in <2s, provides tiered recommendations with cost estimates.

---

## 📁 Project Structure

```
server_scaling_predictor/
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── Dockerfile
├── train.py                 # One-command pipeline
├── predict.py               # CLI inference script
├── generate_pdf.py          # Technical write-up PDF generator
├── data/
│   ├── synthetic_generator.py
│   └── synthetic_server_metrics.csv
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── forecasting.py       # Prophet + PyTorch LSTM
│   ├── anomaly_detection.py # IsolationForest + domain rules
│   ├── recommendation.py
│   └── scaling_executor.py  # Dry-run / production scaling API hooks
├── dashboard/
│   └── app.py               # Full Streamlit UI
├── tests/
│   └── test_pipeline.py     # Smoke + unit tests (pytest)
├── .github/
│   └── workflows/ci.yml     # GitHub Actions CI (Python 3.11 + 3.12)
└── models/                  # Auto-generated artifacts (not committed)
    ├── prophet_model.pkl
    ├── lstm_model.pt
    ├── anomaly_model.pkl
    └── metadata.json
```

---

## 🚀 Quick Start (5 minutes)

```bash
git clone https://github.com/yourusername/server-scaling-predictor-mmo.git
cd server-scaling-predictor-mmo
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional but recommended for Prophet (can be tricky on some systems):
# conda install -c conda-forge prophet   OR   pip install prophet --no-binary :all:

python train.py                    # Generates data + trains all 3 models (~2-3 min)
streamlit run dashboard/app.py     # Opens at http://localhost:8501
```

**First run** automatically trains all models (Prophet + LSTM + Isolation Forest) and saves them to `models/`.  The dashboard will bootstrap itself on first load if `train.py` has not been run yet.

> **Note:** If you have previously trained models from an older version of this repo, delete `models/` and retrain — the LSTM now uses `shuffle=False` (correct for time-series) which significantly improves accuracy.

### Docker (one command)

```bash
docker build -t server-scaling-predictor .
docker run -p 8501:8501 server-scaling-predictor
# → http://localhost:8501
```

The Docker image pre-trains all models at build time so startup is instant.

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover data ingestion contracts, feature shapes, LSTM forward pass, anomaly detection, recommendations, and the scaling executor — runnable on a fresh clone with no pre-trained models.

---

## 📜 License & Reproducibility

MIT License. Full reproducibility via fixed seed, pinned requirements, and serialized artifacts (`metadata.json` contains exact metrics + training timestamp).

One-command reproduction (as above) works on fresh Ubuntu/macOS/Windows with Python 3.11+.

---