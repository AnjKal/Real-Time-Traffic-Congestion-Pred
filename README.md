# Bangalore DCRNN Traffic Prediction

## Project goal

Predict traffic speeds at 8 Bangalore locations (Indiranagar, M.G. Road, Koramangala,
Jayanagar, Hebbal, Whitefield, Yeshwanthpur, Electronic City) at +5, +10 and +20 min
horizons using a DCRNN, with a 3-minute polling cadence. Data comes from the TomTom Flow
Segment API (live) plus synthetic bootstrap data. The final deliverable is a FastAPI + Folium
web app that shows the least-congested route between two points.

Reference implementation: https://github.com/liyaguang/DCRNN (TensorFlow).

## 12-step plan

The project is built strictly sequentially, one step at a time. This repo is currently at
**Step 1 of 12: project scaffolding** (no data, models or API calls yet). Later steps include
node/graph construction (Steps 2-3), synthetic data (Step 5), the DCRNN model (Step 7) and the
web app.

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your real TOMTOM_API_KEY
python -m scripts.verify_step1
pytest
```
