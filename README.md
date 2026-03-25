# fraud-detection-risk-intelligence-system
An end-to-end, real-time Machine Learning pipeline designed to intercept, predict, and explain fraudulent financial transactions.

## Architecture

This enterprise-grade system separates concerns between core predictive modeling, blazing-fast API inference, and insightful UI rendering.

1. **Machine Learning Core (`src/train.py`)**
   - Utilizes advanced feature engineering to detect structural balance inconsistencies.
   - Handled a highly imbalanced dataset (99.9% clean) natively using XGBoost's `scale_pos_weight` matrix to minimize synthetic noise generation.
   - Automatically tunes evaluation thresholds prioritizing Recall and PR-AUC.
2. **Real-time API Backend (`api/main.py`)**
   - High-throughput **FastAPI** application wrapper.
   - Processes simulated streaming transactional payloads and translates them into model-ready engineered vectors on the fly.
3. **Analyst Dashboard (`dashboard/app.py`)**
   - Interactive **Streamlit** user interface.
   - Offers real-time alerts, cross-scenario Risk Level modeling, and a highly intuitive "What-If" Analysis engine.
   - Embedded with **SHAP Explainer**, visually and textually demonstrating exactly *why* a particular transaction triggered an intervention.

## Dependencies

Ensure all requirements are installed:
```bash
pip install fastapi uvicorn streamlit xgboost shap pandas matplotlib pydantic scikit-learn
```

## How to Run Locally

### 1. Launch the API Service
Start the inference server locally on port 8000:
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
*Visit `http://localhost:8000/docs` to interact with the API endpoints directly.*

### 2. Launch the Streamlit Dashboard
With the API running, open a new terminal and launch the analyst UI:
```bash
python -m streamlit run dashboard/app.py
```
