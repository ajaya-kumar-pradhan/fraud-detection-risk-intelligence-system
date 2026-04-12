# Fraud Detection System
### End-to-End Machine Learning | Classification | XGBoost | FastAPI | Streamlit + AI Chatbot

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red?style=flat&logo=xgboost)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.3-150458?style=flat&logo=pandas)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat)

---

## 🚀 Problem Statement

Financial fraud is a multi-billion dollar problem causing significant losses and eroding customer trust. Legacy rule-based detection systems flag massive transaction volumes randomly (high false-positive rates) while missing sophisticated, subtle cash-out exploits.

This project builds an **enterprise-grade Machine Learning fraud interception system** with a real-time REST API, an interactive investigation dashboard, and an AI-powered SHAP diagnostic chatbot—enabling security teams to detect and explain fraudulent anomalies instantly.

---

## 🎯 Objective

- Classify fraudulent transactions in real-time.
- Handle massive class imbalance (99.9% legitimate traffic).
- Build a production-ready API + dashboard for fraud analysts.
- Provide "What-If" scenario tracking for vulnerability testing.
- Enable transparent AI querying via a SHAP-based Natural Language Chatbot.

---

## 🧠 ML System Pipeline

```
Raw Data → Synthetic Data Handling → Feature Engineering → Cost-Sensitive Training (XGBoost) → Threshold Tuning → API Deployment → Interactive Dashboard
```

---

## 📊 Dataset

| Property         | Detail                                    |
|------------------|-------------------------------------------|
| Records          | Over 6 Million simulated transactions      |
| Imbalance        | ~0.13% Fraudulent                         |
| Target Variable  | `isFraud` (Binary Classification)         |
| Domain           | FinTech / Digital Banking                 |
| Features         | Amounts, balances, transaction types      |

---

## 🔍 Key Insights (EDA)

- **Extreme Class Imbalance:** Fraud is exceptionally rare, requiring precision optimization over pure accuracy.
- **Top predictors:**
  - Structural discrepancies between expected balances and actual balances.
  - Transaction Type (`TRANSFER` and `CASH_OUT` dominate fraud vectors).
  - Transaction Volume (spikes triggering heuristic flags).

---

## ⚙️ Feature Engineering

Created **advanced engineered features** to capture accounting anomalies:

```python
# ── Mathematical Discrepancy Features ──
# Flagging when the math of the transfer doesn't add up (classic bypass exploit).
errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
errorBalanceDest = oldbalanceDest + amount - newbalanceDest

# ── Temporal Features ──
hour_of_day = step % 24
```

| Feature              | Category    | Description                              |
|----------------------|-------------|------------------------------------------|
| errorBalanceOrig     | Accounting  | Sender's post-transaction balance discrepancy |
| errorBalanceDest     | Accounting  | Receiver's post-transaction balance discrepancy|
| hour_of_day          | Temporal    | Translates abstract sequence step into 24H cycle |

---

## 🤖 Model Training & Evaluation

### Training Configuration (Cost-Sensitive XGBoost)
Due to extreme imbalance, standard models fail or predict 0 universally. We utilized XGBoost natively weighted using `scale_pos_weight` to mathematically force the trees to care about the rare minority class, bypassing computationally heavy and noisy SMOTE techniques.

```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    scale_pos_weight=imbalance_ratio,
    learning_rate=0.1,
    random_state=42
)
```

### Approach
- Threshold calibration maximizing F-Beta and PR-AUC.
- Advanced Precision-Recall intersection tuning to minimize false-positives for human analysts.

---

## 📈 Results

**👉 Selected Model: XGBoost**

### 🎯 Why PR-AUC?
In fraud systems where negatives outnumber positives 1,000 to 1, ROC-AUC is misleading. The model was evaluated against **Precision-Recall AUC (PR-AUC)**, ensuring that when the AI alerts an analyst, the transaction is genuinely suspicious.

---

## 🌐 Production System

### FastAPI REST API (`api.py`)

| Endpoint          | Method | Description                |
|-------------------|--------|----------------------------|
| `/predict`        | POST   | Real-time fraud scoring    |
| `/explain`        | POST   | Extract SHAP driver forces |

```bash
# Example API call
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "CASH_OUT",
    "amount": 10000.0,
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
  }'

# Response: {"is_fraud": true, "risk_score": 98, "probability": 0.982, "threshold": 0.96}
```

### Streamlit Dashboard (`app.py`)

| Feature                  | Description                                         |
|--------------------------|-----------------------------------------------------|
| 📊 KPI Cards            | Fraud Probability, Alert Status, Risk Score         |
| 📈 Forecast Chart        | Visual probability breakdown bar scales             |
| 🔬 What-If Analysis      | Dynamic transaction amount hacking scenarios        |
| 💬 AI Chatbot            | SHAP-driven mathematical diagnostic agent           |

---

## 💬 AI Chatbot — Fraud AI Diagnostics

The built-in Natural Language generator understands the background XGBoost trees via SHAP and responds with explainable insights:

*Example Response:*
> **AI Diagnostic:** I categorized this as **HIGH RISK** primarily because the `errorBalanceOrig` is abnormal. This specific feature dramatically increased the probability of fraud (+3.42 log-odds).
> *Explanation: Our internal accounting checks show a massive discrepancy between what the user had, what they transferred, and what their final balance was recorded as. This is a massive red flag for bypassing accounting rules during a cash-out!*

---

## 🔬 What-If Scenario Analysis

| Scenario                    | What it Simulates                        |
|-----------------------------|------------------------------------------|
| 🏷️ Amount Variance          | Altering the target transfer footprint while proportionally maintaining logical accounting bounds to map the risk sensitivity gradient without injecting fake bugs. |

---

## 💡 Business Impact

- **Intercept Cash Outs** instantly before funds leave the domain.
- **Explainable AI (XAI)** ensuring banking analysts trust the system.
- **REST API integration** seamlessly attaching to existing transaction microservices.
- **Reduced False Positives** preserving legitimate user experience.

---

## 📁 Project Structure

```
fraud-detection-risk-intelligence-system/
│
├── Fraud_Detection_ML.py         # Full ML pipeline (EDA + training)
├── api.py                        # FastAPI REST API
├── app.py                        # Streamlit Dashboard + AI Chatbot
├── model_artifacts/
│   ├── xgboost_fraud_model.json  # Trained XGBoost model
│   ├── feature_columns.pkl       # Feature names list
│   └── threshold.txt             # Calibrated decision threshold
├── .gitignore
├── requirements.txt
├── DEPLOYMENT.md
├── STREAMLIT_DOCKER_DEPLOYMENT.md
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ajayakumarpradhan/fraud-detection-risk-intelligence-system.git
cd fraud-detection-risk-intelligence-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates model_artifacts/)
python Fraud_Detection_ML.py

# 4. Start FastAPI (port 8000)
python -m uvicorn api:app --port 8000

# 5. Start Streamlit Dashboard (port 8501)
python -m streamlit run app.py --server.port 8501
```

---

## 🛠️ Tech Stack

| Category         | Tools                                  |
|------------------|----------------------------------------|
| Language         | Python 3.10+                           |
| ML Framework     | XGBoost, Scikit-learn, SHAP            |
| Data Processing  | Pandas, NumPy                          |
| Visualization    | Matplotlib                             |
| API Framework    | FastAPI + Uvicorn                      |
| Dashboard        | Streamlit                              |
| Task Type        | Binary Classification (Fraud)          |

---

## 👤 Author

**Ajaya Kumar Pradhan**
*Data Analyst | Machine Learning Enthusiast*
