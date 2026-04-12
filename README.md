# Fraud Risk Intelligence System
### Decision Support Engine for Financial Transaction Security

This repository contains a high-performance system designed to identify and analyze fraudulent patterns in financial transaction data. By combining gradient-boosted decision trees with real-time API services and an interactive investigation dashboard, it provides a comprehensive solution for security teams to monitor and mitigate risk.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red?style=flat&logo=xgboost)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat&logo=streamlit)

---

## 🏦 The Challenge

Financial institutions face billions in losses annually due to sophisticated fraud exploits. Traditional rule-based systems often struggle with high false-positive rates or fail to detect subtle anomalies in complex transaction flows. This system addresses these challenges by analyzing transactional "DNA" and balance inconsistencies to identify risk markers in real-time.

---

## 🎯 Project Objectives

*   **Real-Time Classification:** Rapid identification of suspicious transactions using a low-latency API.
*   **Handling Class Imbalance:** Optimization strategies for datasets with extreme imbalance (~0.13% fraud rate).
*   **Analyst Intelligence:** A professional dashboard providing diagnostic insights into why specific transactions are flagged.
*   **Risk Sensitivity Testing:** Tools to simulate transaction variations and observe their impact on risk scores.

---

## 📊 Dataset & Features

The system is trained on a dataset of over **6 million transactions**, focusing on structural discrepancies between expected and actual balances.

| Attribute | Detail |
| :--- | :--- |
| **Volume** | 6 Million+ Records |
| **Domain** | Digital Banking / FinTech |
| **Class Imbalance** | 0.13% Fraudulent Cases |
| **Primary Vectors** | `TRANSFER` and `CASH_OUT` operations |

### Feature Engineering
We developed specialized features to capture accounting bypass exploits:
- **`errorBalanceOrig`**: Discrepancy between sender's initial balance, transaction amount, and reported final balance.
- **`errorBalanceDest`**: Discrepancy in the recipient's balance logic.
- **`hour_of_day`**: Temporal patterns derived from transaction sequence steps.

---

## ⚙️ Methodology & Performance

### Model Optimization
We utilized **XGBoost** with a cost-sensitive learning approach. By adjusting the `scale_pos_weight` parameter, we forced the model to prioritize detection of the rare minority class without the noise introduced by oversampling techniques like SMOTE.

### Evaluation Metrics
We prioritize **Precision-Recall AUC (PR-AUC)** over traditional ROC-AUC. In fraud detection, it is critical that when an alert is generated, the likelihood of actual fraud is high to minimize operational fatigue for security analysts.

---

## 🌐 System Architecture

### 1. Backend Service (`api.py`)
A FastAPI-based REST service that provides endpoints for prediction and model interpretability.
- `POST /predict`: Generates a real-time risk score.
- `POST /explain`: Returns mathematical drivers for a specific transaction level.

### 2. Investigation Dashboard (`app.py`)
A professional-grade interface for risk analysts to input transaction details, view risk metrics, and perform sensitivity analysis through "What-If" simulations.

---

## 📁 Project Structure

```
fraud-detection-risk-intelligence-system/
├── app.py                # Redesigned Investigation Dashboard
├── api.py                # FastAPI REST Service
├── Fraud_Detection_ML.py # Model Training & Pipeline logic
├── model_artifacts/      # Serialized models and feature configurations
├── README.md             # Technical Documentation
└── requirements.txt      # Project Dependencies
```

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/ajaya-kumar-pradhan/fraud-detection-risk-intelligence-system.git
cd fraud-detection-risk-intelligence-system
pip install -r requirements.txt
```

### 2. Startup
The system requires both the backend and frontend to be active.

**Start the Service Interface:**
```bash
uvicorn api:app --port 8000
```

**Start the Analyst Dashboard:**
```bash
streamlit run app.py
```

---

## 🛠️ Technical Stack

- **Language:** Python 3.10+
- **Modeling:** XGBoost, Scikit-learn, SHAP
- **Data:** Pandas, NumPy
- **Communication:** FastAPI (Uvicorn), Requests
- **Interface:** Streamlit, Matplotlib

---

## 👤 Author
**Ajaya Kumar Pradhan**
*Data Analyst | Machine Learning Enthusiast*
