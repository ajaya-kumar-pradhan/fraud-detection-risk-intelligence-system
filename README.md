# 🛡️ Fraud Risk Intelligence System

> **XGBoost · SHAP · Financial Forensics · Streamlit Deployment**

---

## 🚀 Live Demo
**Click the link below to scan transactions for fraud risk instantly (Free & Always On):**

### 👉 [**Launch Fraud Risk Intelligence Dashboard**](https://huggingface.co/spaces/ajayapradhanconnect/Fraud-Detection-Risk-Intelligence-System)

---

## 📌 Project Overview
The **Fraud Risk Intelligence System** is an enterprise-grade AI solution designed to monitor financial transactions for fraudulent activity in real-time. By combining a high-performance **XGBoost** model with **SHAP (Explainable AI)**, the system not only flags suspicious transfers but also provides a transparent "Diagnostic Summary," explaining *why* a particular transaction was flagged as risky.

### 🌟 Key Features
- **In-Memory Risk Scoring**: Real-time inference on transactional data.
- **Explainable AI (XAI)**: Visualizes the exact features (amount, type, etc.) driving the risk score.
- **Stress Testing Engine**: Interactive "What-If" scenarios to map risk sensitivity to transaction volume.
- **Structural Anomaly Detection**: Identifies "Error Balance" patterns common in ledger-based fraud.
- **Modern Dark/Light UI**: Professional dashboard interface for financial analysts.

---

## 🧠 The Intelligence Engine
The system utilizes a monolithic architecture where the model and interpretability logic are integrated directly into the dashboard for zero-latency analysis.

- **Model**: XGBoost (Extreme Gradient Boosting) optimized for imbalanced financial data.
- **Interpretability**: SHAP (SHapley Additive exPlanations) for local feature contribution mapping.
- **Validation**: Automated "Accounting Audit" of sender/recipient balance consistency.

---

## 🛠️ Tech Stack
- **Backend**: Python, XGBoost, Scikit-learn, joblib.
- **Explainability**: SHAP, Matplotlib.
- **Frontend**: Streamlit (with Custom CSS).
- **Deployment**: Hugging Face Spaces.

---

## 📁 Project Structure
```text
fraud-detection-risk-intelligence-system/
├── app.py                      # Main Intelligence Dashboard
├── requirements.txt            # Production Dependencies
├── model_artifacts/            # Core Engine Files
│   ├── xgboost_fraud_model.json
│   ├── feature_columns.joblib
│   └── threshold.txt
└── README.md                  # You are here!
```

---

## 📦 Run Locally
1. **Clone the repo**
   ```bash
   git clone https://github.com/ajaya-kumar-pradhan/fraud-detection-risk-intelligence-system.git
   cd fraud-detection-risk-intelligence-system
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the system**
   ```bash
   streamlit run app.py
   ```

---

## 👤 Author
**Ajaya Kumar Pradhan**  
Data Analyst · Power BI Developer · ML Engineer  
📍 Bhubaneswar, Odisha, India

[![GitHub](https://img.shields.io/badge/GitHub-ajayaconnect-181717?style=flat-square&logo=github)](https://github.com/ajaya-kumar-pradhan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/)

---
*Built as part of Enterprise Risk Management Portfolio.*
