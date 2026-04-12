# 🚀 Cloud Deployment Guide (Simplified)

Your Fraud Risk Intelligence System now uses a **Monolithic Architecture** (everything in one file). This makes deployment significantly easier and more reliable.

---

## 🎨 Deployment on Streamlit Cloud (Recommended)

Since the "brain" and the "UI" are now combined into `app.py`, you only need to deploy one app:

1.  **Go to [share.streamlit.io](https://share.streamlit.io/)**.
2.  **Deploy a New App**:
    *   Connect your GitHub repo.
    *   **Main file path**: `app.py`
    *   **Branch**: `main`
3.  **Click Deploy!**

Streamlit Cloud will automatically detect your `requirements.txt` and load your ML model. There are **no secrets or backend URLs** required anymore!

---

## 🛠️ Performance & Scalability
- **Direct Inference**: The app loads the XGBoost model directly into memory using `@st.cache_resource`, ensuring near-instant predictions.
- **Joblib Serialization**: Feature metadata is loaded via `joblib` for maximum stability across different Python environments.

## 💻 Local Execution
To run the system on your computer:
```bash
streamlit run app.py
```
*(No second terminal or Uvicorn required)*
