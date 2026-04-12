# 🚀 Cloud Deployment Guide

This guide will help you deploy your Fraud Risk Intelligence System to the cloud for free using **Render** (Backend) and **Streamlit Cloud** (Frontend).

---

## 🏗️ Part 1: Deploying the Backend (FastAPI) on Render

[Render.com](https://render.com/) is perfect for hosting your `api.py`.

1.  **Create a Account**: Sign up at Render using your GitHub account.
2.  **Create a New Web Service**:
    *   Click **"New"** -> **"Web Service"**.
    *   Connect your GitHub repository.
3.  **Configure the Service**:
    *   **Name**: `fraud-detection-api`
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
    *   **Instance Type**: `Free`
4.  **Wait for Deployment**: Render will build your app and give you a URL like `https://fraud-detection-api.onrender.com`. **Copy this URL.**

---

## 🎨 Part 2: Deploying the Frontend on Streamlit Cloud

1.  **Go to [share.streamlit.io](https://share.streamlit.io/)**.
2.  **Deploy a New App**:
    *   Connect your GitHub repo.
    *   **Main file path**: `app.py`
3.  **Set Environment Variables (Critical)**:
    *   Before clicking "Deploy", click **"Advanced settings..."**.
    *   Under **Secrets**, add your Backend URL like this:
        ```toml
        BACKEND_URL = "https://your-api-url-from-render.onrender.com"
        ```
4.  **Click Deploy!**

---

## 🔗 How it Works
The frontend (`app.py`) is now coded to look for an environment variable called `BACKEND_URL`. 
- **Locally**: It defaults to `http://127.0.0.1:8000`.
- **In the Cloud**: It uses the URL you provided in the Streamlit Secrets, allowing the two apps to talk to each other globally.

## 🛠️ Testing Locally with Uvicorn
If you want to run the "Uvicorn Server" on your local machine:
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
This starts the production-ready server and will auto-restart when you change the code.
