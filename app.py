import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import shap
import xgboost as xgb

# ==========================================
# 🛡️ TRANSACTION FRAUD DETECTOR
# Professional financial analysis dashboard.
# ==========================================

def setup_app():
    """Sets up the professional interface and styling."""
    st.set_page_config(
        page_title="Fraud Detector",
        page_icon="🛡️",
        layout="centered"
    )

    # 👔 Professional "Clean Finance" Styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* 1. Reset to Standard Colors */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #fcfcfc !important;
            color: #334155;
            font-family: 'Inter', sans-serif;
        }

        /* 2. Professional Dashboard Cards */
        .report-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        /* 3. Navigation & Actions */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0;
        }
        
        .stButton>button {
            width: 100%;
            background-color: #1e40af; /* Navy Blue */
            color: white !important;
            border-radius: 6px;
            padding: 10px;
            font-weight: 500;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1e3a8a;
        }

        /* 4. Typography Header */
        h1, h2, h3 {
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        
        .status-pill {
            display: inline-block;
            padding: 2px 10px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 100px;
            font-size: 0.7rem;
            font-weight: 600;
            border: 1px solid #e2e8f0;
        }

        /* 5. Hide Elements for Cleanliness */
        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🧠 LOAD ANALYSIS MODELS
# ==========================================

@st.cache_resource
def get_analysis_tools():
    """Loads the fraud model and diagnostic tools."""
    # Check for artifacts in standard locations
    dirs = [
        os.path.join(os.path.dirname(__file__), "model_artifacts"),
        os.path.dirname(__file__)
    ]
    
    m_path, f_path, t_path = None, None, None
    for d in dirs:
        m = os.path.join(d, 'xgboost_fraud_model.json')
        f = os.path.join(d, 'feature_columns.joblib')
        t = os.path.join(d, 'threshold.txt')
        if all(os.path.exists(p) for p in [m, f, t]):
            m_path, f_path, t_path = m, f, t
            break
            
    if not m_path:
        st.error("Model files not found. Please verify the repository structure.")
        return None, None, None, None
        
    model = xgb.XGBClassifier()
    model.load_model(m_path)
    features = joblib.load(f_path)
    
    with open(t_path, 'r') as f_ptr:
        threshold = float(f_ptr.read().strip())
        
    explainer = shap.TreeExplainer(model)
    return model, features, threshold, explainer

def prepare_data(data, features):
    """Processes transaction data for the model."""
    df = pd.DataFrame([data])
    
    # Feature Engineering
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['hour_of_day'] = df['step'] % 24
    
    # Matching model features
    out = {}
    for col in features:
        if col in df.columns:
            out[col] = df[col].iloc[0]
        elif col.startswith('type_'):
            t = col.split('_')[1]
            out[col] = 1 if data['type'] == t else 0
        else:
            out[col] = 0
    return pd.DataFrame([out])

# ==========================================
# 🖥️ USER INTERFACE
# ==========================================

def run_app():
    setup_app()
    
    # Header Section
    st.markdown("""
        <div style="margin-bottom: 30px;">
            <p class="status-pill">SECURE SESSION ENCRYPTED</p>
            <h1 style="margin: 0; font-size: 2.2rem;">Fraud Detection App</h1>
            <p style="color: #64748b; font-size: 0.9rem;">Analyze and audit transaction risk in real-time.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load Tools
    model, features, threshold, explainer = get_analysis_tools()
    if not model: return

    # Sidebar: Input Form
    with st.sidebar:
        st.markdown("### Transaction Details")
        st.write("Enter the transaction data below to check for risk.")
        
        step = st.number_input("Time Step (Hour)", min_value=1, value=1)
        tx_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
        amount = st.number_input("Amount ($)", min_value=0.0, value=2500.0)
        
        st.write("---")
        st.write("**Account Balances**")
        old_orig = st.number_input("Sender Initial Balance", value=5000.0)
        new_orig = st.number_input("Sender Final Balance", value=2500.0)
        old_dest = st.number_input("Recipient Initial Balance", value=0.0)
        new_dest = st.number_input("Recipient Final Balance", value=2500.0)
        
        analyze = st.button("Run Risk Analysis")

    if analyze:
        payload = {
            "step": step, "type": tx_type, "amount": amount,
            "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest
        }
        
        # Calculation
        data_processed = prepare_data(payload, features)
        prob = model.predict_proba(data_processed)[0][1]
        is_fraud = prob >= threshold
        
        # DISPLAY RESULTS
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("Analysis Result")
        
        if is_fraud:
            st.error(f"**High Risk Detected** ({prob:.1%})")
            st.write("This transaction exhibits patterns consistent with fraudulent activity.")
        else:
            st.success(f"**Legitimate Transaction** (Trust Level: {(1-prob):.1%})")
            st.write("No significant risk patterns were detected for this transaction.")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Detailed Report (Columns)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Risk Volatility")
            # Simple Matplotlib
            v_mults = [0.5, 1.0, 2.0, 5.0]
            v_res = []
            orig_off = payload["newbalanceOrig"] + payload["amount"] - payload["oldbalanceOrg"]
            dest_off = payload["oldbalanceDest"] + payload["amount"] - payload["newbalanceDest"]
            
            for m in v_mults:
                t_p = payload.copy()
                t_p["amount"] = payload["amount"] * m
                t_p["newbalanceOrig"] = max(0.0, payload["oldbalanceOrg"] - t_p["amount"] + orig_off)
                t_p["newbalanceDest"] = max(0.0, payload["oldbalanceDest"] + t_p["amount"] - dest_off)
                t_df = prepare_data(t_p, features)
                v_res.append(model.predict_proba(t_df)[0][1])
            
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(v_mults, v_res, 'o-', color='#1e40af', linewidth=2)
            ax.set_title("Amount Sensitivity", fontsize=10)
            ax.set_ylabel("Risk %", fontsize=8)
            ax.set_xlabel("Transaction Multiplier", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

        with col2:
            st.markdown("#### Primary Drivers")
            # SHAP
            shap_vals = explainer.shap_values(data_processed)
            imps = [{"f": f, "v": float(v)} for f, v in zip(features, shap_vals[0])]
            imps.sort(key=lambda x: abs(x["v"]), reverse=True)
            top_imps = pd.DataFrame(imps).head(5)
            
            fig_s, ax_s = plt.subplots(figsize=(5, 3))
            colors = ['#dc2626' if v > 0 else '#16a34a' for v in top_imps['v']]
            ax_s.barh(top_imps['f'], top_imps['v'], color=colors)
            ax_s.set_title("Feature Impact", fontsize=10)
            ax_s.invert_yaxis()
            st.pyplot(fig_s)
            
        # Summary Note
        st.info(f"**Reviewer Note:** The primary factor contributing to this score is `{imps[0]['f']}`. Standard review protocol is recommended for all transactions with risk above 40%.")

    else:
        # Welcome Page (Human Design)
        st.markdown("""
            <div style="text-align: center; padding: 60px 0;">
                <h2 style="color: #1e40af;">Ready to Audit Transactions</h2>
                <p style="color: #64748b; max-width: 500px; margin: 0 auto;">
                    This application helps you identify high-risk transactions by comparing them with historical patterns in bank ledgers. 
                </p>
                <div style="margin-top: 40px; text-align: left; max-width: 400px; margin-left: auto; margin-right: auto; background: #ffffff; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px;">
                    <p>🔵 <b>Simple Input:</b> Enter Sender/Receiver balances.</p>
                    <p>🔴 <b>Risk Scoring:</b> Get a probability of fraud instantly.</p>
                    <p>🟢 <b>Clear Audit:</b> Understand the exact factors driving the score.</p>
                </div>
                <p style="margin-top: 40px; font-size: 0.8rem; color: #94a3b8;">
                    Enter transaction details in the left sidebar to begin.
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()
