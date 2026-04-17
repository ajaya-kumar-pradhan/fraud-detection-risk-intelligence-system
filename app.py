import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import shap
import xgboost as xgb

# ================================
# ⚙️ SYSTEM CONFIGURATION
# ================================

def setup_page():
    """Configures the aesthetic layout of the dashboard."""
    st.set_page_config(
        page_title="Fraud Risk Intelligence",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        /* Fix for layout shaking */
        html {
            overflow-anchor: none;
            scroll-behavior: auto;
        }
        .main {
            background-color: #f8f9fa;
        }
        .stMetric {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            white-space: nowrap;
        }
        .stButton>button {
            border-radius: 5px;
            font-weight: 600;
        }
        /* Prevent header jumping */
        h1, h2, h3 {
            color: #1e293b;
            margin-top: 0px !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ================================
# 🧠 INTELLIGENCE ENGINE (MONOLITHIC)
# ================================

@st.cache_resource
def load_assets():
    """Loads model and artifacts once and caches them for performance. Checks multiple locations."""
    # List of possible locations
    base_dirs = [
        os.path.join(os.path.dirname(__file__), "model_artifacts"),
        os.path.dirname(__file__) # Root directory
    ]
    
    model_path = None
    features_path = None
    threshold_path = None
    
    # Search for the files
    for d in base_dirs:
        m = os.path.join(d, 'xgboost_fraud_model.json')
        f = os.path.join(d, 'feature_columns.joblib')
        t = os.path.join(d, 'threshold.txt')
        
        if os.path.exists(m) and os.path.exists(f) and os.path.exists(t):
            model_path, features_path, threshold_path = m, f, t
            break
            
    if not model_path:
        return None, None, None, None
        
    # Load XGBoost Model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Load Feature Columns
    features = joblib.load(features_path)
    
    # Load Threshold
    with open(threshold_path, 'r') as f_ptr:
        threshold = float(f_ptr.read().strip())
        
    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    return model, features, threshold, explainer

def preprocess_transaction(payload, features):
    """Local preprocessing logic migrated from api.py."""
    df = pd.DataFrame([payload])
    
    # Engineering smart features
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['hour_of_day'] = df['step'] % 24
    
    # Feature Alignment (One-Hot Encoding handling)
    out_dict = {}
    for col in features:
        if col in df.columns:
            out_dict[col] = df[col].iloc[0]
        elif col.startswith('type_'):
            type_val = col.split('_')[1]
            out_dict[col] = 1 if payload['type'] == type_val else 0
        else:
            out_dict[col] = 0
            
    return pd.DataFrame([out_dict])

# ================================
# 📊 UI COMPONENTS
# ================================

class ControlPanel:
    """Handles the sidebar configuration and transaction data entry."""
    @staticmethod
    def render():
        with st.sidebar:
            st.title("🛡️ Risk Parameters")
            st.caption("Configure transaction details for direct analysis")
            
            step = st.number_input("Step (Hours since start)", min_value=1, value=1)
            type_tx = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
            amount = st.number_input("Amount ($)", min_value=0.0, value=2500.0, step=100.0)
            
            st.divider()
            st.subheader("Account Verification")
            old_orig = st.number_input("Sender Original Balance", value=5000.0)
            new_orig = st.number_input("Sender Final Balance", value=2500.0)
            old_dest = st.number_input("Recipient Original Balance", value=0.0)
            new_dest = st.number_input("Recipient Final Balance", value=2500.0)
            
            submit = st.button("Analyze Risk Profile", type="primary", use_container_width=True)
            
            return {
                "payload": {
                    "step": step, "type": type_tx, "amount": amount,
                    "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
                    "oldbalanceDest": old_dest, "newbalanceDest": new_dest
                },
                "submit": submit
            }

def main():
    setup_page()
    
    # Header
    st.title("Fraud Risk Intelligence Dashboard")
    st.write("Direct in-memory risk scoring & pattern analysis")
    
    # Load Intelligence Engine
    with st.spinner("Initializing Intelligence Engine..."):
        model, features, threshold, explainer = load_assets()
    
    config = ControlPanel.render()
    
    if config["submit"]:
        payload = config["payload"]
        
        # 1. Prediction Workflow
        df_processed = preprocess_transaction(payload, features)
        proba = model.predict_proba(df_processed)[0][1]
        is_fraud = bool(proba >= threshold)
        
        # Display Status
        if proba > 0.8:
            st.error("🚨 **High Risk Identified**: Transaction exhibits structural anomalies consistent with fraud patterns.")
        elif proba > 0.4:
            st.warning("⚠️ **Review Recommended**: Moderate risk profile detected. Verify recipient authenticity.")
        else:
            st.success("✅ **Risk Verified**: Transaction appears legitimate within current systemic parameters.")
            
        # 2. Metric Reporting
        cols = st.columns(3)
        cols[0].metric("Fraud Probability", f"{proba*100:.1f}%")
        cols[1].metric("Risk Status", "CRITICAL" if proba > 0.8 else "ELEVATED" if proba > 0.4 else "STABLE")
        cols[2].metric("System Verdict", "FRAUDULENT" if is_fraud else "LEGITIMATE")
        
        st.divider()
        
        left, right = st.columns(2)
        
        with left:
            st.subheader("📊 Performance Volatility")
            # Sensitivity Analysis
            variances = [0.5, 1.0, 2.0, 5.0]
            sens_results = []
            orig_offset = payload["newbalanceOrig"] + payload["amount"] - payload["oldbalanceOrg"]
            dest_offset = payload["oldbalanceDest"] + payload["amount"] - payload["newbalanceDest"]
            
            for mult in variances:
                test_amt = payload["amount"] * mult
                test_payload = payload.copy()
                test_payload["amount"] = test_amt
                test_payload["newbalanceOrig"] = max(0.0, payload["oldbalanceOrg"] - test_amt + orig_offset)
                test_payload["newbalanceDest"] = max(0.0, payload["oldbalanceDest"] + test_amt - dest_offset)
                
                t_df = preprocess_transaction(test_payload, features)
                t_prob = model.predict_proba(t_df)[0][1]
                sens_results.append({"Multiplier": f"{mult}x", "Prob": t_prob})
            
            sens_df = pd.DataFrame(sens_results)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(sens_df["Multiplier"], sens_df["Prob"], marker='o', color='#2563eb', linewidth=2)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Risk %")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
        with right:
            st.subheader("🔍 Interpretability Drivers")
            # SHAP Explanation
            shap_values = explainer.shap_values(df_processed)
            feature_imp = [
                {"feature": f, "value": float(val)} 
                for f, val in zip(features, shap_values[0])
            ]
            feature_imp.sort(key=lambda x: abs(x["value"]), reverse=True)
            shap_df = pd.DataFrame(feature_imp).head(6)
            
            fig_s, ax_s = plt.subplots(figsize=(7, 4))
            colors = ['#dc2626' if v > 0 else '#16a34a' for v in shap_df['value']]
            ax_s.barh(shap_df['feature'], shap_df['value'], color=colors)
            ax_s.invert_yaxis()
            ax_s.set_xlabel("SHAP Value (Risk Impact)")
            st.pyplot(fig_s)
            
            # Diagnostic Text
            top_f = shap_df.iloc[0]['feature']
            diag_style = st.info if not is_fraud else st.warning
            diag_style(f"**Diagnostic Summary:** Primary risk driver identified as `{top_f}`. This factor is pushing the risk scoring toward {'Fraud' if shap_df.iloc[0]['value'] > 0 else 'Legitimate'} due to its statistical variance from historical baseline.")

    else:
        # Landing Page
        colA, colB = st.columns([1, 1.2])
        with colA:
            st.image("https://img.freepik.com/free-vector/security-concept-illustration_114360-463.jpg", use_container_width=True)
        with colB:
            st.markdown("""
                ### Integrated Governance Engine
                This dashboard utilizes a monolithic architecture for high-performance risk monitoring.
                
                - **Direct In-Memory Inference**: Zero latency from external API calls.
                - **Explainable Insights**: Real-time SHAP analysis for transaction transparency.
                - **Stress Testing**: Built-in "What-If" scenarios to map risk sensitivity.
                - **Accounting Audit**: Automated validation of transactional balance consistency.
                
                *Select transaction parameters in the sidebar to initiate a scan.*
            """)

if __name__ == "__main__":
    main()
