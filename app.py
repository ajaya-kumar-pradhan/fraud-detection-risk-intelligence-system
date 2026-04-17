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
        layout="centered"
    )
    
    # 🏛️ Airline-Style Stability System
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #f8faff !important;
            color: #1e293b;
            font-family: 'Inter', sans-serif;
        }

        .glass-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }

        [data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 15px !important;
        }

        .stButton>button {
            width: 100%;
            background: #4f46e5;
            color: white !important;
            border-radius: 10px;
            padding: 12px;
            font-weight: 600;
            border: none;
        }

        h1, h2, h3 {
            color: #0f172a !important;
            font-weight: 800 !important;
            margin-top: 0px !important;
        }
        
        #MainMenu, footer, header { visibility: hidden; }
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
    
    # 📋 Premium Header (Light)
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; border-bottom: 1px solid #e2e8f0; padding-bottom: 15px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 2rem;">🛡️</span>
                <div>
                    <h2 style="margin: 0; line-height: 1; color: #0f172a;">FRAUD RISK INTELLIGENCE</h2>
                    <p style="margin: 0; font-size: 0.7rem; color: #4f46e5; letter-spacing: 2px; font-weight: 700;">V5.0 ENTERPRISE PLATFORM • ONLINE</p>
                </div>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 0.7rem; color: #64748b;">SYSTEM STATUS</p>
                <p style="margin: 0; font-size: 0.8rem; color: #16a34a; font-weight: 800;">● OPERATIONAL</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
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
        
        # 📊 Intelligence Result (Centered & Stable)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔮 Intelligence Result")
        
        if is_fraud:
            st.markdown(f"""
                <div style="text-align: center; padding: 25px; background: #fff1f2; border-radius: 12px; border: 1px solid #fda4af; margin-bottom: 20px;">
                    <h4 style="color: #be123c; margin: 0;">⚠️ FRAUDULENT DETECTED</h4>
                    <h1 style="color: #be123c; margin: 0; font-size: 3rem;">{proba:.1%}</h1>
                    <p style="color: #9f1239; margin-top: 10px;">HIGH RISK PROFILE IDENTIFIED</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="text-align: center; padding: 25px; background: #f0fdf4; border-radius: 12px; border: 1px solid #bbf7d0; margin-bottom: 20px;">
                    <h4 style="color: #15803d; margin: 0;">✅ LEGITIMATE VERIFIED</h4>
                    <h1 style="color: #15803d; margin: 0; font-size: 3rem;">{(1-proba):.1%}</h1>
                    <p style="color: #166534; margin-top: 10px;">SYSTEMIC TRUST VERIFIED</p>
                </div>
            """, unsafe_allow_html=True)

        # Sub-metrics
        st.metric("Risk Probability", f"{proba*100:.1f}%")
        st.metric("Risk Status", "CRITICAL" if proba > 0.8 else "ELEVATED" if proba > 0.4 else "STABLE")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()
        
        st.subheader("📊 Engine Stability")
        sens_df = pd.DataFrame(sens_results)
        fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='none')
        ax.set_facecolor('none')
        ax.plot(sens_df["Multiplier"], sens_df["Prob"], marker='o', color='#4f46e5', linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8, colors='#64748b')
        for spine in ax.spines.values():
            spine.set_color('#e2e8f0')
        st.pyplot(fig)
        
        st.subheader("🔍 Factors")
        # SHAP Explanation
        shap_values = explainer.shap_values(df_processed)
        feature_imp = [
            {"feature": f, "value": float(val)} 
            for f, val in zip(features, shap_values[0])
        ]
        feature_imp.sort(key=lambda x: abs(x["value"]), reverse=True)
        shap_df = pd.DataFrame(feature_imp).head(4)
        
        fig_s, ax_s = plt.subplots(figsize=(6, 2.5), facecolor='none')
        ax_s.set_facecolor('none')
        colors = ['#e11d48' if v > 0 else '#16a34a' for v in shap_df['value']]
        ax_s.barh(shap_df['feature'], shap_df['value'], color=colors)
        ax_s.invert_yaxis()
        ax_s.tick_params(labelsize=8, colors='#64748b')
        for spine in ax_s.spines.values():
            spine.set_color('#e2e8f0')
        st.pyplot(fig_s)
            
        # Diagnostic Text
        top_f = shap_df.iloc[0]['feature']
        diag_style = st.info if not is_fraud else st.warning
        diag_style(f"**Diagnostic Summary:** Primary risk driver identified as `{top_f}`. This factor is pushing the risk scoring toward {'Fraud' if shap_df.iloc[0]['value'] > 0 else 'Legitimate'} due to its statistical variance from historical baseline.")

    else:
        # 🏛️ Minimalist Landing Page
        st.markdown('<div class="glass-card" style="max-width: 800px; margin: 0 auto; text-align: center; padding: 60px 20px;">', unsafe_allow_html=True)
        
        st.markdown("""
            <h1 style="font-size: 2.5rem; margin-bottom: 20px;">FRAUD RISK<br><span style="color: #4f46e5;">INTELLIGENCE</span></h1>
            <p style="color: #64748b; font-size: 1.1rem; line-height: 1.6;">
            A high-performance intelligence system for real-time financial tracking and anomaly detection.
            </p>
            <div style="text-align: left; max-width: 500px; margin: 40px auto; background: #f8fafc; padding: 25px; border-radius: 12px; border: 1px solid #e2e8f0;">
                <p style="margin-bottom: 12px;">✅ <b>Instant Scoring:</b> Zero-latency in-memory engine.</p>
                <p style="margin-bottom: 12px;">✅ <b>Explainable AI:</b> Transparent SHAP diagnostics.</p>
                <p style="margin-bottom: 12px;">✅ <b>Audit Ready:</b> Automated ledger validation.</p>
            </div>
            <p style="color: #94a3b8; font-size: 0.8rem; letter-spacing: 1px;">Ready to initiate scan. Use sidebar to start.</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
