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
    
    # 🏛️ Ultra-Minimalist Stability System
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* 1. Global Stability Reset */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
            color: #1e293b;
            font-family: 'Outfit', sans-serif;
            overflow: hidden !important; 
        }

        /* 2. Flat Minimal Cards */
        .glass-card {
            background: #ffffff;
            border: 1px solid #f1f5f9;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }

        /* 3. Static Metrics (No-Shaking) */
        [data-testid="stMetric"] {
            background: #f8fafc !important;
            border: 1px solid #f1f5f9;
            border-radius: 6px;
            padding: 10px !important;
            height: 90px !important; /* Fixed height to prevent vertical jitter */
            overflow: hidden;
        }
        [data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #64748b !important; }
        [data-testid="stMetricValue"] { font-size: 1.3rem !important; color: #0f172a !important; }

        /* 4. Minimal Buttons */
        .stButton>button {
            background: #0f172a;
            color: #ffffff !important;
            border-radius: 6px;
            padding: 8px;
            font-size: 0.9rem;
            transition: none !important;
        }

        /* 5. Typography */
        h1, h2, h3, h4 {
            color: #0f172a !important;
            margin: 0 !important;
            padding-bottom: 8px !important;
        }
        
        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }
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
        
        # 📊 Intelligence Result (Redesigned)
        if config["submit"]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔮 Intelligence Result")
            res_col1, res_col2 = st.columns([1, 1.5])
            
            with res_col1:
                if is_fraud:
                    st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background: #fff1f2; border-radius: 12px; border: 1px solid #fda4af;">
                            <h4 style="color: #be123c; margin: 0;">⚠️ FRAUD</h4>
                            <h2 style="color: #be123c; margin: 0;">{proba:.1%}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background: #f0fdf4; border-radius: 12px; border: 1px solid #bbf7d0;">
                            <h4 style="color: #15803d; margin: 0;">✅ SAFE</h4>
                            <h2 style="color: #15803d; margin: 0;">{(1-proba):.1%}</h2>
                        </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown(f"""
                    <div style="padding-left: 10px;">
                        <p style="font-size: 0.9rem; color: #334155;">
                        <b>Scan Verdict:</b> { 'High Risk' if proba > 0.8 else 'Manual Review' if proba > 0.4 else 'Safe' }.
                        </p>
                        <div style="width: 100%; height: 6px; background: #e2e8f0; border-radius: 3px; margin-top: 10px; overflow: hidden;">
                            <div style="width: {proba*100}%; height: 100%; background: #4f46e5; border-radius: 3px;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Sub-metrics (Mini Cards)
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Fraud Probability", f"{proba*100:.1f}%")
            with m_col2:
                st.metric("Risk Status", "CRITICAL" if proba > 0.8 else "ELEVATED" if proba > 0.4 else "STABLE")
            with m_col3:
                st.metric("System Verdict", "FRAUDULENT" if is_fraud else "LEGITIMATE")
            
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
            fig, ax = plt.subplots(figsize=(5, 2), facecolor='none')
            ax.set_facecolor('none')
            ax.plot(sens_df["Multiplier"], sens_df["Prob"], marker='o', color='#4f46e5', linewidth=1.5, markersize=5)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7, colors='#64748b')
            for spine in ax.spines.values():
                spine.set_color('#e2e8f0')
            ax.grid(axis='y', linestyle='--', alpha=0.2)
            st.pyplot(fig, use_container_width=True)
            
        with right:
            st.subheader("🔍 Factors")
            # SHAP Explanation
            shap_values = explainer.shap_values(df_processed)
            feature_imp = [
                {"feature": f, "value": float(val)} 
                for f, val in zip(features, shap_values[0])
            ]
            feature_imp.sort(key=lambda x: abs(x["value"]), reverse=True)
            shap_df = pd.DataFrame(feature_imp).head(4)
            
            fig_s, ax_s = plt.subplots(figsize=(5, 2), facecolor='none')
            ax_s.set_facecolor('none')
            colors = ['#e11d48' if v > 0 else '#16a34a' for v in shap_df['value']]
            ax_s.barh(shap_df['feature'], shap_df['value'], color=colors)
            ax_s.invert_yaxis()
            ax_s.tick_params(labelsize=7, colors='#64748b')
            for spine in ax_s.spines.values():
                spine.set_color('#e2e8f0')
            st.pyplot(fig_s, use_container_width=True)
            
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
