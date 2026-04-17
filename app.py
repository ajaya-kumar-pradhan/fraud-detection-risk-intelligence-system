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
    
    # Stability & Premium Theme CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* 1. Global Reset & Theme */
        html, body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 10% 20%, rgb(5, 8, 16) 0%, rgb(10, 15, 30) 90%);
            color: #d1d5db;
            font-family: 'Outfit', sans-serif;
            overflow-x: hidden;
            overflow-anchor: none;
        }

        /* 2. Glassmorphism Design Token */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border: 1px solid rgba(0, 242, 255, 0.3);
            box-shadow: 0 0 20px rgba(0, 242, 255, 0.1);
        }

        /* 3. Metric Overhaul */
        [data-testid="stMetric"] {
            background: rgba(0, 0, 0, 0.2) !important;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        [data-testid="stMetricValue"] {
            color: #00f2ff !important;
            font-weight: 800 !important;
        }

        /* 4. Sidebar Modernization */
        section[data-testid="stSidebar"] {
            background-color: rgba(5, 8, 16, 0.5) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* 5. Custom Buttons */
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #00f2ff, #0077ff);
            color: #050810 !important;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 242, 255, 0.3);
            background: linear-gradient(90deg, #00ff88, #00f2ff);
        }

        /* 6. Typography & Headings */
        h1, h2, h3 {
            font-weight: 800 !important;
            background: linear-gradient(to right, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 0px !important;
            letter-spacing: -0.5px;
        }

        /* 7. Stability Fix */
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
    
    # 🛰️ High-Tech Top Bar
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 15px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 2rem;">🛡️</span>
                <div>
                    <h2 style="margin: 0; line-height: 1;">FRAUD RISK INTELLIGENCE</h2>
                    <p style="margin: 0; font-size: 0.7rem; color: #00f2ff; letter-spacing: 2px;">V5.0 MONOLITHIC ENGINE • LIVE</p>
                </div>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 0.7rem; color: #94a3b8;">SYSTEM STATUS</p>
                <p style="margin: 0; font-size: 0.8rem; color: #00ff88;">● OPERATIONAL</p>
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
                        <div style="text-align: center; padding: 30px 10px; background: rgba(255, 0, 85, 0.05); border-radius: 20px; border: 1px solid #ff0055;">
                            <h3 style="color: #ff0055; margin: 0; -webkit-text-fill-color: #ff0055;">⚠️ FRAUDULENT</h3>
                            <p style="color: #d1d5db; font-size: 0.8rem; margin: 10px 0;">CRITICAL RISK IDENTIFIED</p>
                            <h1 style="color: #fff; margin: 0; -webkit-text-fill-color: #fff;">{proba:.1%}<br><span style="font-size: 0.8rem; color: #94a3b8;">FRAUD PROBABILITY</span></h1>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="text-align: center; padding: 30px 10px; background: rgba(0, 255, 136, 0.05); border-radius: 20px; border: 1px solid #00ff88;">
                            <h3 style="color: #00ff88; margin: 0; -webkit-text-fill-color: #00ff88;">✅ LEGITIMATE</h3>
                            <p style="color: #d1d5db; font-size: 0.8rem; margin: 10px 0;">STABLE RISK PROFILE</p>
                            <h1 style="color: #fff; margin: 0; -webkit-text-fill-color: #fff;">{(1-proba):.1%}<br><span style="font-size: 0.8rem; color: #94a3b8;">TRUST SCORE</span></h1>
                        </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown(f"""
                    <div style="padding-left: 20px;">
                        <p style="margin-bottom: 5px; color: #94a3b8; font-size: 0.8rem;">ENGINE ANALYSIS</p>
                        <p style="font-size: 0.95rem; line-height: 1.4; color: #d1d5db;">
                        The transaction for <b>${payload['amount']:,.2f}</b> ({payload['type']}) has been evaluated against historical structural anomalies. 
                        Verdict: { 'High priority review required' if proba > 0.8 else 'Manual audit recommended' if proba > 0.4 else 'Systemic verification successful' }.
                        </p>
                        <div style="margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px; border-left: 4px solid #00f2ff;">
                            <span style="font-size: 0.75rem; color: #00f2ff; letter-spacing: 1px;">RISK GRADIENT SCAN</span><br>
                            <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; margin-top: 8px;">
                                <div style="width: {proba*100}%; height: 100%; background: linear-gradient(90deg, #00f2ff, #0077ff); border-radius: 4px; box-shadow: 0 0 15px rgba(0, 242, 255, 0.5);"></div>
                            </div>
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
            fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='none')
            ax.set_facecolor('none')
            ax.plot(sens_df["Multiplier"], sens_df["Prob"], marker='o', color='#00f2ff', linewidth=3, markersize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Risk %", color='#94a3b8')
            ax.set_xlabel("Transaction Multiplier", color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_color((1, 1, 1, 0.1))
            ax.grid(axis='y', linestyle='--', alpha=0.1)
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
            
            fig_s, ax_s = plt.subplots(figsize=(7, 4), facecolor='none')
            ax_s.set_facecolor('none')
            colors = ['#ff0055' if v > 0 else '#00ff88' for v in shap_df['value']]
            ax_s.barh(shap_df['feature'], shap_df['value'], color=colors)
            ax_s.invert_yaxis()
            ax_s.set_xlabel("SHAP Impact Score", color='#94a3b8')
            ax_s.tick_params(colors='#94a3b8')
            for spine in ax_s.spines.values():
                spine.set_color((1, 1, 1, 0.1))
            st.pyplot(fig_s)
            
            # Diagnostic Text
            top_f = shap_df.iloc[0]['feature']
            diag_style = st.info if not is_fraud else st.warning
            diag_style(f"**Diagnostic Summary:** Primary risk driver identified as `{top_f}`. This factor is pushing the risk scoring toward {'Fraud' if shap_df.iloc[0]['value'] > 0 else 'Legitimate'} due to its statistical variance from historical baseline.")

    else:
        # 🛸 Cyberpunk Landing Page (Stabilized)
        st.markdown('<div class="glass-card" style="padding: 80px 40px; text-align: center; max-width: 900px; margin: 0 auto;">', unsafe_allow_html=True)
        
        st.markdown("""
            <h1 style="margin-bottom: 25px; font-size: 3.5rem;">NEXT-GEN RISK<br><span style="color: #00f2ff; -webkit-text-fill-color: #00f2ff;">GOVERNANCE</span></h1>
            <p style="font-size: 1.2rem; color: #94a3b8; line-height: 1.6; max-width: 700px; margin: 0 auto;">
            This dashboard utilizes a high-performance <b>Monolithic Architecture</b> for real-time financial tracking and anomaly detection.
            </p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 50px; text-align: left;">
                <div style="padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <span style="color: #00f2ff; font-weight: 800; font-size: 1.1rem;">⚡ ZERO LATENCY</span><br>
                    <span style="font-size: 0.9rem; color: #64748b;">Direct In-Memory Inference engine for instant scoring.</span>
                </div>
                <div style="padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <span style="color: #00ff88; font-weight: 800; font-size: 1.1rem;">🔍 EXPLAINABLE</span><br>
                    <span style="font-size: 0.9rem; color: #64748b;">Real-time SHAP local explanations for full transparency.</span>
                </div>
                <div style="padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <span style="color: #ff0055; font-weight: 800; font-size: 1.1rem;">🧪 STRESS TEST</span><br>
                    <span style="font-size: 0.9rem; color: #64748b;">Built-in sensitivity analysis for "What-If" scenarios.</span>
                </div>
                <div style="padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <span style="color: #7000ff; font-weight: 800; font-size: 1.1rem;">📝 AUDIT READY</span><br>
                    <span style="font-size: 0.9rem; color: #64748b;">Automated balance consistency and audit logging.</span>
                </div>
            </div>
            
            <div style="margin-top: 50px; padding-top: 30px; border-top: 1px solid rgba(255,255,255,0.05);">
                <p style="color: #94a3b8; font-style: italic; font-size: 1rem; opacity: 0.8;">
                "Empowering financial institutions with transparent, deep-learning-driven risk mitigation."
                </p>
                <p style="color: #00f2ff; font-size: 0.7rem; margin-top: 15px; letter-spacing: 2px;">READY FOR SYSTEM SCAN</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
