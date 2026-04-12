import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration & Constants
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def setup_page():
    """Configures the aesthetic layout of the dashboard."""
    st.set_page_config(
        page_title="Fraud Risk Intelligence",
        page_icon="🛡️",
        layout="wide"
    )
    
    # Custom CSS for a refined, professional appearance
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stMetric {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .stButton>button {
            border-radius: 5px;
            font-weight: 600;
        }
        h1, h2, h3 {
            color: #1e293b;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_content_allowed=True)

class ControlPanel:
    """Handles the sidebar configuration and transaction data entry."""
    @staticmethod
    def render():
        with st.sidebar:
            st.title("🛡️ Risk Parameters")
            st.caption("Configure transaction details for real-time analysis")
            
            with st.container():
                step = st.number_input("Step (Hours since start)", min_value=1, value=1)
                type_tx = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
                amount = st.number_input("Amount ($)", min_value=0.0, value=2500.0, step=100.0)
                
                st.divider()
                st.subheader("Balance Audit")
                oldbalanceOrg = st.number_input("Sender Original Balance", value=5000.0)
                newbalanceOrig = st.number_input("Sender Final Balance", value=2500.0)
                
                st.divider()
                oldbalanceDest = st.number_input("Recipient Original Balance", value=0.0)
                newbalanceDest = st.number_input("Recipient Final Balance", value=2500.0)
                
            return {
                "step": step,
                "type": type_tx,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "submit": st.button("Analyze Risk Profile", type="primary", use_container_width=True)
            }

class RiskAnalytics:
    """Encapsulates prediction logic and visual reporting."""
    
    @staticmethod
    def fetch_prediction(payload):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Prediction Service Unavailable: {e}")
            return None

    @staticmethod
    def fetch_explanation(payload):
        try:
            response = requests.post(f"{API_URL}/explain", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    @staticmethod
    def display_metrics(data):
        prob = data["probability"]
        is_fraud = data["is_fraud"]
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Fraud Probability", f"{prob*100:.1f}%")
        with cols[1]:
            risk_label = "CRITICAL" if prob > 0.8 else "ELEVATED" if prob > 0.4 else "STABLE"
            st.metric("Risk Status", risk_label)
        with cols[2]:
            st.metric("System Verdict", "FRAUDULENT" if is_fraud else "LEGITIMATE")

    @staticmethod
    def plot_sensitivity(api_payload):
        st.subheader("📊 Volatility & Sensitivity")
        st.write("Impact of transaction volume on probability score")
        
        variances = [0.5, 1.0, 2.0, 5.0]
        base_amt = api_payload['amount']
        results = []
        
        # Calculate structural offsets for consistent 'what-if' mapping
        orig_offset = api_payload["newbalanceOrig"] + base_amt - api_payload["oldbalanceOrg"]
        dest_offset = api_payload["oldbalanceDest"] + base_amt - api_payload["newbalanceDest"]

        for mult in variances:
            test_amt = base_amt * mult
            test_payload = api_payload.copy()
            test_payload["amount"] = test_amt
            test_payload["newbalanceOrig"] = max(0.0, api_payload["oldbalanceOrg"] - test_amt + orig_offset)
            test_payload["newbalanceDest"] = max(0.0, api_payload["oldbalanceDest"] + test_amt - dest_offset)
            
            res = RiskAnalytics.fetch_prediction(test_payload)
            if res:
                results.append({"Multiplier": f"{mult}x", "Prob": res["probability"]})
        
        if results:
            df = pd.DataFrame(results)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(df["Multiplier"], df["Prob"], marker='o', color='#2563eb', linewidth=2)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

def main():
    setup_page()
    
    st.title("Fraud Risk Intelligence Dashboard")
    st.write("Real-time transactional integrity analysis & risk scoring engine")
    
    config = ControlPanel.render()
    
    if config["submit"]:
        payload = {k: v for k, v in config.items() if k != "submit"}
        
        with st.status("Analyzing Transactional DNA...", expanded=True) as status:
            st.write("Contacting intelligence API...")
            data = RiskAnalytics.fetch_prediction(payload)
            
            if data:
                status.update(label="Analysis Complete", state="complete", expanded=False)
                
                # Summary Alerts
                if data["probability"] > 0.8:
                    st.error("🚨 **High Risk Identified**: Transaction exhibits structural anomalies consistent with known fraud patterns.")
                elif data["probability"] > 0.4:
                    st.warning("⚠️ **Review Recommended**: Moderate risk profile detected. Verify recipient authenticity.")
                else:
                    st.success("✅ **Risk Verified**: Transaction appears legitimate within current systemic parameters.")
                
                RiskAnalytics.display_metrics(data)
                st.divider()
                
                left, right = st.columns(2)
                
                with left:
                    RiskAnalytics.plot_sensitivity(payload)
                
                with right:
                    st.subheader("🔍 Interpretability Drivers")
                    st.write("Mathematical signals contributing to the current risk score")
                    
                    explain_data = RiskAnalytics.fetch_explanation(payload)
                    if explain_data:
                        shap_df = pd.DataFrame(explain_data["shap_values"]).head(6)
                        # Stylized horizontal bar chart
                        fig, ax = plt.subplots(figsize=(7, 4))
                        colors = ['#dc2626' if v > 0 else '#16a34a' for v in shap_df['value']]
                        ax.barh(shap_df['feature'], shap_df['value'], color=colors)
                        ax.invert_yaxis()
                        st.pyplot(fig)
                        
                        # Conversational AI Diagnostic
                        st.divider()
                        top_feat = shap_df.iloc[0]['feature']
                        impact = "increased" if shap_df.iloc[0]['value'] > 0 else "decreased"
                        
                        diag_style = st.info if not data["is_fraud"] else st.warning
                        diag_style(f"**Diagnostic Summary:** The primary risk driver is `{top_feat}`, which {impact} the overall probability. This indicates a deviation in established balance flow behavior.")

    else:
        # Welcome State
        st.info("Input transaction details in the sidebar to begin risk assessment.")
        
        # Placeholder/Landing visuals
        colA, colB = st.columns(2)
        with colA:
            st.image("https://img.freepik.com/free-vector/security-concept-illustration_114360-463.jpg", use_container_width=True)
        with colB:
            st.markdown("""
                ### System Features
                - **Real-time Scoring**: Instant processing of transactional metadata.
                - **Deep Explanation**: SHAP-based feature importance mapping.
                - **What-If Simulations**: Test transaction sensitivity to amount variations.
                - **Secure API Integration**: Backed by a high-concurrency XGBoost engine.
            """)

if __name__ == "__main__":
    main()
