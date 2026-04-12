import os
import joblib
import pandas as pd
import shap
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Real-Time Fraud Detection API")

# Load model and artifacts
base_dir = os.path.join(os.path.dirname(__file__), "model_artifacts")
model = xgb.XGBClassifier()
model.load_model(os.path.join(base_dir, 'xgboost_fraud_model.json'))
features = joblib.load(os.path.join(base_dir, 'feature_columns.pkl'))

with open(os.path.join(base_dir, 'threshold.txt'), 'r') as f:
    threshold = float(f.read().strip())

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    
def preprocess(tx: Transaction) -> pd.DataFrame:
    df = pd.DataFrame([tx.dict()])
    
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['hour_of_day'] = df['step'] % 24
    
    # Feature alignment
    out_dict = {}
    for col in features:
        if col == 'errorBalanceOrig': out_dict[col] = df['errorBalanceOrig'].iloc[0]
        elif col == 'errorBalanceDest': out_dict[col] = df['errorBalanceDest'].iloc[0]
        elif col == 'hour_of_day': out_dict[col] = df['hour_of_day'].iloc[0]
        elif col == 'step': out_dict[col] = df['step'].iloc[0]
        elif col == 'amount': out_dict[col] = df['amount'].iloc[0]
        elif col == 'oldbalanceOrg': out_dict[col] = df['oldbalanceOrg'].iloc[0]
        elif col == 'newbalanceOrig': out_dict[col] = df['newbalanceOrig'].iloc[0]
        elif col == 'oldbalanceDest': out_dict[col] = df['oldbalanceDest'].iloc[0]
        elif col == 'newbalanceDest': out_dict[col] = df['newbalanceDest'].iloc[0]
        elif col.startswith('type_'):
            type_val = col.split('_')[1]
            out_dict[col] = 1 if df['type'].iloc[0] == type_val else 0
        else:
            out_dict[col] = 0
            
    return pd.DataFrame([out_dict])

@app.post("/predict")
def predict_fraud(tx: Transaction):
    try:
        df = preprocess(tx)
        proba = model.predict_proba(df)[0][1]
        is_fraud = bool(proba >= threshold)
        risk_score = min(int(proba * 100), 100)
        
        return {
            "is_fraud": is_fraud,
            "risk_score": risk_score,
            "probability": float(proba),
            "threshold": threshold
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain")
def explain_fraud(tx: Transaction):
    try:
        df = preprocess(tx)
        shap_values = explainer.shap_values(df)
        
        feature_importance = [
            {"feature": f, "value": float(val)} 
            for f, val in zip(features, shap_values[0])
        ]
        feature_importance.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        explainer_expected_value = float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0])
        
        return {
            "expected_value": explainer_expected_value,
            "shap_values": feature_importance
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
