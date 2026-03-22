import sys
import os
from pathlib import Path

# --- THE FIX: DYNAMIC PATH INJECTION ---
current_dir = Path(__file__).resolve().parent
root_path = current_dir.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd
import joblib

# These imports will now work perfectly
from src.data_processor import load_and_merge_data
from src.model import train_model

st.set_page_config(page_title="Health Equity Dashboard", layout="wide")

try:
    # 1. Load Data
    df, report = load_and_merge_data()

    # 2. Handle Model
    model_path = root_path / "models" / "cost_predictor.pkl"
    if not model_path.exists():
        with st.spinner("Initializing Predictive Engine..."):
            train_model(df)
    
    model = joblib.load(model_path)

    # 3. UI
    st.title("🏥 Health Equity Insights Dashboard")
    st.success("System Online: Analytics and Model Loaded.")
    st.dataframe(report)

except Exception as e:
    st.error("Dashboard Initialization Error")
    st.exception(e)
