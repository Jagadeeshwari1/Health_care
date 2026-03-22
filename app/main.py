import sys
import os
from pathlib import Path

# --- THE FIX: DYNAMIC PATH INJECTION ---
# This identifies the folder '/mount/src/health_care'
# and tells the Python interpreter: "Look here for the 'src' folder"
current_dir = Path(__file__).resolve().parent
root_path = current_dir.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# --- NOW DO THE IMPORTS ---
import streamlit as st
import pandas as pd
import joblib

# These will now work because we manually added the root to the system path
try:
    from src.data_processor import load_and_merge_data
    from src.model import train_model
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info(f"System Path: {sys.path}") # This helps debug if it fails again

# --- APP LOGIC ---
st.set_page_config(page_title="Health Equity Dashboard", layout="wide")

try:
    # 1. Load Data
    df, report = load_and_merge_data()

    # 2. Handle Model Training
    model_path = root_path / "models" / "cost_predictor.pkl"
    if not model_path.exists():
        with st.spinner("Training Predictive Engine..."):
            train_model(df)
    
    model = joblib.load(model_path)

    st.title("🏥 Health Equity Insights Dashboard")
    st.success("System Online: Analytics and Model Loaded.")
    
    # 3. Display Results
    st.subheader("Vertical Equity Analysis (Income vs. City)")
    st.dataframe(report)

except Exception as e:
    st.error("Dashboard Initialization Error")
    st.exception(e)
